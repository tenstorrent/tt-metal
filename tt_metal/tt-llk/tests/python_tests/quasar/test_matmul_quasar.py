# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import data_formats
from helpers.device import BootMode
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIM,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PERF_RUN_TYPE,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

kt_dims = [1, 2, 4]


def matmul_math_fidelities(format):
    # Integer matmul is LoFi-only on Quasar.
    return (
        [MathFidelity.LoFi]
        if format.input_format == DataFormat.Int8
        else [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]
    )


def matmul_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def matmul_dest_acc_modes(format):
    return (
        [DestAccumulation.Yes]
        if format.input_format == DataFormat.Int8
        else [DestAccumulation.Yes, DestAccumulation.No]
    )


def matmul_dimensions(dest_acc, dest_sync, *, exact_dest_fill=False):
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )
    return [
        ([mt_dim * TILE_DIM, kt_dim * TILE_DIM], [kt_dim * TILE_DIM, nt_dim * TILE_DIM])
        for mt_dim in range(1, max_tiles + 1)
        if not exact_dest_fill or max_tiles % mt_dim == 0
        for nt_dim in (
            [max_tiles // mt_dim]
            if exact_dest_fill
            else range(1, max_tiles // mt_dim + 1)
        )
        for kt_dim in kt_dims
    ]


def matmul_implied_math_formats(format, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if format.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def matmul_register_format_hints(format):
    return (
        [DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B]
        # MxFp4_2x is Quasar only. Quasar Architecture derivations don't support it.
        if format.input_format == DataFormat.MxFp4 and _ARCH == ChipArchitecture.QUASAR
        else [None]
    )


def matmul_enable_direct_indexing(register_format_hint):
    return [False] if register_format_hint is None else [True, False]


# Generate format-aware combinations. MxFp4 is an input-only (L1) format here: the
# unpacker produces MxFp4_2x_A/B in the src registers, so drop the cross-product
# entries where MxFp4 would land as an output.
MATMUL_FORMAT = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
) + [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)]

_ARCH = get_chip_architecture()


@pytest.mark.quasar
@parametrize(
    format=MATMUL_FORMAT,
    math_fidelity=matmul_math_fidelities,
    dest_sync_mode=lambda: matmul_dest_sync_modes(is_perf=False),
    dest_acc=matmul_dest_acc_modes,
    dimensions=runtime(
        lambda dest_acc, dest_sync_mode: matmul_dimensions(
            dest_acc,
            dest_sync_mode,
        )
    ),
    implied_math_format=lambda format: matmul_implied_math_formats(format),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    transpose=[Transpose.No],
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
# Note: this test is used to test boot modes, that is why it has them piped as default arguments to the test itself
def test_matmul(
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    format,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    boot_mode=BootMode.TRISC,
    *,
    is_perf=False,
    perf_report=None,
):

    # Reassign format with register_format_hint so that test config generation and stimulus generation are aware of the register format hint.
    format = InputOutputFormat(
        format.input_format,
        format.output_format,
        input_format_B=format.input_format_B,
        register_format_hint=register_format_hint,
    )

    input_A_dimensions, input_B_dimensions = dimensions

    if format.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=format.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=format.input_format
    )

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    if not is_perf:
        torch_format = format_dict[format.output_format]
        src_A_golden = src_A
        src_B_golden = src_B
        if format.input_format.is_mx_format():
            tilized_A_golden = quantize_mx_tensor_chunked(
                tilized_A.flatten().to(torch.bfloat16), format.input_format
            ).reshape(tilized_A.shape)
            tilized_B_golden = quantize_mx_tensor_chunked(
                tilized_B.flatten().to(torch.bfloat16), format.input_format
            ).reshape(tilized_B.shape)
            src_A_golden = untilize_block(
                tilized_A_golden,
                stimuli_format=format.input_format,
                dimensions=input_A_dimensions,
            )
            src_B_golden = untilize_block(
                tilized_B_golden,
                stimuli_format=format.input_format,
                dimensions=input_B_dimensions,
            )

        if transpose == Transpose.Yes:
            t_matrix = get_golden_generator(TransposeGolden)

            src_B_golden = t_matrix.transpose_faces_multi_tile(
                src_B_golden,
                format.input_format,
                num_tiles=tile_cnt_B,
                tilize=True,
                input_dimensions=input_B_dimensions,
            )
            src_B_golden = t_matrix.transpose_within_faces_multi_tile(
                src_B_golden,
                format.input_format,
                num_tiles=tile_cnt_B,
                untilize=True,
                input_dimensions=input_B_dimensions,
            )

        formats_config = data_formats(
            input_format=format.input_format,
            input_format_B=format.input_format_B,
            output_format=format.output_format,
            is_fp32_dest_acc_en=dest_acc,
            num_iterations=1,
            unpacking_to_dest=False,
            # 2x register-format opt-in needs to flow through inference; only disable
            # for plain MX formats where there's nothing to infer.
            disable_format_inference=(
                format.input_format.is_mx_format()
                and format.register_format_hint is None
            ),
            register_format_hint=format.register_format_hint,
        )[0]
        pack_src_format = formats_config.pack_src

        generate_golden = get_golden_generator(MatmulGolden)
        golden_tensor = generate_golden(
            src_A_golden,
            src_B_golden,
            format.output_format,
            math_fidelity,
            input_A_dimensions=input_A_dimensions,
            input_B_dimensions=input_B_dimensions,
            tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
            input_A_format=format.input_format,
            input_B_format=format.input_format,
            math_format=pack_src_format,  # For accumulation of results in matmul we require to calculate in pack_src_format.
            dest_acc=dest_acc,
        )

    num_faces = 4

    templates = [
        MATH_FIDELITY(math_fidelity),
        IMPLIED_MATH_FORMAT(implied_math_format),
        ENABLE_2X_FORMAT(
            format.register_format_hint
            in (DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B)
        ),
        ENABLE_DIRECT_INDEXING(enable_direct_indexing),
        DEST_SYNC(dest_sync_mode),
        UNPACK_TRANS_FACES(transpose),
    ]
    runtimes = [
        CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
        TILE_COUNT(matmul_dims.output_tile_cnt),
        NUM_FACES(num_faces, num_faces, num_faces),
        LOOP_FACTOR(loop_factor),
    ]
    variant_stimuli = StimuliConfig(
        tilized_A.flatten(),
        format.input_format,
        tilized_B.flatten(),
        format.input_format,
        format.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=matmul_dims.output_tile_cnt,
        num_faces=num_faces,
    )
    disable_format_inference = (
        format.input_format.is_mx_format() and format.register_format_hint is None
    )

    if is_perf:
        if perf_report is None:
            raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/matmul_quasar_test.cpp",
        "formats": format,
        "templates": templates,
        "runtimes": runtimes,
        "variant_stimuli": variant_stimuli,
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
        "disable_format_inference": disable_format_inference,
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    # The shared source keys off PERF_RUN_TYPE. PerfConfig injects it per run type;
    # the functional path runs the plain L1-to-L1 kernel, so supply it explicitly.
    configuration = TestConfig(
        boot_mode=boot_mode,
        **{
            **test_config_kwargs,
            "templates": templates + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )
    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # For MX outputs, model the packer: quantize the golden onto the MX lattice (from the
    # math/pack_src format the result was produced in) so the comparison validates the
    # device's MX output quantization, not just matmul-math-to-MX-precision. The lattice-
    # aware compare in passed_test then supplies the small HW-vs-reference rounding slack.
    if format.output_format.is_mx_format():
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(format_dict[pack_src_format]), format.output_format
        ).to(torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        format.output_format,
    ), "Assert against golden failed"
