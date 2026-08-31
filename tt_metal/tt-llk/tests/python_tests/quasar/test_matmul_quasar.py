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
    format_tile_sizes,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    IMPLIED_MATH_FORMAT,
    IN_FACE_DIMS,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tile_constants import calculate_tile_size_bytes
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

kt_dims = [
    1,
    2,
    4,
]
PERF_KT_DIMS = [1, 4]


class IndependentMatmulStimuliConfig(StimuliConfig):
    """Dense L1 layout with independent A, B, and result tile shapes."""

    def __init__(
        self, *args, input_a_tile_dimensions, input_b_tile_dimensions, **kwargs
    ):
        self.input_a_tile_dimensions = input_a_tile_dimensions
        self.input_b_tile_dimensions = input_b_tile_dimensions
        super().__init__(*args, **kwargs)

    def _calculate_tile_sizes(self):
        super()._calculate_tile_sizes()
        self.tile_size_A_bytes = calculate_tile_size_bytes(
            self.stimuli_A_format,
            self.input_a_tile_dimensions,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("A"),
        )
        self.tile_size_B_bytes = calculate_tile_size_bytes(
            self.stimuli_B_format,
            self.input_b_tile_dimensions,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("B"),
        )
        self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A
        self.buf_res_addr = self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B

    def _write_dense_tile_dimensions(self, location="0,0"):
        for (
            buffer,
            tile_count,
            data_format,
            address,
            tile_size,
            tile_dimensions,
        ) in (
            (
                self.buffer_A,
                self.tile_count_A,
                self.stimuli_A_format,
                self.buf_a_addr,
                self.tile_size_A_bytes,
                self.input_a_tile_dimensions,
            ),
            (
                self.buffer_B,
                self.tile_count_B,
                self.stimuli_B_format,
                self.buf_b_addr,
                self.tile_size_B_bytes,
                self.input_b_tile_dimensions,
            ),
        ):
            tile_shape = construct_tile_shape(tile_dimensions)
            StimuliConfig.write_matrix_w_tile_dimensions(
                buffer,
                tile_count,
                StimuliConfig.get_packer(data_format),
                address,
                tile_size,
                tile_shape.total_num_faces(),
                tile_shape.face_r_dim,
                tile_dimensions,
                location,
            )


def matmul_math_fidelities(format, *, is_perf=False):
    # Integer matmul is LoFi-only on Quasar. MX is already full precision at LoFi,
    # so perf skips the extra HiFi phases.
    if format.input_format == DataFormat.Int8 or (
        is_perf and format.input_format.is_mx_format()
    ):
        return [MathFidelity.LoFi]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def matmul_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def matmul_dest_acc_modes(format):
    return (
        [DestAccumulation.Yes]
        if format.input_format == DataFormat.Int8
        else [DestAccumulation.Yes, DestAccumulation.No]
    )


def matmul_dimensions(dest_acc, dest_sync, *, exact_dest_fill=False, is_perf=False):
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )
    # Perf keeps dest-full tall (max_tiles, 1) and wide (1, max_tiles) so both
    # ct>=rt and ct<rt MOP addr_mod branches are covered, and kt=1 vs kt=4.
    mt_dims = (1, max_tiles) if is_perf else range(1, max_tiles + 1)
    selected_kt_dims = PERF_KT_DIMS if is_perf else kt_dims
    return [
        ([mt_dim * TILE_DIM, kt_dim * TILE_DIM], [kt_dim * TILE_DIM, nt_dim * TILE_DIM])
        for mt_dim in mt_dims
        if not exact_dest_fill or max_tiles % mt_dim == 0
        for nt_dim in (
            [max_tiles // mt_dim]
            if exact_dest_fill
            else range(1, max_tiles // mt_dim + 1)
        )
        for kt_dim in selected_kt_dims
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
    format=[
        format
        for format in MATMUL_FORMAT
        if format.input_format == DataFormat.Float16_b
    ],
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=runtime(
        lambda dest_acc, dest_sync_mode: matmul_dimensions(
            dest_acc,
            dest_sync_mode,
        )
    ),
    implied_math_format=[ImpliedMathFormat.No],
    register_format_hint=[None],
    enable_direct_indexing=[False],
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
    input_tile_dimensions=None,
    matmul_tile_dims=None,
):

    # Reassign format with register_format_hint so that test config generation and stimulus generation are aware of the register format hint.
    format = InputOutputFormat(
        format.input_format,
        format.output_format,
        input_format_B=format.input_format_B,
        register_format_hint=register_format_hint,
    )

    input_A_dimensions, input_B_dimensions = dimensions
    tiny_tiles = input_tile_dimensions is not None

    if format.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    if tiny_tiles:
        input_a_tile_dimensions, input_b_tile_dimensions = input_tile_dimensions
        src_A, tile_cnt_A, _, _ = generate_stimuli(
            stimuli_format_A=format.input_format,
            input_dimensions_A=input_A_dimensions,
            stimuli_format_B=format.input_format,
            input_dimensions_B=input_A_dimensions,
            spec_A=stimuli_spec,
            spec_B=stimuli_spec,
            tile_dimensions=input_a_tile_dimensions,
            output_format=format.output_format,
        )
        src_B, tile_cnt_B, _, _ = generate_stimuli(
            stimuli_format_A=format.input_format,
            input_dimensions_A=input_B_dimensions,
            stimuli_format_B=format.input_format,
            input_dimensions_B=input_B_dimensions,
            spec_A=stimuli_spec,
            spec_B=stimuli_spec,
            tile_dimensions=input_b_tile_dimensions,
            output_format=format.output_format,
        )
        tilized_A = tilize_block(
            src_A,
            dimensions=input_A_dimensions,
            stimuli_format=format.input_format,
            tile_dimensions=input_a_tile_dimensions,
        )
        tilized_B = tilize_block(
            src_B,
            dimensions=input_B_dimensions,
            stimuli_format=format.input_format,
            tile_dimensions=input_b_tile_dimensions,
        )
        ct_dim, rt_dim, kt_dim = matmul_tile_dims
        output_tile_dimensions = (
            input_a_tile_dimensions[0],
            input_b_tile_dimensions[1],
        )
        output_tile_cnt = rt_dim * ct_dim
    else:
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
        matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))
        ct_dim, rt_dim, kt_dim = (
            matmul_dims.ct_dim,
            matmul_dims.rt_dim,
            matmul_dims.kt_dim,
        )
        output_tile_dimensions = (TILE_DIM, TILE_DIM)
        output_tile_cnt = matmul_dims.output_tile_cnt

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
            tilize=not tiny_tiles,
            input_A_format=format.input_format,
            input_B_format=format.input_format,
            math_format=pack_src_format,  # For accumulation of results in matmul we require to calculate in pack_src_format.
            dest_acc=dest_acc,
        )
        if tiny_tiles:
            golden_tensor = tilize_block(
                golden_tensor,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=format.output_format,
                tile_dimensions=output_tile_dimensions,
            ).flatten()

    input_a_shape = construct_tile_shape(
        input_tile_dimensions[0] if tiny_tiles else (TILE_DIM, TILE_DIM)
    )
    input_b_shape = construct_tile_shape(
        input_tile_dimensions[1] if tiny_tiles else (TILE_DIM, TILE_DIM)
    )
    output_shape = construct_tile_shape(output_tile_dimensions)
    enable_2x_format = format.register_format_hint in (
        DataFormat.MxFp4_2x_A,
        DataFormat.MxFp4_2x_B,
    )

    templates = [
        MATH_FIDELITY(math_fidelity),
        IMPLIED_MATH_FORMAT(implied_math_format),
        ENABLE_2X_FORMAT(enable_2x_format),
        ENABLE_DIRECT_INDEXING(enable_direct_indexing),
        DEST_SYNC(dest_sync_mode),
        UNPACK_TRANS_FACES(transpose),
    ]
    runtimes = [
        CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
        TILE_COUNT(output_tile_cnt * kt_dim),
        NUM_FACES(
            output_shape.total_num_faces(),
            input_a_shape.total_num_faces(),
            input_b_shape.total_num_faces(),
        ),
        IN_TILE_DIMS(
            *input_a_shape.tile_dims,
            *input_b_shape.tile_dims,
        ),
        IN_FACE_DIMS(
            input_a_shape.face_r_dim,
            input_a_shape.face_c_dim,
            input_b_shape.face_r_dim,
            input_b_shape.face_c_dim,
        ),
        NUM_FACES_R_DIM(
            input_a_shape.num_faces_r_dim,
            input_b_shape.num_faces_r_dim,
        ),
        NUM_FACES_C_DIM(
            input_a_shape.num_faces_c_dim,
            input_b_shape.num_faces_c_dim,
        ),
        LOOP_FACTOR(loop_factor),
    ]
    stimuli_kwargs = {
        "tile_count_A": tile_cnt_A,
        "tile_count_B": tile_cnt_B,
        "tile_count_res": output_tile_cnt,
        "num_faces": output_shape.total_num_faces(),
        "face_r_dim": output_shape.face_r_dim,
        "tile_dimensions": output_tile_dimensions,
        "use_dense_tile_dimensions": tiny_tiles,
    }
    if tiny_tiles:
        variant_stimuli = IndependentMatmulStimuliConfig(
            tilized_A.flatten(),
            format.input_format,
            tilized_B.flatten(),
            format.input_format,
            format.output_format,
            input_a_tile_dimensions=input_a_shape.tile_dims,
            input_b_tile_dimensions=input_b_shape.tile_dims,
            **stimuli_kwargs,
        )
    else:
        variant_stimuli = StimuliConfig(
            tilized_A.flatten(),
            format.input_format,
            tilized_B.flatten(),
            format.input_format,
            format.output_format,
            **stimuli_kwargs,
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

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
        boot_mode=boot_mode,
    )
    if is_perf:
        configuration.run(perf_report)
        return

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


TINY_MATMUL_FORMAT = [
    format for format in MATMUL_FORMAT if format.input_format == DataFormat.Float16_b
]
TINY_MATMUL_SHAPE_CASES = [((16, 16), (16, 16))] + [
    ((height, 32), (32, width)) for height in (1, 2, 4, 8, 16, 32) for width in (16, 32)
]


def tiny_matmul_tile_dimensions(
    dest_acc, dest_sync_mode, *, exact_dest_fill=False, is_perf=False
):
    return [
        (
            input_b_dimensions[1] // TILE_DIM,
            input_a_dimensions[0] // TILE_DIM,
            input_a_dimensions[1] // TILE_DIM,
        )
        for input_a_dimensions, input_b_dimensions in matmul_dimensions(
            dest_acc,
            dest_sync_mode,
            exact_dest_fill=exact_dest_fill,
            is_perf=is_perf,
        )
    ]


def run_tiny_matmul(
    input_tile_dimensions,
    matmul_tile_dims,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    format,
    implied_math_format,
    transpose,
    run_types,
    loop_factor,
    *,
    register_format_hint=None,
    enable_direct_indexing=False,
    is_perf=False,
    perf_report=None,
):
    input_a_tile_dimensions, input_b_tile_dimensions = input_tile_dimensions
    ct_dim, rt_dim, kt_dim = matmul_tile_dims
    input_A_dimensions = [
        rt_dim * input_a_tile_dimensions[0],
        kt_dim * input_a_tile_dimensions[1],
    ]
    input_B_dimensions = [
        kt_dim * input_b_tile_dimensions[0],
        ct_dim * input_b_tile_dimensions[1],
    ]
    test_matmul(
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        (input_A_dimensions, input_B_dimensions),
        format,
        implied_math_format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types,
        loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        input_tile_dimensions=input_tile_dimensions,
        matmul_tile_dims=matmul_tile_dims,
    )


@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(TINY_MATMUL_SHAPE_CASES),
    format=TINY_MATMUL_FORMAT,
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    matmul_tile_dims=runtime(
        lambda dest_acc, dest_sync_mode: tiny_matmul_tile_dimensions(
            dest_acc, dest_sync_mode
        )
    ),
    implied_math_format=[ImpliedMathFormat.No],
    transpose=[Transpose.No],
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_matmul_tiny(
    input_tile_dimensions,
    format,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    matmul_tile_dims,
    implied_math_format,
    transpose,
    run_types,
    loop_factor,
):
    run_tiny_matmul(
        input_tile_dimensions,
        matmul_tile_dims,
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        format,
        implied_math_format,
        transpose,
        run_types,
        loop_factor,
    )
