# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test

TILE_ELEMS = 32 * 32
FACE_ELEMS = 16 * 16

BINARY_BROADCAST_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
) + [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)]


def binary_broadcast_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def binary_broadcast_implied_math_formats(format, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if format.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def binary_broadcast_math_fidelities(format, mathop, *, is_perf=False):
    if format.input_format == DataFormat.Int8:
        return [MathFidelity.LoFi]
    if is_perf:
        return [MathFidelity.LoFi]
    return get_valid_math_fidelities(format, mathop)


def binary_broadcast_input_dimensions(dest_acc, dest_sync_mode, *, is_perf=False):
    if is_perf:
        # Nested list: parametrize treats a flat list as multiple values, so
        # [32, 32] would become input_dimensions=32 (int) and break generate_stimuli.
        return [[32, 32]]
    return generate_unary_input_dimensions(dest_acc, dest_sync_mode)


@pytest.mark.quasar
@parametrize(
    formats=BINARY_BROADCAST_FORMATS,
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    broadcast_type=[
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    math_fidelity=lambda formats, mathop: binary_broadcast_math_fidelities(
        formats, mathop
    ),
    implied_math_format=lambda formats: binary_broadcast_implied_math_formats(formats),
    dest_sync_mode=lambda: binary_broadcast_dest_sync_modes(is_perf=False),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode: binary_broadcast_input_dimensions(
            dest_acc, dest_sync_mode, is_perf=False
        )
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_eltwise_binary_broadcast_quasar(
    formats,
    dest_acc,
    mathop,
    broadcast_type,
    math_fidelity,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):

    if formats.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
    )

    generate_broadcast_golden = get_golden_generator(BroadcastGolden)
    bcast_src_B_tensor = generate_broadcast_golden(
        broadcast_type,
        src_B,
        formats.output_format,
        num_faces=4,
        tile_cnt=tile_cnt_A,
        face_r_dim=16,
        input_format=formats.input_format,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    input_format = formats.input_format
    input_format_B = (
        DataFormat.Float16_b
        if formats.input_format.is_mx_format()
        else formats.input_format
    )
    golden_tensor = generate_golden(
        mathop,
        src_A,
        bcast_src_B_tensor,
        formats.output_format,
        math_fidelity,
        input_format=input_format,
        input_format_B=input_format_B,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_binary_broadcast_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=4,
        ),
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
        "boot_mode": boot_mode,
        "disable_format_inference": formats.input_format.is_mx_format(),
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )
    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
