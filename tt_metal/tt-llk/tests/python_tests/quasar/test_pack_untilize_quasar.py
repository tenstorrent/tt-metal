# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test


def pack_untilize_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def pack_untilize_input_dimensions(*, is_perf=False):
    return [32, 32] if is_perf else None


def generate_pack_untilize_combinations(
    formats_list: List[FormatConfig],
    *,
    is_perf=False,
):
    """
    Generate pack_untilize combinations.

    Args:
        formats_list: List of input-output format pairs

    Returns: List of (format, dest_acc, dest_sync, input_dimensions) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        # If input format is Int16, output format must also be Int16, and vice versa
        if (in_fmt == DataFormat.Int16) ^ (out_fmt == DataFormat.Int16):
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        # Int16 requires 16bit mode dest register
        if in_fmt == DataFormat.Int16:
            return (DestAccumulation.No,)
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    dimensions_cache = (
        None
        if is_perf
        else {
            (dest_acc, dest_sync): tuple(
                generate_unary_input_dimensions(dest_acc, dest_sync)
            )
            for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
            for dest_sync in (DestSync.Half, DestSync.Full)
        }
    )

    dest_sync_modes = pack_untilize_dest_sync_modes(is_perf=is_perf)
    perf_dimensions = pack_untilize_input_dimensions(is_perf=is_perf)
    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        # MX as output format produces flaky results on Quasar.
        if out_fmt.is_mx_format():
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            for dest_sync in dest_sync_modes:
                dimensions_list = (
                    [perf_dimensions]
                    if is_perf
                    else dimensions_cache[(dest_acc, dest_sync)]
                )
                for dimensions in dimensions_list:
                    combinations.append((fmt, dest_acc, dest_sync, dimensions))

    return combinations


PACK_UNTILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Int16,
        DataFormat.Int32,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
)
ALL_PACK_UNTILIZE_COMBINATIONS = generate_pack_untilize_combinations(
    PACK_UNTILIZE_FORMATS
)
PERF_PACK_UNTILIZE_COMBINATIONS = generate_pack_untilize_combinations(
    PACK_UNTILIZE_FORMATS,
    is_perf=True,
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_dimensions=ALL_PACK_UNTILIZE_COMBINATIONS,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_pack_untilize_quasar(
    formats_dest_acc_sync_dimensions,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    (formats, dest_acc, dest_sync_mode, input_dimensions) = (
        formats_dest_acc_sync_dimensions
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    if not is_perf:
        golden_tensor = generate_golden(
            src_A,
            formats.output_format,
            input_dimensions,
            input_format=formats.input_format,
        )

    num_faces = 4

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/pack_untilize_quasar_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
        ],
        "runtimes": [
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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
            num_faces=num_faces,
        ),
        "unpack_to_dest": (
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        "dest_acc": dest_acc,
        "disable_format_inference": (
            formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
        ),
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

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"
