# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    UnpackerEngine,
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
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
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


def generate_unpack_tilize_combinations(
    formats_list: List[FormatConfig],
    *,
    is_perf=False,
):
    """
    Generate unpack_tilize combinations.

    Rules:
    1. 32-bit formats require DestAccumulation.Yes

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, unpacker_sel, input_dimensions) tuples
    """
    dimensions_cache = {
        (dest_acc, dest_sync): tuple(
            generate_unary_input_dimensions(dest_acc, dest_sync)
        )
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    combinations = []

    perf_dimensions = [32, 32]
    dest_sync_modes = (DestSync.Half,) if is_perf else (DestSync.Half, DestSync.Full)

    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (
                (DestAccumulation.No,)
                if in_fmt in [DataFormat.Float16, DataFormat.Int16]
                else (DestAccumulation.No, DestAccumulation.Yes)
            )
        )
        # 32-bit tilize uses unpack_to_dest (UNP_DEST)
        unpacker_engines = (
            (UnpackerEngine.UnpDest,)
            if in_fmt.is_32_bit()
            else (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
        )

        if is_perf:
            if in_fmt.is_32_bit():
                continue
            for dest_acc in dest_acc_modes:
                for dest_sync in dest_sync_modes:
                    for unpacker_sel in (UnpackerEngine.UnpA,):
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                unpacker_sel,
                                perf_dimensions,
                            )
                        )
            continue

        for dest_acc in dest_acc_modes:
            for dest_sync in dest_sync_modes:
                for unpacker_sel in unpacker_engines:
                    for dimensions in dimensions_cache[(dest_acc, dest_sync)]:
                        combinations.append(
                            (fmt, dest_acc, dest_sync, unpacker_sel, dimensions)
                        )

    return combinations


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,  # Input format and output format are the same
)
ALL_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS
)
PERF_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS,
    is_perf=True,
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_sel_dimensions=ALL_UNPACK_TILIZE_COMBINATIONS,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_tilize_quasar(
    formats_dest_acc_sync_unpack_sel_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    (formats, dest_acc, dest_sync_mode, unpacker_sel, input_dimensions) = (
        formats_dest_acc_sync_unpack_sel_dimensions
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    if formats.input_format.is_mx_format():
        golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)
    golden_tensor = generate_golden(
        golden_src, input_dimensions, formats.output_format, num_faces=4
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/unpack_tilize_quasar_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),
            NUM_FACES(),
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
        "unpack_to_dest": (
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        "dest_acc": dest_acc,
        "disable_format_inference": formats.input_format.is_mx_format(),
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        boot_mode=boot_mode,
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
