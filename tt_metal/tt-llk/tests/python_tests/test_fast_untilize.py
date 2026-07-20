# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
    Fast-untilize LLK test (tt-metal#42048 + #42049).

Pipeline: fast_untilize unpack -> dedicated fast_untilize math -> fast_untilize
pack. Output: row-major strip.

Hardcoded: unit_dim={4,2,3} for regular streams; compressed BFP inputs unpack
one tile at a time. num_faces=4.
Focused goal: silicon-validate the fast-untilize LLK path against golden.
"""

import struct

import pytest
import torch
from fast_untilize_common import (
    FAST_UNTILIZE_DEST_SYNC_MODES,
    FAST_UNTILIZE_DIMS,
    FAST_UNTILIZE_FACE_C,
    FAST_UNTILIZE_FACE_R,
    FAST_UNTILIZE_NUM_FACES,
    FAST_UNTILIZE_TILE_C,
    FAST_UNTILIZE_TILE_FACE_ROWS,
    FAST_UNTILIZE_TILE_R,
    fast_untilize_dest_acc_modes,
    fast_untilize_formats,
)
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_GUARD_TILES,
    PERF_RUN_TYPE,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import read_from_device


def generate_tile_face_row_ids(tile_count, dtype=torch.bfloat16):
    values = []
    for tile in range(tile_count):
        for face in range(FAST_UNTILIZE_NUM_FACES):
            for row in range(FAST_UNTILIZE_FACE_R):
                value = (
                    tile * FAST_UNTILIZE_TILE_FACE_ROWS
                    + face * FAST_UNTILIZE_FACE_R
                    + row
                    + 1
                )
                values.extend([value] * FAST_UNTILIZE_FACE_C)
    return torch.tensor(values, dtype=dtype)


def make_fast_untilize_test_config(
    formats,
    input_dimensions,
    tile_count,
    src_A,
    tile_cnt_A,
    src_B,
    tile_cnt_B,
    dest_acc,
    dest_sync,
    perf_run_type,
    tile_count_res=None,
    guard_tiles=0,
):
    tile_count_res = tile_count if tile_count_res is None else tile_count_res
    return TestConfig(
        "sources/fast_untilize_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            PERF_RUN_TYPE(perf_run_type),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_GUARD_TILES(guard_tiles),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count_res,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )


def _run_fast_untilize_correctness(
    formats, dest_acc, dimensions, dest_sync, stimulus_kind, allocate_src_b=True
):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    assert (
        input_width_tiles >= 2
    ), "BH fast_untilize supports ct>=2; ct=1 uses the standard fallback path"

    input_dimensions = [
        input_height_tiles * FAST_UNTILIZE_TILE_R,
        input_width_tiles * FAST_UNTILIZE_TILE_C,
    ]
    tile_count = input_height_tiles * input_width_tiles

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    if stimulus_kind == "row_id":
        src_A = generate_tile_face_row_ids(
            tile_count, dtype=format_dict[formats.input_format]
        )
    if not allocate_src_b:
        src_B = torch.empty(0, dtype=format_dict[formats.input_format_B])
        tile_cnt_B = 0

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
    )

    configuration = make_fast_untilize_test_config(
        formats,
        input_dimensions,
        tile_count,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
        dest_acc,
        dest_sync,
        PerfRunType.L1_TO_L1,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length {len(res_from_L1)} != golden length {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Float32 uses the current SrcA/SrcB route, which narrows L1 Float32 input
    # to Tf32 before math. Validate tolerance here; bit-exact Float32->Float32
    # needs a future unpack-to-DEST fast-untilize design.
    is_exact_format = formats.output_format != DataFormat.Float32
    if not is_exact_format:
        assert passed_test(golden_tensor, res_tensor, formats.output_format)
        return

    mismatches = torch.nonzero(res_tensor != golden_tensor, as_tuple=False).flatten()
    if mismatches.numel() > 0:
        idx = int(mismatches[0])
        context_start = max(idx - 8, 0)
        context_end = min(idx + 8, len(res_tensor))

        def row_chunks(tensor, row):
            row_start = row * input_dimensions[1]
            return [
                float(tensor[row_start + i].item())
                for i in range(0, input_dimensions[1], 16)
            ]

        rows = range(14, 20)
        result_rows = {row: row_chunks(res_tensor, row) for row in rows}
        golden_rows = {row: row_chunks(golden_tensor, row) for row in rows}
        pytest.fail(
            f"fast_untilize output mismatch at index {idx}: "
            f"dims={dimensions} dest_sync={dest_sync} "
            f"result={res_tensor[idx].item()} golden={golden_tensor[idx].item()} "
            f"row0={row_chunks(res_tensor, 0)} row0_golden={row_chunks(golden_tensor, 0)} "
            f"result_rows={result_rows} golden_rows={golden_rows} "
            f"result_context={res_tensor[context_start:context_end].tolist()} "
            f"golden_context={golden_tensor[context_start:context_end].tolist()}"
        )


@parametrize(
    formats=fast_untilize_formats(),
    dest_acc=fast_untilize_dest_acc_modes,
    dimensions=FAST_UNTILIZE_DIMS,
    dest_sync=FAST_UNTILIZE_DEST_SYNC_MODES,
    stimulus_kind=["row_id", "random"],
)
def test_fast_untilize(formats, dest_acc, dimensions, dest_sync, stimulus_kind):
    _run_fast_untilize_correctness(
        formats, dest_acc, dimensions, dest_sync, stimulus_kind
    )


def test_fast_untilize_wide_full_ct_dim_133_bf16():
    _run_fast_untilize_correctness(
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        (1, 133),
        DestSync.Half,
        "row_id",
        allocate_src_b=False,
    )


def test_fast_untilize_wide_full_ct_dim_256_bf16():
    _run_fast_untilize_correctness(
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        (1, 256),
        DestSync.Half,
        "row_id",
        allocate_src_b=False,
    )


# fp32 output doubles the byte row-stride vs bf16, so the carried output-Y
# window (256 KiB) is reached at half the row width. The phase-2 rebase fires at
# ct >= 67; the row still fits inside a single 16-row phase (15 * stride < window)
# up to ct = 136, so this width is in the rebase-fixable band.
def test_fast_untilize_wide_full_ct_dim_133_fp32():
    _run_fast_untilize_correctness(
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
        (1, 133),
        DestSync.Half,
        "row_id",
        allocate_src_b=False,
    )


# At ct = 137 a single 16-row phase overflows the carried output-Y window
# (15 * output_row_stride >= 256 KiB), so the strided pack splits each phase into
# sub-runs (rows_per_run = 8 here) and rebases the L1 base per sub-run. Without
# the sub-run rebase this corrupts on silicon (golden 8768 vs 13073.57).
def test_fast_untilize_wide_full_ct_dim_137_fp32():
    _run_fast_untilize_correctness(
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
        (1, 137),
        DestSync.Half,
        "row_id",
        allocate_src_b=False,
    )


@parametrize(
    formats=fast_untilize_formats(),
    dest_acc=fast_untilize_dest_acc_modes,
    dimensions=FAST_UNTILIZE_DIMS,
    dest_sync=FAST_UNTILIZE_DEST_SYNC_MODES,
    perf_run_type=[PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE],
)
def test_fast_untilize_overflow_guard(
    formats, dest_acc, dimensions, dest_sync, perf_run_type
):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    input_dimensions = [
        input_height_tiles * FAST_UNTILIZE_TILE_R,
        input_width_tiles * FAST_UNTILIZE_TILE_C,
    ]
    tile_count = input_height_tiles * input_width_tiles
    guard_tiles = 5

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.constant(0.5),
        spec_B=StimuliSpec.constant(0.5),
    )

    configuration = make_fast_untilize_test_config(
        formats,
        input_dimensions,
        tile_count,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
        dest_acc,
        dest_sync,
        perf_run_type,
        tile_count_res=tile_count + guard_tiles,
        guard_tiles=guard_tiles,
    )

    configuration.run().result

    stim = configuration.variant_stimuli
    last_guard_addr = (
        stim.buf_res_addr + (tile_count + guard_tiles - 1) * stim.buf_res_tile_size
    )
    raw = read_from_device(TestConfig.TENSIX_LOCATION, last_guard_addr, num_bytes=10)
    marker = struct.unpack_from("<H", raw, 0)[0]
    assert marker == 0x4680, f"Sentinel marker missing; got 0x{marker:04x}"

    for g in range(guard_tiles - 1):
        corrupted = struct.unpack_from("<H", raw, (g + 1) * 2)[0]
        assert (
            corrupted == 0
        ), f"L1 overflow: Guard[{g}] has {corrupted} corrupted uint16 words (dims={dimensions}, dest_sync={dest_sync}, run_type={perf_run_type})"
