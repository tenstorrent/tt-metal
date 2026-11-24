# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
topk_xl split sort (Blackhole): a 2048-element row sorted as two 1024 halves
(ckernel_sfpu_topk_xl_halves.h) or four 512 quarters (ckernel_sfpu_topk_xl_quarters.h),
then combined into one list in the stock topk_xl 2048 order.

Input:  bf16 values, part p in input tile p (a quarter fills the first 512 elements).
Stamp:  the element's index in the row, with the core id in bits [15:11], ORed into the
        low 16 bits of its fused Dest word.
Output: raw uint32 Dest words. Combined: the row's 2048 words in Dest tiles 0 and 1.
        Local sorts only: each part's tile.

The golden (TopKXLSortedListGolden) is the sorted list of the fused words placed by the
stock order, so every word, its index and its position are checked.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TopKXLSortedListGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    TILE_COUNT,
    TOPK_XL_PARTS,
)
from test_topk_xl_unfused_macro import TOPK_XL_MACRO_KNOBS

pytestmark = [skip_for_wormhole, skip_for_quasar]

ROW = 2048
ELEMENTS_PER_TILE = 1024
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt32)
BF16_NEG_INF = 0xFF800000  # what the copy writes into a tile's unused lanes
CORE_ID_SHIFT = 11


def _row_hi16(mode: str, seed: int) -> torch.Tensor:
    """The row's 2048 bf16 bit patterns.

    distinct: 1024 values >= 1.0 and 1024 values <= -1.0, shuffled, so the value alone orders the row.
    ties:     nine values, negatives among them, so the index bits order most of the row.
    """
    gen = torch.Generator().manual_seed(seed)
    if mode == "distinct":
        hi16 = torch.cat(
            [0x3F80 + torch.arange(ROW // 2), 0xBF80 + torch.arange(ROW // 2)]
        )
        return hi16[torch.randperm(ROW, generator=gen)]
    if mode == "ties":
        palette = (
            torch.tensor(
                [-4.0, -2.0, -1.5, -0.5, 0.25, 1.0, 1.5, 3.0, 8.0], dtype=torch.bfloat16
            )
            .view(torch.int16)
            .to(torch.int64)
            & 0xFFFF
        )
        return palette[torch.randint(len(palette), (ROW,), generator=gen)]
    raise ValueError(f"unknown mode {mode}")


def _run(
    parts,
    combine,
    mode="distinct",
    core_id=0,
    ascending=False,
    half0_ascending=True,
    quarter0_ascending=True,
    extra_templates=(),
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Build and run one variant. Returns (packed words, the row's fused words in row order)."""
    part_len = ROW // parts
    hi16 = _row_hi16(mode, seed=parts)
    values = (hi16 << 16).to(torch.int32).view(torch.float32)

    src_A = torch.zeros(parts * ELEMENTS_PER_TILE, dtype=torch.float32)
    for p in range(parts):
        start = p * ELEMENTS_PER_TILE
        src_A[start : start + part_len] = values[p * part_len : (p + 1) * part_len]
    src_A = src_A.to(torch.bfloat16)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[FORMATS.input_format])
    result_tiles = 2 if combine else parts

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/topk_xl_parts_test.cpp",
            formats=FORMATS,
            templates=[
                DEST_SYNC(DestSync.Full),  # Dest tiles 0..7, fp32
                TOPK_XL_PARTS(
                    parts=parts,
                    parts_combine=combine,
                    parts_ascending=ascending,
                    half0_ascending=half0_ascending,
                    quarter0_ascending=quarter0_ascending,
                    parts_core_id=core_id,
                ),
                *extra_templates,
            ],
            runtimes=[TILE_COUNT(tile_cnt=parts)],
            variant_stimuli=StimuliConfig(
                src_A,
                FORMATS.input_format,
                src_B,
                FORMATS.input_format,
                FORMATS.output_format,
                tile_count_A=parts,
                tile_count_B=1,
                tile_count_res=result_tiles,
            ),
            dest_acc=DestAccumulation.Yes,  # 32-bit fused value|index words in Dest
        ),
    )
    if is_perf:
        return None, None
    words = (hi16 << 16) | (core_id << CORE_ID_SHIFT) | torch.arange(ROW)
    result = torch.tensor(outcome.result, dtype=format_dict[FORMATS.output_format])
    return result.to(torch.int64) & 0xFFFFFFFF, words


def _assert_image(got, expected, what):
    differ = (got != expected).nonzero().flatten()
    assert differ.numel() == 0, (
        f"{what}: {differ.numel()} of {expected.numel()} Dest words differ from the golden, first at "
        f"{differ[:8].tolist()}: got {[hex(int(got[i])) for i in differ[:4]]}, "
        f"want {[hex(int(expected[i])) for i in differ[:4]]}"
    )


def _check_combined(result, words, ascending):
    expected = get_golden_generator(TopKXLSortedListGolden)(words, ROW, ascending)
    _assert_image(
        result, expected, f"combined row ({'ascending' if ascending else 'descending'})"
    )


def _check_parts(result, words, parts, half0_ascending, quarter0_ascending):
    golden = get_golden_generator(TopKXLSortedListGolden)
    part_len = ROW // parts
    for p in range(parts):
        if parts == 2:
            ascending = (p == 0) == half0_ascending
        else:
            ascending = (p % 2 == 0) == quarter0_ascending
        expected = golden(
            words[p * part_len : (p + 1) * part_len],
            part_len,
            ascending,
            fill=BF16_NEG_INF,
        )
        got = result[p * ELEMENTS_PER_TILE : (p + 1) * ELEMENTS_PER_TILE]
        _assert_image(
            got,
            expected,
            f"part {p} local sort ({'ascending' if ascending else 'descending'})",
        )


@parametrize(parts=[2, 4])
def test_topk_xl_parts_local_sort(parts):
    """The out-of-line local sort and the based index stamp: each part sorted alone, in both directions."""
    (parts,) = parts
    result, words = _run(parts, combine=False, core_id=5)
    _check_parts(result, words, parts, half0_ascending=True, quarter0_ascending=True)


@parametrize(
    parts=[2, 4],
    ascending=[False, True],
    half0_ascending=[False, True],
    quarter0_ascending=lambda parts: [True, False] if parts == 4 else [True],
)
def test_topk_xl_parts_combine(parts, ascending, half0_ascending, quarter0_ascending):
    """Halves or quarters combined into one sorted 2048 list, for each direction of the parts and of the result."""
    result, words = _run(
        parts,
        combine=True,
        ascending=ascending,
        half0_ascending=half0_ascending,
        quarter0_ascending=quarter0_ascending,
    )
    _check_combined(result, words, ascending)


@parametrize(parts=[2, 4], ascending=[False, True])
def test_topk_xl_parts_combine_ties(parts, ascending):
    """Nine distinct values over 2048 elements: the index bits, with core id 31 above them, order equal values."""
    result, words = _run(
        parts, combine=True, mode="ties", core_id=31, ascending=ascending
    )
    _check_combined(result, words, ascending)


@parametrize(parts=[2, 4])
def test_topk_xl_parts_combine_macro_opt_out(parts):
    """The same pipeline built without the fused SFPLOADMACRO step groups (DISABLE_TOPK_XL_SFPLOADMACRO)."""
    (parts,) = parts
    result, words = _run(
        parts, combine=True, extra_templates=[TOPK_XL_MACRO_KNOBS(opt_out=True)]
    )
    _check_combined(result, words, ascending=False)
