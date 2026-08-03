# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Generic MoE-gate top-k SFPU test.

Covers tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h
and its two implementation halves (_top8.h, _top16.h):
`_init_generic_moe_gate_topk_` / `_generic_moe_gate_topk_<normalize,
num_selected_experts, num_total_experts, zero_tail, full_sort>`.

Semantics (the DeepSeek biased gate)
------------------------------------
* Experts are selected by their BIASED score (DEST tile 2 = score + bias).
* What comes back is the ORIGINAL score (carried as the HI16 payload alongside the
  expert id in LO16) and the expert id.
* With `normalize`, each winner score is rescaled by
  `scale / (sum_of_selected_original_scores + eps)`.

Layout, for num_total_experts == 256
------------------------------------
The 256 experts occupy face 0 of the tile with id = 16 * row + column -- that is what
`_topk_moe_generate_indices_` writes (LTILEID-seeded, stored to the even and odd
column halves of each row band). The winners come back in COLUMN 0 ("lane 0") of DEST
rows 0-7. `_generic_moe_gate_top8_merge_instances_` is a reduce-to-one-lane
rotate-and-merge network, not an all-reduce, so the other even columns are left
holding partial merges (measured: they contain ids outside the top-8) and only lane 0
is meaningful. The top-8 path never writes the odd columns; the top-16 path does
(`_generic_moe_gate_store_16_rows_even_odd_split_` stores both halves), which is why
the odd-column check below is restricted to the top-8 path.

How many winners come back
--------------------------
Eight, for every `num_selected_experts` except 16. `_generic_moe_gate_topk_` routes
only 16 to the top-16 path; every other value routes to top-8, which emits eight
winners regardless of what was asked for. api/compute/experimental/generic_moe_gate.h
documents this and deliberately does not static_assert it, because blaze's
gptoss_moe_router validates 1..16 and forwards straight through. So this test asserts
the top-8 set for num_selected_experts of both 8 and 4 -- for 4 that is the contract
the LLK actually implements, and it is distinct code: with full_sort=False,
`_generic_moe_gate_top8_sort_rows_<4, false>` takes the SFPSWAP + TTI_NOP branch that
num_selected_experts=8 never reaches.

What this test asserts, and why in this form
--------------------------------------------
Row ORDER inside the winner rows is not asserted, because the op does not guarantee
one: with full_sort=True the measured rank order of the eight winners is
[0, 1, 3, 4, 2, 5, 6, 7], i.e. not sorted (see the xfail reasons below, which is where
that matters). Callers consume the winners as a set. So the checks are:
  1. the returned id set == the golden top-8 id set;
  2. every returned score matches the normalized original score of the id it is
     paired with -- this is the check that would catch a payload/key mix-up, which is
     the interesting failure mode for a bitonic sort carrying a payload;
  3. for the top-8 path, the odd columns still hold their generated expert id and
     their staged score, which confirms the store touched only the even columns and
     pins the even/odd reading of the SFPU DEST address.

Stimuli are built so the top-8 boundary is never ambiguous: the biased scores are a
permutation of the 256 exactly-representable bfloat16 values in [1.0, 2.0), so all 256
sort keys are distinct in the format the hardware compares in, and there is no tie to
break.

Scope: the SFPU kernel only. In the compute API the score and biased tiles are staged
by the FPU kernel `_llk_math_deepseek_moe_gate_eltwise_binary_`; here they are staged
by plain datacopy so this test isolates the SFPU half. Cover the FPU kernel
separately.

Configurations left uncovered, with the measurement behind each
---------------------------------------------------------------
* `zero_tail` is pinned False. With zero_tail=True and num_selected_experts=4 the
  kernel blanks rows 4-7 and leaves rows 0-3 holding golden ranks {0, 1, 3, 4} -- the
  wrong top-4, with rank 2 destroyed rather than misplaced. Same root cause as the
  normalize xfail below. Re-enable as an axis once that is fixed.
* `num_total_experts` is pinned 256. At 128 the top-8 path returns golden ranks
  {0, 1, 2, 3, 4, 5, 7, 8} -- rank 6 dropped, rank 8 substituted -- while the odd
  columns still hold the expected id = 16 * row + column, so the staging layout is
  right and the selection is wrong. Not yet triaged; no variant here asserts it.
* `dest_acc` is pinned No, structurally: the kernel carries the expert id in the LO16
  and the score in the HI16 of one DEST word, which only exists for a 16-bit DEST
  format. A 32-bit DEST leaves no room for the payload.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MOE_GATE_NORMALIZE_PARAMS,
    MOE_GATE_TOPK,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FACE_DIM = 16
NUM_TOTAL_EXPERTS = 256  # one face of a 32x32 tile
NUM_RESULT_TILES = 2  # winner scores, winner indices

EPS_BITS = 0x00000000  # 0.0f -- all scores are positive so the sum never vanishes
SCALE_BITS = 0x3F800000  # 1.0f

# Every path except num_selected_experts=16 emits exactly this many winners.
TOP8_WINNERS = 8

# num_selected_experts < 8 combined with normalize, and num_selected_experts == 16,
# both return the wrong expert set on Blackhole. Measured, with the mechanism where it
# is established; see the module docstring for the configurations pinned out entirely.
_XFAIL_SUB8_NORMALIZE = (
    "num_selected_experts<8 with normalize selects the wrong experts. top8.h:245 runs "
    "_generic_moe_gate_top8_zero_tail_ for `zero_tail || (normalize && "
    "num_selected_experts < 8)`, which zeroes the LREG5 payload and so blanks winner "
    "rows 4-7, leaving only rows 0-3 live. Those rows are the output of the "
    "SFPSWAP(LREG0, LREG1, ALL_ROWS_MAX) half-cleaner that opens "
    "_generic_moe_gate_top8_sort_rows_, and a half-cleaner yields the true top half "
    "only for a bitonic input. The eight winners are not bitonic -- measured rank "
    "order [0, 1, 3, 4, 2, 5, 6, 7] even with full_sort=True -- so rank 2 is paired "
    "against rank 0 and dies while rank 4 survives its own pair. Rows 0-3 come back "
    "as ranks {0, 1, 3, 4}. Sorting after the halving cannot recover it."
)
_XFAIL_TOP16 = (
    "the top-16 path returns 15 of the 16 correct ids in lane 0, substituting golden "
    "rank 16 for rank 14. The correct top-16 set is present in lane 1 (column 2), so "
    "the fault is in the final reduce-to-lane-0 "
    "(_generic_moe_gate_top16_reduce_lanes_<4, false> plus "
    "_generic_moe_gate_top16_shift_left_once_) rather than in the sort network. "
    "Reproduces identically at num_total_experts 128 and 256. Not asserted against "
    "lane 1: lane 0 is the documented result lane, so reading lane 2 instead would "
    "encode the bug and break when it is fixed."
)


def _distinct_bf16_keys() -> torch.Tensor:
    """The 256 exactly-representable bfloat16 values in [1.0, 2.0), shuffled.

    bfloat16 has an 8-bit mantissa, so [1, 2) contains exactly 256 values spaced
    1/256 apart. Using a permutation of them makes every sort key distinct in the
    comparison format, so the top-8 boundary cannot be a tie.
    """
    keys = 1.0 + torch.arange(NUM_TOTAL_EXPERTS, dtype=torch.float32) / 256.0
    return keys[torch.randperm(NUM_TOTAL_EXPERTS)]


def _face0_tile(values: torch.Tensor, torch_format) -> torch.Tensor:
    """Place 256 values row-major into face 0 of a [32, 32] tile; rest zero."""
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)
    tile[:FACE_DIM, :FACE_DIM] = values.reshape(FACE_DIM, FACE_DIM)
    return tile.to(torch_format)


def _bits_to_float(bits: int) -> float:
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()


def assert_odd_columns_untouched(result_indices, result_scores, scores, rows):
    """The top-8 winner store writes only the even columns; the odd ones stay intact.

    That makes the odd columns a direct readout of two things the test otherwise has
    to take on faith: the expert numbering `_topk_moe_generate_indices_` produced
    (id = 16 * row + column, so an odd column still holds its own id), and the raw
    scores the datacopy staged (normalize only scales the even columns). It is also
    what pins the even/odd reading of the SFPU DEST address -- under a
    first-half/second-half reading these columns would have been overwritten.

    Top-8 path only. The top-16 network stores both column halves as it works, so its
    odd columns legitimately hold intermediate state by the time it finishes.
    """
    for row in range(rows):
        for col in range(1, FACE_DIM, 2):
            got_id = int(result_indices[row, col].item())
            want_id = FACE_DIM * row + col
            assert got_id == want_id, (
                f"indices: odd column [{row}, {col}] should still hold its generated "
                f"id {want_id}, got {got_id}"
            )
            got_score = result_scores[row, col].item()
            want_score = scores[FACE_DIM * row + col].item()
            assert got_score == want_score, (
                f"scores: odd column [{row}, {col}] should still hold the staged "
                f"score {want_score}, got {got_score}"
            )


@parametrize(
    dest_acc=[DestAccumulation.No],
    num_selected_experts=[8, 4, 16],
    full_sort=[True, False],
    normalize=[True, False],
)
def test_sfpu_generic_moe_gate_topk(
    request, dest_acc, num_selected_experts, full_sort, normalize
):
    # Known-wrong configurations still compile and run, so the day either is fixed the
    # suite reports XPASS instead of quietly staying green.
    if num_selected_experts == 16:
        request.node.add_marker(pytest.mark.xfail(reason=_XFAIL_TOP16, strict=True))
    elif num_selected_experts < TOP8_WINNERS and normalize:
        request.node.add_marker(
            pytest.mark.xfail(reason=_XFAIL_SUB8_NORMALIZE, strict=True)
        )

    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # Sort keys: distinct by construction. Raw scores: independent, positive, and
    # deliberately on a different scale from the keys so that reporting the key
    # instead of the score would be caught.
    biased = _distinct_bf16_keys()
    scores = (
        torch.empty(NUM_TOTAL_EXPERTS, dtype=torch.float32)
        .uniform_(0.05, 0.95)
        .to(torch_format)
        .to(torch.float32)
    )

    scores_tile = _face0_tile(scores, torch_format)
    biased_tile = _face0_tile(biased, torch_format)

    src_A = torch.cat(
        [
            tilize_block(
                t.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
            ).flatten()
            for t in (scores_tile, biased_tile)
        ]
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    configuration = TestConfig(
        "sources/sfpu_generic_moe_gate_topk_test.cpp",
        formats,
        templates=[
            MOE_GATE_TOPK(
                num_selected_experts=num_selected_experts,
                num_total_experts=NUM_TOTAL_EXPERTS,
                normalize=normalize,
                zero_tail=False,  # see the module docstring
                full_sort=full_sort,
            ),
            MOE_GATE_NORMALIZE_PARAMS(eps_bits=EPS_BITS, scale_bits=SCALE_BITS),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=NUM_RESULT_TILES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    score_words = res_tensor[:ELEMENTS_PER_TILE]
    index_words = res_tensor[ELEMENTS_PER_TILE : 2 * ELEMENTS_PER_TILE]

    result_scores = untilize_block(
        score_words, formats.output_format, [TILE_DIM, TILE_DIM]
    ).reshape(TILE_DIM, TILE_DIM)
    # The index tile was packed as UInt16, so reinterpret the raw 16-bit words.
    result_indices = (
        untilize_block(index_words, formats.output_format, [TILE_DIM, TILE_DIM])
        .contiguous()
        .view(torch.uint16)
        .to(torch.int32)
        .reshape(TILE_DIM, TILE_DIM)
    )

    # Only 16 selects the top-16 path; everything else emits eight winners.
    is_top16 = num_selected_experts == 16
    num_winners = 16 if is_top16 else TOP8_WINNERS

    golden_ids = torch.argsort(biased, descending=True)[:num_winners].tolist()

    got_ids = [int(result_indices[row, 0].item()) for row in range(num_winners)]
    got_scores = torch.tensor(
        [result_scores[row, 0].item() for row in range(num_winners)],
        dtype=torch.float32,
    )

    assert sorted(got_ids) == sorted(golden_ids), (
        f"selected expert ids differ.\n"
        f"  got    {sorted(got_ids)}\n"
        f"  golden {sorted(golden_ids)}"
    )

    # Pairing check: each returned score must be the normalized original score of the
    # expert id it came back with. This is what catches a key/payload mix-up.
    scale = _bits_to_float(SCALE_BITS)
    eps = _bits_to_float(EPS_BITS)
    total = scores[golden_ids].to(torch.float32).sum()
    factor = scale / (total + eps) if normalize else 1.0
    expected_paired = torch.tensor(
        [scores[i].item() * factor for i in got_ids], dtype=torch.float32
    )
    assert passed_test(
        expected_paired, got_scores, formats.output_format, print_errors=True
    ), "returned scores are not the (normalized) original scores of the returned ids"

    if not normalize:
        # Sanity: without normalize the scores come through untouched, so the golden
        # multiset is just the raw scores of the winners.
        assert passed_test(
            torch.sort(scores[golden_ids].to(torch.float32)).values,
            torch.sort(got_scores).values,
            formats.output_format,
        ), "un-normalized winner scores do not match the raw input scores"

    if not is_top16:
        assert_odd_columns_untouched(result_indices, result_scores, scores, num_winners)
