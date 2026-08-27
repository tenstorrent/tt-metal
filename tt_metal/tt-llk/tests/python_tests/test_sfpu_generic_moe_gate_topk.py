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
rows 0-7 on the top-8 path and rows 0-15 on the top-16 path.
`_generic_moe_gate_top8_merge_instances_` is a reduce-to-one-lane rotate-and-merge
network, not an all-reduce, so the other even columns are left holding partial merges
(measured: they contain ids outside the top-8) and only lane 0 is meaningful. The
top-8 path never writes the odd columns; the top-16 path does
(`_generic_moe_gate_store_16_rows_even_odd_split_` stores both halves), which is why
the odd-column check below is restricted to the top-8 path.

How many winners come back
--------------------------
`_generic_moe_gate_topk_` routes num_selected_experts > 8 to the top-16 path, which
emits 16 winner rows; values <= 8 route to top-8, which emits eight rows.

On either path the tail beyond num_selected_experts is blanked -- id 0, score 0.0 --
when `zero_tail || normalize`. With neither flag, the entire 8- or 16-row result
remains live. With normalize, blanking happens before `_generic_moe_gate_normalize_`
so its denominator contains only the requested winners. `_expected_live_winners`
encodes this rule.

num_selected_experts=4 is also distinct code beyond the tail: with full_sort=False,
`_generic_moe_gate_top8_sort_rows_<4, false>` takes the SFPSWAP + TTI_NOP branch that
num_selected_experts=8 never reaches.

What this test asserts, and why in this form
--------------------------------------------
Row order is asserted whenever full_sort=True. full_sort=False gives a partial order
on the top-8 path; the intermediate 9-15 path still performs the final 16-row sort
because truncation requires the requested winners to occupy the leading rows. The
checks are:
  1. the live winner id set == the golden top-N id set, for the N above;
  2. every returned score matches the normalized original score of the id it is
     paired with -- this is the check that would catch a payload/key mix-up, which is
     the interesting failure mode for a bitonic sort carrying a payload;
  3. the blanked tail rows, when there are any, hold exactly id 0 and score 0.0;
  4. for full_sort=True, the live rows are in descending-key order;
  5. for the top-8 path, the odd columns still hold their generated expert id and
     their staged score, which confirms the store touched only the even columns and
     pins the even/odd reading of the SFPU DEST address. This holds for all eight
     emitted rows including blanked ones -- zero_tail zeroes the even-column payload,
     never the odd columns.

Stimuli are built so the top-8 boundary is never ambiguous: the biased scores are a
permutation of 256 consecutive bfloat16 encodings (1.0 .. 3.984375), so all 256 sort
keys are distinct in the format the hardware compares in -- and stay distinct in the
golden, which sorts the same values -- so there is no tie to break.

Scope: the SFPU kernel only. In the compute API the score and biased tiles are staged
by the FPU kernel `_llk_math_deepseek_moe_gate_eltwise_binary_`; here they are staged
by plain datacopy so this test isolates the SFPU half. Cover the FPU kernel
separately.

Expert-count coverage
---------------------
The full flag cross-product remains pinned to 256 experts. A focused companion sweep
covers 16-aligned counts, the top-8 64-expert path, and the 288-expert padding case on
both top-8 and top-16. Its unused lanes contain large positive poison keys, so a
missing `-inf` mask fails by selecting an out-of-range generated id.

Configurations left uncovered
-----------------------------
* `dest_acc` is pinned No, structurally: the kernel carries the expert id in the LO16
  and the score in the HI16 of one DEST word, which only exists for a 16-bit DEST
  format. A 32-bit DEST leaves no room for the payload.
* `generate_indices` is pinned true -- the driver instantiates
  `_generic_moe_gate_topk_` with five of its six template arguments, so the kernel
  always numbers the experts itself and the caller-supplied index-mapping path is not
  reached. See MOE_GATE_TOPK in helpers/test_variant_parameters.py.
"""

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    MoeGateTopkGolden,
    get_golden_generator,
)
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

# Winner rows each path emits, whatever num_selected_experts asked for.
TOP8_WINNERS = 8
TOP16_WINNERS = 16


def _expected_live_winners(
    num_selected_experts: int, normalize: bool, zero_tail: bool
) -> tuple[int, int]:
    """(rows the path emits, rows that hold a live winner) for one configuration.

    The top-8 path emits eight rows and the top-16 path emits sixteen. On either
    path, normalize or zero_tail blanks rows beyond num_selected_experts.
    """
    emitted_rows = (
        TOP16_WINNERS if num_selected_experts > TOP8_WINNERS else TOP8_WINNERS
    )
    if num_selected_experts < emitted_rows and (normalize or zero_tail):
        return emitted_rows, num_selected_experts
    return emitted_rows, emitted_rows


def _distinct_bf16_keys(num_total_experts: int) -> torch.Tensor:
    """Consecutive bfloat16 encodings starting at 1.0, shuffled.

    bfloat16 has 7 fraction bits, so [1, 2) holds only 128 distinct values, spaced
    1/128 apart -- keys spaced 1/256 would collapse in pairs the moment _face0_tile
    casts them to the format the hardware compares in, reintroducing exactly the ties
    this is meant to avoid (while the golden still sorted the un-collapsed fp32
    values). Walking consecutive bf16 bit patterns instead -- 0x3F80..0x407F, i.e.
    1.0 up to 3.984375, spacing 1/128 in [1, 2) and 1/64 in [2, 4) -- keeps all 256
    distinct with no rounding at all, so the top-8 boundary cannot be a tie.
    """
    bits = (0x3F80 + torch.arange(num_total_experts, dtype=torch.int32)) << 16
    keys = bits.contiguous().view(torch.float32)
    return keys[torch.randperm(num_total_experts)]


def _experts_tile(
    values: torch.Tensor, torch_format, pad_value: float = 0.0
) -> torch.Tensor:
    """Place values consecutively in face order; fill unused lanes with padding."""
    tile = torch.full((TILE_DIM, TILE_DIM), pad_value, dtype=torch.float32)
    for face_idx in range(4):
        start = face_idx * FACE_DIM * FACE_DIM
        if start >= values.numel():
            break
        end = min(start + FACE_DIM * FACE_DIM, values.numel())
        face_values = values[start:end]
        face = torch.full((FACE_DIM, FACE_DIM), pad_value, dtype=torch.float32)
        face.reshape(-1)[: face_values.numel()] = face_values
        face_row = (face_idx // 2) * FACE_DIM
        face_col = (face_idx % 2) * FACE_DIM
        tile[
            face_row : face_row + FACE_DIM,
            face_col : face_col + FACE_DIM,
        ] = face
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


def _run_sfpu_generic_moe_gate_topk(
    num_selected_experts,
    full_sort,
    normalize,
    zero_tail,
    num_total_experts=NUM_TOTAL_EXPERTS,
):
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # Sort keys: distinct by construction. Raw scores: independent, positive, and
    # deliberately on a different scale from the keys so that reporting the key
    # instead of the score would be caught.
    biased = _distinct_bf16_keys(num_total_experts)
    scores = (
        torch.empty(num_total_experts, dtype=torch.float32)
        .uniform_(0.05, 0.95)
        .to(torch_format)
        .to(torch.float32)
    )

    # Make unread/padded lanes obvious: without the SFPU -inf mask they beat
    # every valid key and the test returns an out-of-range generated id.
    scores_tile = _experts_tile(scores, torch_format, pad_value=7.0)
    biased_tile = _experts_tile(biased, torch_format, pad_value=100.0)

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
                num_total_experts=num_total_experts,
                normalize=normalize,
                zero_tail=zero_tail,
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
        # Pinned, not swept: the kernel packs the expert id and the score into the LO16
        # and HI16 of one DEST word, which only exists for a 16-bit DEST.
        dest_acc=DestAccumulation.No,
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

    emitted_rows, num_winners = _expected_live_winners(
        num_selected_experts, normalize, zero_tail
    )
    is_top16_path = num_selected_experts > TOP8_WINNERS

    golden_generator = get_golden_generator(MoeGateTopkGolden)
    scale = _bits_to_float(SCALE_BITS)
    eps = _bits_to_float(EPS_BITS)
    golden_ids, _ = golden_generator(
        biased, scores, num_winners, normalize, eps=eps, scale=scale
    )

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
    # expert id it came back with. This is what catches a key/payload mix-up. Asked for
    # in the order the device returned, but normalized over the golden winner set --
    # which, when the tail is blanked, is the live winners only, since the kernel blanks
    # before `_generic_moe_gate_normalize_` sums the rows.
    expected_paired = golden_generator.scores_for_ids(
        got_ids, golden_ids, scores, normalize, eps=eps, scale=scale
    )
    assert passed_test(
        expected_paired, got_scores, formats.output_format, print_errors=True
    ), "returned scores are not the (normalized) original scores of the returned ids"

    # full_sort must keep each score paired with its id while ordering by biased key.
    if full_sort:
        assert got_ids == golden_ids, (
            f"full_sort=True must return the winners in descending-key order.\n"
            f"  got    {got_ids}\n"
            f"  golden {golden_ids}"
        )

    # The blanked tail, when the configuration has one: id 0 and score 0.0 exactly.
    for row in range(num_winners, emitted_rows):
        got_id = int(result_indices[row, 0].item())
        got_score = result_scores[row, 0].item()
        assert got_id == 0 and got_score == 0.0, (
            f"row {row} is past num_selected_experts={num_selected_experts} with "
            f"normalize={normalize} / zero_tail={zero_tail}, so it must be blanked: "
            f"got id {got_id}, score {got_score}"
        )

    if not is_top16_path and num_total_experts >= 128:
        # All eight emitted rows, blanked ones included -- the tail zeroing hits the
        # even-column payload, never the odd columns.
        assert_odd_columns_untouched(
            result_indices, result_scores, scores, emitted_rows
        )


@skip_for_wormhole
@parametrize(
    num_selected_experts=[4, 8, *range(9, 17)],
    full_sort=[True, False],
    normalize=[True, False],
    zero_tail=[False, True],
)
def test_sfpu_generic_moe_gate_topk(
    num_selected_experts, full_sort, normalize, zero_tail
):
    _run_sfpu_generic_moe_gate_topk(
        num_selected_experts,
        full_sort,
        normalize,
        zero_tail,
    )


@skip_for_wormhole
@pytest.mark.parametrize(
    "num_total_experts,num_selected_experts",
    [
        pytest.param(16, 8, id="top8-16"),
        pytest.param(32, 8, id="top8-32"),
        pytest.param(48, 8, id="top8-48"),
        pytest.param(64, 8, id="top8-64"),
        pytest.param(192, 8, id="top8-192"),
        pytest.param(288, 8, id="top8-288"),
        pytest.param(288, 16, id="top16-288"),
    ],
)
def test_sfpu_generic_moe_gate_topk_expert_count_padding(
    num_total_experts, num_selected_experts
):
    _run_sfpu_generic_moe_gate_topk(
        num_selected_experts,
        full_sort=True,
        normalize=False,
        zero_tail=True,
        num_total_experts=num_total_experts,
    )
