# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# This directory is swept wholesale per-PR on Wormhole and on both simulators,
# and the op has only ever been built and measured on Blackhole. The kernel runs
# an fp32 dest accumulator, whose register budget differs by architecture, so
# "no architecture-specific code" is not evidence that it lands correctly there.
# Drop this gate once someone has run the file on a Wormhole card.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="attn_res_stats has only been validated on Blackhole")

# Both statistics come out of one fp32 dest accumulator and round once at the
# pack, so the gate is against torch in fp32 and it is strict. The composed form
# this replaces packs the sum of squares to bfloat16 before its own typecast and
# cannot hold this at the wide shapes below; that is the accuracy the op buys.
PCC = 0.9999

# A sum of `d` squares sits near `d` with a spread of only `sqrt(2d)`, so a
# bfloat16 pack quantizes it at 2^-8 of its magnitude rather than of its spread
# and costs roughly 8e-4 of correlation at the production width. The dots are
# zero-mean and lose nothing measurable. This is why the caller asks for an fp32
# output, and why the two halves are gated apart.
BFLOAT16_SQUARES_PCC = 0.999


def _place(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _reference(torch_v, torch_q):
    """The two statistics stacked on the candidate axis, in the order the op emits."""
    v, q = torch_v.float(), torch_q.float()
    return torch.cat([(v * v).sum(-1, keepdim=True), (v * q).sum(-1, keepdim=True)], dim=1)


def _assert_halves(golden, got, candidates):
    """Gate the two statistics separately.

    A sum of squares sits near `d` while a dot sits near zero, so a correlation
    taken across the stacked pair is dominated by the gap between the two blocks
    and stays close to 1 even when one half is wrong.
    """
    assert_with_pcc(golden[:, :candidates], got[:, :candidates], PCC)
    assert_with_pcc(golden[:, candidates:], got[:, candidates:], PCC)


@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 256, 1792],
        [1, 1, 640, 1792],
        [1, 9, 256, 1792],
        [1, 8, 128, 3584],
        [1, 1, 64, 32],
        [1, 9, 32, 128],
        [1, 3, 128, 64],
    ),
    ids=[
        # What the split form calls per read site on a (2,4) mesh — one candidate
        # against d/4 — with the token count cut to keep the test cheap, then at
        # the production 640 per chip.
        "split_short",
        "split_full",
        # The direct form at S=8: the live stream stacked onto eight snapshots.
        "direct_s8",
        # Twice the production width, where the composed form's bfloat16 pack of
        # the squares drops below this gate and the fused op does not.
        "wide_d",
        # Wt=1: one tile of d, so the row is consumed in a single pass and the
        # reduction never accumulates across tiles.
        "wt1_single_tile_row",
        # Ht=1: one tile of tokens against many candidates, which turns the
        # candidate loop over on every output row.
        "ht1_many_candidates",
        # An odd candidate count, so the writer's page arithmetic for the second
        # statistic starts at a page that is not a multiple of the row count.
        "c3_odd",
    ],
)
def test_attn_res_stats(shape, device):
    torch.manual_seed(2026)
    _, candidates, _, hidden = shape

    torch_v = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, hidden], dtype=torch.bfloat16)

    output = ttnn.experimental.attn_res_stats(_place(torch_v, device), _place(torch_q, device), dtype=ttnn.float32)

    assert list(output.shape) == [shape[0], 2 * candidates, shape[2], 1]
    assert output.dtype == ttnn.float32
    _assert_halves(_reference(torch_v, torch_q), ttnn.to_torch(output).float(), candidates)


@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.float32), ids=["bf16_out", "fp32_out"])
def test_attn_res_stats_output_dtype(dtype, device):
    """The collective downstream carries whichever the caller asks for.

    bfloat16 halves what crosses the fabric and costs the squares the pack
    described at `BFLOAT16_SQUARES_PCC`; fp32 carries the accumulator's precision
    through. The dots take the strict gate either way, so a regression in them
    cannot hide behind the looser half.
    """
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]
    candidates = shape[1]

    torch_v = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 1792], dtype=torch.bfloat16)

    output = ttnn.experimental.attn_res_stats(_place(torch_v, device), _place(torch_q, device), dtype=dtype)

    assert output.dtype == dtype
    golden, got = _reference(torch_v, torch_q), ttnn.to_torch(output).float()
    squares_gate = PCC if dtype == ttnn.float32 else BFLOAT16_SQUARES_PCC
    assert_with_pcc(golden[:, :candidates], got[:, :candidates], squares_gate)
    assert_with_pcc(golden[:, candidates:], got[:, candidates:], PCC)


def test_attn_res_stats_fp32_input(device):
    """fp32 operands, which double every circular buffer the kernel holds."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 1792]

    torch_v = torch.randn(shape, dtype=torch.float32)
    torch_q = torch.randn([1, 1, 1, 1792], dtype=torch.float32)

    output = ttnn.experimental.attn_res_stats(
        _place(torch_v, device, ttnn.float32), _place(torch_q, device, ttnn.float32), dtype=ttnn.float32
    )

    _assert_halves(_reference(torch_v, torch_q), ttnn.to_torch(output).float(), shape[1])


def test_attn_res_stats_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    `from_torch` zero-pads, and a padded row contributes zero to both statistics,
    so the logical region is unaffected — this asserts that rather than assuming
    it."""
    torch.manual_seed(2026)
    shape = [1, 9, 100, 128]

    torch_v = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 128], dtype=torch.bfloat16)

    output = ttnn.experimental.attn_res_stats(_place(torch_v, device), _place(torch_q, device), dtype=ttnn.float32)

    assert list(output.shape) == [1, 18, 100, 1]
    _assert_halves(_reference(torch_v, torch_q), ttnn.to_torch(output).float(), shape[1])


def test_attn_res_stats_program_cache(device):
    """Second call must hit the program cache and still read the right buffers.

    The descriptor declares both input addresses as buffer bindings; a cache hit
    patches them in place rather than rebuilding. If that binding were wrong the
    second call would silently reduce the first call's tensors, which a
    single-shot test cannot see."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 256]

    entries_before = device.num_program_cache_entries()
    for _ in range(2):
        torch_v = torch.randn(shape, dtype=torch.bfloat16)
        torch_q = torch.randn([1, 1, 1, 256], dtype=torch.bfloat16)
        # Held until after the check so the next iteration cannot reuse the
        # addresses, which would make a stale binding look correct.
        tt_v, tt_q = _place(torch_v, device), _place(torch_q, device)
        output = ttnn.experimental.attn_res_stats(tt_v, tt_q, dtype=ttnn.float32)
        _assert_halves(_reference(torch_v, torch_q), ttnn.to_torch(output).float(), shape[1])

    assert device.num_program_cache_entries() - entries_before == 1


@pytest.mark.parametrize(
    "bad, message",
    (
        ("v_batched", "requires a leading dim of 1 on v"),
        ("q_candidates", "requires a single query row"),
        ("q_rows", "must occupy exactly one tile row"),
        ("d_mismatch", "contracts v and q over d"),
        ("rank", "requires a rank-4 v"),
        ("dtype", "only supports specific data types"),
    ),
)
def test_attn_res_stats_rejects(bad, message, device, expect_error):
    """The narrow contract is enforced, not documented and hoped for.

    Each case pins its own message, so a rejection for the wrong reason fails
    here rather than counting as coverage."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 128]
    tt_v = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    tt_q = _place(torch.randn([1, 1, 1, 128], dtype=torch.bfloat16), device)

    if bad == "v_batched":
        tt_v = _place(torch.randn([2, 9, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "q_candidates":
        tt_q = _place(torch.randn([1, 2, 1, 128], dtype=torch.bfloat16), device)
    elif bad == "q_rows":
        # Two tile rows of query: the kernel broadcasts one row down the tokens
        # and has no way to choose between them.
        tt_q = _place(torch.randn([1, 1, 64, 128], dtype=torch.bfloat16), device)
    elif bad == "d_mismatch":
        tt_q = _place(torch.randn([1, 1, 1, 64], dtype=torch.bfloat16), device)
    elif bad == "rank":
        tt_v = _place(torch.randn([9, 128, 128], dtype=torch.bfloat16), device)
    elif bad == "dtype":
        tt_v = _place(torch.randn(shape, dtype=torch.float32), device, ttnn.bfloat8_b)

    with expect_error(RuntimeError, message):
        ttnn.experimental.attn_res_stats(tt_v, tt_q, dtype=ttnn.float32)
