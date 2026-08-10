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
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="attn_res_accum_stats has only been validated on Blackhole")

PCC = 0.9999

# A sum of `d` squares sits near `d` with a spread of only `sqrt(2d)`, so a
# bfloat16 pack quantizes it at 2^-8 of its magnitude rather than of its spread.
# The dots are zero-mean and lose nothing measurable, so the two halves are
# gated apart.
BFLOAT16_SQUARES_PCC = 0.999


def _place(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _reference(torch_a, torch_b, torch_q):
    """The sum, and the two statistics of that sum stacked on the candidate axis.

    The statistics are taken against the rounded sum rather than an fp32 one:
    the kernel packs its accumulator once and both reductions read what the
    writer emitted, so a reference that keeps full precision through the add
    would be measuring a different quantity.
    """
    total = torch_a + torch_b
    t, q = total.float(), torch_q.float()
    stats = torch.cat([(t * t).sum(-1, keepdim=True), (t * q).sum(-1, keepdim=True)], dim=1)
    return total, stats


def _assert_sum(golden, got):
    """The sum is exact up to its own pack.

    Both addends are already representable, so the accumulation in dest is
    exact and the only error is the single rounding on the way out — which the
    device packer and torch may resolve to different neighbours. One ULP is
    2**-7 relative for any normal, and anything looser than that is a bug the
    correlation gate used on the statistics would not catch.
    """
    assert got.dtype == golden.dtype
    ulp = 2**-7 if golden.dtype == torch.bfloat16 else 2**-23
    torch.testing.assert_close(got.float(), golden.float(), rtol=ulp, atol=0.0)


def _assert_halves(golden, got, candidates, squares_gate=PCC):
    """Gate the two statistics separately.

    A sum of squares sits near `d` while a dot sits near zero, so a correlation
    taken across the stacked pair is dominated by the gap between the two blocks
    and stays close to 1 even when one half is wrong.
    """
    assert_with_pcc(golden[:, :candidates], got[:, :candidates], squares_gate)
    assert_with_pcc(golden[:, candidates:], got[:, candidates:], PCC)


@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 256, 1792],
        [1, 1, 640, 1792],
        [1, 9, 256, 1792],
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
def test_attn_res_accum_stats(shape, device):
    torch.manual_seed(2026)
    _, candidates, _, hidden = shape

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, hidden], dtype=torch.bfloat16)

    total, stats = ttnn.experimental.attn_res_accum_stats(
        _place(torch_a, device), _place(torch_b, device), _place(torch_q, device), stats_dtype=ttnn.float32
    )

    assert list(total.shape) == shape
    assert total.dtype == ttnn.bfloat16
    assert list(stats.shape) == [shape[0], 2 * candidates, shape[2], 1]
    assert stats.dtype == ttnn.float32

    golden_total, golden_stats = _reference(torch_a, torch_b, torch_q)
    _assert_sum(golden_total, ttnn.to_torch(total))
    _assert_halves(golden_stats, ttnn.to_torch(stats).float(), candidates)


def test_attn_res_accum_stats_reduces_the_sum_not_an_addend(device):
    """The reductions must read what the add produced.

    A kernel that reduced `a` and emitted `a + b` would pass a correlation gate
    on the sum alone, so this drives `b` to cancel `a`: the statistics of the sum
    are then zero and those of either addend are not.
    """
    torch.manual_seed(2026)
    shape = [1, 4, 128, 512]

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 512], dtype=torch.bfloat16)

    _, stats = ttnn.experimental.attn_res_accum_stats(
        _place(torch_a, device), _place(-torch_a, device), _place(torch_q, device), stats_dtype=ttnn.float32
    )

    assert torch.count_nonzero(ttnn.to_torch(stats)) == 0


@pytest.mark.parametrize("stats_dtype", (ttnn.bfloat16, ttnn.float32), ids=["bf16_stats", "fp32_stats"])
def test_attn_res_accum_stats_output_dtype(stats_dtype, device):
    """The collective downstream carries whichever the caller asks for.

    bfloat16 halves what crosses the fabric and costs the squares the pack
    described at `BFLOAT16_SQUARES_PCC`; fp32 carries the accumulator's precision
    through. The dots take the strict gate either way, so a regression in them
    cannot hide behind the looser half.
    """
    torch.manual_seed(2026)
    shape = [1, 9, 256, 1792]
    candidates = shape[1]

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 1792], dtype=torch.bfloat16)

    total, stats = ttnn.experimental.attn_res_accum_stats(
        _place(torch_a, device), _place(torch_b, device), _place(torch_q, device), stats_dtype=stats_dtype
    )

    assert stats.dtype == stats_dtype
    # The sum keeps the addends' dtype regardless of what the statistics carry.
    assert total.dtype == ttnn.bfloat16

    golden_total, golden_stats = _reference(torch_a, torch_b, torch_q)
    _assert_sum(golden_total, ttnn.to_torch(total))
    _assert_halves(
        golden_stats,
        ttnn.to_torch(stats).float(),
        candidates,
        PCC if stats_dtype == ttnn.float32 else BFLOAT16_SQUARES_PCC,
    )


def test_attn_res_accum_stats_fp32_query(device):
    """An fp32 query against bfloat16 addends, which is what the caller holds.

    The dot is the only place `q` is read, and it is contracted immediately, so
    the wider query costs one circular buffer and buys the reduction its input
    at full precision.
    """
    torch.manual_seed(2026)
    shape = [1, 9, 128, 1792]

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 1792], dtype=torch.float32)

    total, stats = ttnn.experimental.attn_res_accum_stats(
        _place(torch_a, device),
        _place(torch_b, device),
        _place(torch_q, device, ttnn.float32),
        stats_dtype=ttnn.float32,
    )

    golden_total, golden_stats = _reference(torch_a, torch_b, torch_q)
    _assert_sum(golden_total, ttnn.to_torch(total))
    _assert_halves(golden_stats, ttnn.to_torch(stats).float(), shape[1])


def test_attn_res_accum_stats_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    `from_torch` zero-pads, and a padded row contributes zero to both statistics,
    so the logical region is unaffected — this asserts that rather than assuming
    it."""
    torch.manual_seed(2026)
    shape = [1, 9, 100, 128]

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    torch_q = torch.randn([1, 1, 1, 128], dtype=torch.bfloat16)

    total, stats = ttnn.experimental.attn_res_accum_stats(
        _place(torch_a, device), _place(torch_b, device), _place(torch_q, device), stats_dtype=ttnn.float32
    )

    assert list(stats.shape) == [1, 18, 100, 1]
    golden_total, golden_stats = _reference(torch_a, torch_b, torch_q)
    _assert_sum(golden_total, ttnn.to_torch(total))
    _assert_halves(golden_stats, ttnn.to_torch(stats).float(), shape[1])


def test_attn_res_accum_stats_program_cache(device):
    """Second call must hit the program cache and still read the right buffers.

    The descriptor declares three input and two output addresses as buffer
    bindings; a cache hit patches them in place rather than rebuilding. If one
    binding were wrong the second call would silently work on the first call's
    tensors, which a single-shot test cannot see."""
    torch.manual_seed(2026)
    shape = [1, 9, 128, 256]

    entries_before = device.num_program_cache_entries()
    for _ in range(2):
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)
        torch_q = torch.randn([1, 1, 1, 256], dtype=torch.bfloat16)
        # Held until after the check so the next iteration cannot reuse the
        # addresses, which would make a stale binding look correct.
        tt_a, tt_b, tt_q = _place(torch_a, device), _place(torch_b, device), _place(torch_q, device)
        total, stats = ttnn.experimental.attn_res_accum_stats(tt_a, tt_b, tt_q, stats_dtype=ttnn.float32)

        golden_total, golden_stats = _reference(torch_a, torch_b, torch_q)
        _assert_sum(golden_total, ttnn.to_torch(total))
        _assert_halves(golden_stats, ttnn.to_torch(stats).float(), shape[1])

    assert device.num_program_cache_entries() - entries_before == 1


@pytest.mark.parametrize(
    "bad, message",
    (
        ("fp32_addend", "only supports specific data types"),
        ("shape_mismatch", "sums matching shapes"),
        ("a_batched", "requires a leading dim of 1 on a"),
        ("q_rows", "broadcasts a single query row"),
        ("d_mismatch", "contracts the sum and q over d"),
    ),
)
def test_attn_res_accum_stats_rejects(bad, message, device, expect_error):
    """The narrow contract is enforced, not documented and hoped for.

    Each case pins its own message, so a rejection for the wrong reason fails
    here rather than counting as coverage."""
    torch.manual_seed(2026)
    shape = [1, 2, 64, 128]

    a = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    b = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    q = _place(torch.randn([1, 1, 1, 128], dtype=torch.bfloat16), device)

    if bad == "fp32_addend":
        b = _place(torch.randn(shape, dtype=torch.float32), device, ttnn.float32)
    elif bad == "shape_mismatch":
        b = _place(torch.randn([1, 2, 32, 128], dtype=torch.bfloat16), device)
    elif bad == "a_batched":
        a = _place(torch.randn([2, 2, 64, 128], dtype=torch.bfloat16), device)
        b = _place(torch.randn([2, 2, 64, 128], dtype=torch.bfloat16), device)
    elif bad == "q_rows":
        q = _place(torch.randn([1, 1, 64, 128], dtype=torch.bfloat16), device)
    elif bad == "d_mismatch":
        q = _place(torch.randn([1, 1, 1, 256], dtype=torch.bfloat16), device)

    with expect_error(RuntimeError, message):
        ttnn.experimental.attn_res_accum_stats(a, b, q, stats_dtype=ttnn.float32)


def test_attn_res_accum_stats_rejects_oversized_d(device, expect_error):
    """The L1 budget is checked up front, not discovered during allocation.

    Holding both addends double-buffered alongside q, the sum and a transformed
    copy is roughly twice what the unfused statistics kernel needs, which caps d
    at a little over 2800 — well above the production width, but a real limit
    the caller has to learn from the guard rather than from a descriptor failure
    further down.
    """
    torch.manual_seed(2026)
    shape = [1, 1, 32, 4096]

    a = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    b = _place(torch.randn(shape, dtype=torch.bfloat16), device)
    q = _place(torch.randn([1, 1, 1, 4096], dtype=torch.bfloat16), device)

    with expect_error(RuntimeError, "needs .* B per core"):
        ttnn.experimental.attn_res_accum_stats(a, b, q, stats_dtype=ttnn.float32)
