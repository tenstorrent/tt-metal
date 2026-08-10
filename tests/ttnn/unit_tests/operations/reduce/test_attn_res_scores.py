# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# See test_attn_res_stats.py — same reason: swept per-PR on architectures the op
# has never been built for.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="attn_res_scores has only been validated on Blackhole")

# The whole normalization stays in dest registers and rounds once at the pack,
# where the six-op form it replaces rounds five times. The gate is against torch
# in fp32 and it is strict.
PCC = 0.9999

# The model's full hidden size, not the per-rank shard: the statistics reaching
# this op are already summed across tensor-parallel ranks.
HIDDEN_SIZE = 7168
INV_HIDDEN_SIZE = 1.0 / HIDDEN_SIZE
EPS = 1e-6


def _place(tensor, device, dtype=ttnn.float32):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _stats(candidates, tokens, width, torch_dtype):
    """A plausible globally summed pair.

    The sums of squares must be positive or the reciprocal square root is not
    defined, and their scale sets what `inv_hidden_size` normalizes to — drawing
    them uniformly around `HIDDEN_SIZE` puts the normalized value near one, which
    is where the real statistics land.
    """
    shape = [1, candidates, tokens, width]
    squares = (torch.rand(shape) + 0.5) * HIDDEN_SIZE
    dots = torch.randn(shape) * (HIDDEN_SIZE**0.5)
    return torch.cat([squares, dots], dim=1).to(torch_dtype)


def _reference(torch_stats, candidates):
    stats = torch_stats.float()
    squares, dots = stats[:, :candidates], stats[:, candidates:]
    return dots * torch.rsqrt(squares * INV_HIDDEN_SIZE + EPS)


@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 256, 1],
        [1, 1, 640, 1],
        [1, 9, 640, 1],
        [1, 24, 128, 1],
        [1, 1, 32, 1],
        [1, 9, 128, 32],
    ),
    ids=[
        # What the split form produces per read site — one candidate, one scalar
        # per token — with the token count cut, then at the production 640.
        "split_short",
        "split_full",
        # The direct form at S=8: eight snapshots plus the live stream.
        "direct_s8",
        # A block's worth of read sites, which is the widest candidate axis the
        # caller ever stacks.
        "block_of_sites",
        # One tile of tokens, so a single output tile covers the whole pair.
        "single_tile",
        # A full tile of columns rather than the one scalar per row the caller
        # uses, which is the shape the op's contract allows but the model never
        # asks for.
        "wide_last_dim",
    ],
)
def test_attn_res_scores(shape, device):
    torch.manual_seed(2026)
    _, candidates, tokens, width = shape

    torch_stats = _stats(candidates, tokens, width, torch.float32)

    output = ttnn.experimental.attn_res_scores(_place(torch_stats, device), INV_HIDDEN_SIZE, EPS, dtype=ttnn.bfloat16)

    assert list(output.shape) == shape
    assert output.dtype == ttnn.bfloat16
    assert_with_pcc(_reference(torch_stats, candidates), ttnn.to_torch(output).float(), PCC)


@pytest.mark.parametrize("num_partials", (2, 4, 8), ids=["tp2", "tp4", "tp8"])
def test_attn_res_scores_folds_partials(num_partials, device):
    """Per-rank statistics, still unsummed and stacked rank-major, fold in the op.

    This is what lets the collective ahead of it gather instead of reduce, so the
    reference is the same normalization over the summed pair: the op has to
    reproduce a sum that never happened on the wire."""
    torch.manual_seed(2026)
    candidates, tokens, width = 9, 128, 1

    # Each rank carries its share of the whole, so the summed pair lands at the
    # scale `inv_hidden_size` normalizes against however many ranks there are.
    partials = [_stats(candidates, tokens, width, torch.float32) / num_partials for _ in range(num_partials)]
    stacked = torch.cat(partials, dim=1)
    summed = torch.stack(partials).sum(dim=0)

    output = ttnn.experimental.attn_res_scores(
        _place(stacked, device), INV_HIDDEN_SIZE, EPS, num_partials=num_partials, dtype=ttnn.float32
    )

    assert list(output.shape) == [1, candidates, tokens, width]
    assert_with_pcc(_reference(summed, candidates), ttnn.to_torch(output).float(), PCC)


@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.float32), ids=["bf16", "fp32"])
def test_attn_res_scores_dtypes(dtype, device):
    """The statistics arrive in whatever the collective carried, and the score
    leaves in whatever the consumer wants; neither is tied to the other."""
    torch.manual_seed(2026)
    candidates, tokens = 9, 256

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_stats = _stats(candidates, tokens, 1, torch_dtype)

    output = ttnn.experimental.attn_res_scores(_place(torch_stats, device, dtype), INV_HIDDEN_SIZE, EPS, dtype=dtype)

    assert output.dtype == dtype
    assert_with_pcc(_reference(torch_stats, candidates), ttnn.to_torch(output).float(), PCC)


def test_attn_res_scores_unaligned_rows(device):
    """A token count that is not a multiple of the tile height.

    The padded rows carry a zero sum of squares, so their score is whatever
    `rsqrt(eps)` times zero gives; only the logical region is gated."""
    torch.manual_seed(2026)
    candidates, tokens = 9, 100

    torch_stats = _stats(candidates, tokens, 1, torch.float32)

    output = ttnn.experimental.attn_res_scores(_place(torch_stats, device), INV_HIDDEN_SIZE, EPS, dtype=ttnn.bfloat16)

    assert list(output.shape) == [1, candidates, tokens, 1]
    assert_with_pcc(_reference(torch_stats, candidates), ttnn.to_torch(output).float(), PCC)


def test_attn_res_scores_matches_composed(device):
    """Against the chain it replaces, at the same precision.

    The torch gate above says the op is correct; this one says it is a drop-in
    for the typecast, the two slices that unpack the pair, and the four
    elementwise ops that normalize."""
    torch.manual_seed(2026)
    candidates, tokens = 9, 256

    torch_stats = _stats(candidates, tokens, 1, torch.float32)
    tt_stats = _place(torch_stats, device)

    fused = ttnn.experimental.attn_res_scores(tt_stats, INV_HIDDEN_SIZE, EPS, dtype=ttnn.bfloat16)

    squares = ttnn.slice(tt_stats, [0, 0, 0, 0], [1, candidates, tokens, 1])
    dots = ttnn.slice(tt_stats, [0, candidates, 0, 0], [1, 2 * candidates, tokens, 1])
    reciprocal_rms = ttnn.rsqrt(ttnn.add(ttnn.mul(squares, INV_HIDDEN_SIZE), EPS))
    composed = ttnn.typecast(ttnn.mul(dots, reciprocal_rms), ttnn.bfloat16)

    assert_with_pcc(ttnn.to_torch(composed).float(), ttnn.to_torch(fused).float(), PCC)


def test_attn_res_scores_program_cache(device):
    """Second call must hit the program cache and still read the right buffer.

    A cache hit patches the input address in place rather than rebuilding. If
    that binding were wrong the second call would silently score the first
    call's statistics, which a single-shot test cannot see."""
    torch.manual_seed(2026)
    candidates, tokens = 9, 128

    entries_before = device.num_program_cache_entries()
    for _ in range(2):
        torch_stats = _stats(candidates, tokens, 1, torch.float32)
        # Held until after the check so the next iteration cannot reuse the
        # address, which would make a stale binding look correct.
        tt_stats = _place(torch_stats, device)
        output = ttnn.experimental.attn_res_scores(tt_stats, INV_HIDDEN_SIZE, EPS, dtype=ttnn.bfloat16)
        assert_with_pcc(_reference(torch_stats, candidates), ttnn.to_torch(output).float(), PCC)

    assert device.num_program_cache_entries() - entries_before == 1


@pytest.mark.parametrize(
    "bad, message",
    (
        ("odd_candidates", "must be a non-zero multiple of 2"),
        ("batched", "requires a leading dim of 1"),
        ("rank", "requires a rank-4 input"),
        ("dtype", "only supports specific data types"),
    ),
)
def test_attn_res_scores_rejects(bad, message, device, expect_error):
    """The narrow contract is enforced, not documented and hoped for.

    Each case pins its own message, so a rejection for the wrong reason fails
    here rather than counting as coverage."""
    torch.manual_seed(2026)
    tt_stats = _place(_stats(9, 128, 1, torch.float32), device)

    if bad == "odd_candidates":
        # An odd dim 1 cannot be a stacked pair, so the split has no midpoint.
        tt_stats = _place(torch.rand([1, 9, 128, 1]) + 1.0, device)
    elif bad == "batched":
        tt_stats = _place(torch.rand([2, 18, 128, 1]) + 1.0, device)
    elif bad == "rank":
        tt_stats = _place(torch.rand([18, 128, 1]) + 1.0, device)
    elif bad == "dtype":
        tt_stats = _place(torch.rand([1, 18, 128, 32]) + 1.0, device, ttnn.bfloat8_b)

    with expect_error(RuntimeError, message):
        ttnn.experimental.attn_res_scores(tt_stats, INV_HIDDEN_SIZE, EPS, dtype=ttnn.bfloat16)
