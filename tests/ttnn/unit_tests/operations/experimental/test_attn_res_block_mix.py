# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = run_for_blackhole()

SHARD_WIDTH = 1792  # 7168 over a 4-way tensor-parallel axis
EPS = 1e-5


def random_stats(num_candidates, num_tokens, num_sites, hidden_size):
    """Statistics at the magnitudes the model hands the op: sums of squares of unit-scale rows
    over the full hidden size, and dots with 0.02-scale queries."""
    sum_squares = hidden_size * (0.5 + torch.rand(1, num_candidates, num_tokens, 1))
    dots = torch.randn(1, num_candidates, num_tokens, num_sites) * 0.02 * hidden_size**0.5 * 4
    padding = torch.zeros(1, num_candidates, num_tokens, 31 - num_sites)
    return torch.cat([sum_squares, dots, padding], dim=-1)


def reference(stats, v, num_sites, hidden_size):
    sum_squares, dots = stats[..., :1], stats[..., 1 : 1 + num_sites]
    scores = dots * torch.rsqrt(sum_squares / hidden_size + EPS)  # [1, C, N, R]
    shift = scores.amax(dim=1, keepdim=True)
    weights = torch.exp(scores - shift)
    mass = weights.sum(dim=1, keepdim=True)
    partials = torch.einsum("bcnr,bcnd->rbnd", weights, v)  # [R, 1, N, d]
    site_major = lambda t: t.permute(3, 1, 2, 0)  # [1, 1, N, R] -> [R, 1, N, 1]
    return partials, site_major(shift), site_major(mass)


def run_mix(device, planes, num_sites, num_partials, num_tokens, num_closed=0):
    """The op's outputs and the reference's, as torch tensors."""
    torch.manual_seed(0)
    num_candidates = sum(planes)
    # Each partial is over one shard, so together they cover the hidden size the op scores over.
    hidden_size = SHARD_WIDTH * num_partials
    snapshots = [torch.randn(1, s, num_tokens, SHARD_WIDTH).bfloat16() for s in planes]
    stats = random_stats(num_candidates, num_tokens, num_sites, hidden_size)
    # Partials side by side on the last dim, as the tensor-parallel gather leaves them:
    # random shares of the statistics.
    shares = torch.rand(num_partials, *stats.shape)
    gathered = torch.cat(list(stats * shares / shares.sum(dim=0)), dim=-1)

    to_device = lambda t, dtype: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    outputs = ttnn.experimental.deepseek_prefill.attn_res_block_mix(
        [to_device(s, ttnn.bfloat16) for s in snapshots],
        to_device(gathered, ttnn.float32),
        num_sites=num_sites,
        eps=EPS,
        num_closed_sites=num_closed,
    )
    num_open = num_sites - num_closed
    assert len(outputs) == 4
    assert tuple(outputs[0].shape) == (num_open, 1, num_tokens, SHARD_WIDTH)
    assert tuple(outputs[1].shape) == tuple(outputs[2].shape) == (num_open, 1, num_tokens, 1)
    if num_closed:
        assert tuple(outputs[3].shape) == (num_closed, 1, num_tokens, SHARD_WIDTH)
    else:
        assert outputs[3] is None

    partials, shift, mass = reference(stats, torch.cat(snapshots, dim=1).float(), num_sites, hidden_size)
    got = tuple(ttnn.to_torch(t).float() for t in outputs if t is not None)
    return got, (partials, shift, mass)


@pytest.mark.parametrize("planes", [[1], [6], [6, 1], [3, 1, 1, 2]], ids=lambda p: "+".join(map(str, p)))
@pytest.mark.parametrize("num_sites", [1, 24])
@pytest.mark.parametrize("num_partials", [1, 4])
@pytest.mark.parametrize("num_tokens", [64, 640])
def test_attn_res_block_mix(device, planes, num_sites, num_partials, num_tokens):
    (out, tt_shift, tt_mass), (partials, shift, mass) = run_mix(device, planes, num_sites, num_partials, num_tokens)

    # Scoring is fp32 throughout.
    torch.testing.assert_close(tt_shift, shift, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tt_mass, mass, rtol=1e-4, atol=1e-4)
    passed, pcc = comp_pcc(partials, out, 0.9999)
    assert passed, f"partials PCC {pcc}"
    # Elementwise, a partial sums up to C unit-scale values with weights that are bf16 at the
    # MAC: ~2^-8 of up to C * 4, so 0.1.
    torch.testing.assert_close(out, partials, rtol=2e-2, atol=1e-1)


def test_attn_res_block_mix_rows_per_core(device):
    """More token rows than cores, so each core mixes several, with a short last site group."""
    test_attn_res_block_mix(device, [2], num_sites=5, num_partials=4, num_tokens=8192)


@pytest.mark.parametrize("planes", [[1], [6, 1]], ids=lambda p: "+".join(map(str, p)))
@pytest.mark.parametrize("num_sites, num_closed", [(2, 1), (25, 1), (25, 3)])
def test_attn_res_block_mix_closed_sites(device, planes, num_sites, num_closed):
    """A closed site comes back as its read, the mixture over its mass; the open sites' batch is unchanged."""
    (out, shift_out, mass_out, closed), (partials, shift, mass) = run_mix(
        device, planes, num_sites, 4, 640, num_closed=num_closed
    )

    reads = partials[:num_closed] / mass[:num_closed]
    passed, pcc = comp_pcc(reads, closed, 0.9999)
    assert passed, f"closed reads PCC {pcc}"
    # A weighted mean of unit-scale values, its weights bf16 at the MAC: ~2^-8 of a few units,
    # and a weight's rounding flips one bf16 step either way.
    torch.testing.assert_close(closed, reads, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(out, partials[num_closed:], rtol=2e-2, atol=1e-1)
    torch.testing.assert_close(shift_out, shift[num_closed:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mass_out, mass[num_closed:], rtol=1e-4, atol=1e-4)
