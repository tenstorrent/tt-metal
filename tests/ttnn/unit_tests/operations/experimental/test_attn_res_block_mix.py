# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = run_for_blackhole()

HIDDEN_SIZE = 7168
SHARD_WIDTH = 1792  # HIDDEN_SIZE over a 4-way tensor-parallel axis
EPS = 1e-5


def random_stats(num_candidates, num_tokens, num_sites):
    """Statistics at the magnitudes the model hands the op: sums of squares of unit-scale rows
    over the full hidden size, and dots with 0.02-scale queries."""
    sum_squares = HIDDEN_SIZE * (0.5 + torch.rand(1, num_candidates, num_tokens, 1))
    dots = torch.randn(1, num_candidates, num_tokens, num_sites) * 0.02 * HIDDEN_SIZE**0.5 * 4
    padding = torch.zeros(1, num_candidates, num_tokens, 31 - num_sites)
    return torch.cat([sum_squares, dots, padding], dim=-1)


def reference(stats, v, num_sites):
    sum_squares, dots = stats[..., :1], stats[..., 1 : 1 + num_sites]
    scores = dots * torch.rsqrt(sum_squares / HIDDEN_SIZE + EPS)  # [1, C, N, R]
    shift = scores.amax(dim=1, keepdim=True)
    weights = torch.exp(scores - shift)
    mass = weights.sum(dim=1, keepdim=True)
    partials = torch.einsum("bcnr,bcnd->rbnd", weights, v)  # [R, 1, N, d]
    site_major = lambda t: t.permute(3, 1, 2, 0)  # [1, 1, N, R] -> [R, 1, N, 1]
    return partials, site_major(shift), site_major(mass)


@pytest.mark.parametrize("planes", [[1], [6], [6, 1], [3, 1, 1, 2]], ids=lambda p: "+".join(map(str, p)))
@pytest.mark.parametrize("num_sites", [1, 24])
@pytest.mark.parametrize("num_partials", [1, 4])
@pytest.mark.parametrize("num_tokens", [64, 640])
def test_attn_res_block_mix(device, planes, num_sites, num_partials, num_tokens):
    torch.manual_seed(0)
    num_candidates = sum(planes)
    snapshots = [torch.randn(1, s, num_tokens, SHARD_WIDTH).bfloat16() for s in planes]
    stats = random_stats(num_candidates, num_tokens, num_sites)
    # Partials side by side on the last dim, as an all-gather across the tensor-parallel axis
    # leaves them: random shares of the statistics.
    shares = torch.rand(num_partials, *stats.shape)
    gathered = torch.cat(list(stats * shares / shares.sum(dim=0)), dim=-1)

    to_device = lambda t, dtype: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_partials, tt_shift, tt_mass = ttnn.experimental.deepseek_prefill.attn_res_block_mix(
        [to_device(s, ttnn.bfloat16) for s in snapshots],
        to_device(gathered, ttnn.float32),
        num_sites=num_sites,
        inv_hidden_size=1.0 / HIDDEN_SIZE,
        eps=EPS,
    )

    partials, shift, mass = reference(stats, torch.cat(snapshots, dim=1).float(), num_sites)
    assert tuple(tt_partials.shape) == (num_sites, 1, num_tokens, SHARD_WIDTH)
    assert tuple(tt_shift.shape) == tuple(tt_mass.shape) == (num_sites, 1, num_tokens, 1)

    # Scores multiply through tf32 operands, and the weights reach the MAC as bf16.
    torch.testing.assert_close(ttnn.to_torch(tt_shift), shift, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(ttnn.to_torch(tt_mass), mass, rtol=3e-2, atol=3e-2)
    out = ttnn.to_torch(tt_partials).float()
    passed, pcc = comp_pcc(partials, out, 0.9999)
    assert passed, f"partials PCC {pcc}"
    # Elementwise, a partial sums up to C unit-scale values with weights that are bf16 at the
    # MAC and carry the scores' tf32 error through exp: ~2^-8 of up to C * 4, so 0.1.
    torch.testing.assert_close(out, partials, rtol=2e-2, atol=1e-1)
