# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

HIDDEN_SIZE = 7168
EPS = 1e-5


def reference(stats, num_sites, step=lambda t: t):
    """The chain the op fuses. `step` rounds every intermediate: the identity gives the exact
    fp32 answer, bf16 rounding gives what `TtAttnRes.inter_block` computed before the op,
    when it narrowed the reduced statistics to bf16 and ran the chain there."""
    stats = step(stats)
    sum_squares, dots = stats[..., :1], stats[..., 1 : 1 + num_sites]
    reciprocal_rms = step(torch.rsqrt(step(step(sum_squares / HIDDEN_SIZE) + EPS)))
    scores = step(dots * reciprocal_rms)  # [1, C, N, R]
    shift = scores.amax(dim=1, keepdim=True)
    exponentials = step(torch.exp(step(scores - shift)))
    mass = step(exponentials.sum(dim=1, keepdim=True))
    site_major = lambda t: t.permute(3, 1, 2, 0)  # [1, ., N, R] -> [R, ., N, 1]
    return site_major(exponentials), site_major(shift), site_major(mass)


@pytest.mark.parametrize("num_partials", [1, 4])
@pytest.mark.parametrize("num_candidates", [1, 6, 7, 8])
@pytest.mark.parametrize("num_sites", [1, 5, 24])
@pytest.mark.parametrize("num_tokens", [64, 640, 5120])
def test_attn_res_scores(device, num_partials, num_candidates, num_sites, num_tokens):
    torch.manual_seed(0)
    # Sums of squares of a d-wide row of unit-scale values, and dots of the same rows
    # with 0.02-scale queries: the magnitudes the model hands the op.
    sum_squares = HIDDEN_SIZE * (0.5 + torch.rand(1, num_candidates, num_tokens, 1))
    dots = torch.randn(1, num_candidates, num_tokens, num_sites) * 0.02 * HIDDEN_SIZE**0.5 * 4
    padding = torch.zeros(1, num_candidates, num_tokens, 31 - num_sites)
    stats = torch.cat([sum_squares, dots, padding], dim=-1)

    # The op sums partials laid side by side on the last dim, as an all-gather across the
    # tensor-parallel ranks leaves them. Split the statistics into that many random shares.
    shares = torch.rand(num_partials, *stats.shape)
    shares = stats * shares / shares.sum(dim=0)
    gathered = torch.cat(list(shares), dim=-1)

    tt_stats = ttnn.from_torch(gathered, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_exp, tt_shift, tt_mass = ttnn.experimental.deepseek_prefill.attn_res_scores(
        tt_stats, num_sites=num_sites, inv_hidden_size=1.0 / HIDDEN_SIZE, eps=EPS
    )

    exact = reference(stats, num_sites)
    composed = reference(stats, num_sites, step=lambda t: t.bfloat16().float())
    names = ("exponentials", "shift", "mass")
    for name, tt_out, ref, old in zip(names, (tt_exp, tt_shift, tt_mass), exact, composed):
        assert tuple(tt_out.shape) == tuple(ref.shape), f"{name}: {tuple(tt_out.shape)} vs {tuple(ref.shape)}"
        assert tt_out.dtype == ttnn.float32
        out = ttnn.to_torch(tt_out)
        # The score multiply runs on the FPU, whose operands are tf32 (10 mantissa bits), and
        # exp turns a score's absolute error into a relative one: at |s| ~ 20 that is ~2%.
        torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2, msg=lambda m: f"{name}: {m}")
        # Never worse than the bf16 chain the op replaced, on the outputs the read consumes.
        error, old_error = (out - ref).abs().max().item(), (old - ref).abs().max().item()
        assert error <= old_error, f"{name}: max error {error:.3e} exceeds the composed chain's {old_error:.3e}"
