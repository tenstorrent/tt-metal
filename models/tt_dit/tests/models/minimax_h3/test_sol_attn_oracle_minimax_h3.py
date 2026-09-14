# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks on the Sol-Attn oracle (S2 step 1 of `pipelines/minimax_h3/PLAN_SOL.md`).

No device, no mesh. Two jobs:

* Pin the arithmetic. The vendored Triton reference cannot run here, so the
  transcription's correctness rests on invariants -- above all that `tau -> -inf`
  routes every block exact and must then equal dense attention.
* Report what the routing actually does at H3's production geometry, which is the
  input to the porting decision in S2: a threshold gives a data-dependent count,
  and we need to know what count before designing a fixed-width device path.
"""

from __future__ import annotations

import math

import pytest
import torch
from loguru import logger

from ....models.transformers.minimax_h3.sol_attn_oracle import (
    SOL_BLOCK_SIZE,
    SOL_DEFAULT_TAU,
    block_summaries,
    query_centroids,
    route,
    routing_threshold,
    sink_block_range,
    sol_attention,
)

HEAD_DIM = 128  # H3's head dim, and the reference's HEAD_DIM constant


def _qkv(tokens: int, heads: int = 2, seed: int = 0, dim: int = HEAD_DIM):
    gen = torch.Generator().manual_seed(seed)
    shape = (1, heads, tokens, dim)
    return (torch.randn(shape, generator=gen) for _ in range(3))


def _dense_reference(q, k, v, scale):
    logits = (q.float() @ k.float().transpose(-1, -2)) * scale
    return torch.softmax(logits, dim=-1) @ v.float()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
@pytest.mark.parametrize("tokens", [128, 320, 384])
def test_tau_minus_inf_is_dense(tokens, thresh_type):
    """Every block routes exact, so the approximate branch empties and the result is dense.

    This is the one invariant that pins the whole arithmetic: the softmax shift, the
    base-2 scores and the denominator all have to be right for it to hold. Sol's own
    correctness gate uses the same degenerate setting (`tau=-1000.0`).
    """
    q, k, v = _qkv(tokens)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    result = sol_attention(q, k, v, tau=-1.0e4, thresh_type=thresh_type)

    assert bool(result.exact.all()), "tau -> -inf must route every block exact"
    assert result.density == pytest.approx(1.0)
    torch.testing.assert_close(result.out, _dense_reference(q, k, v, scale), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tokens", [320, 384])
def test_local_window_and_sink_are_always_exact(tokens):
    """The +-1 local window and the sink range bypass the threshold entirely."""
    q, k, v = _qkv(tokens, seed=3)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    sink_tokens = 128
    kc, _ = block_summaries(k, v)
    q_bar = query_centroids(q)
    # a threshold nothing can clear, so only the unconditional rules can select
    threshold = torch.full(q_bar.shape[:-1], float("inf"))
    exact = route(q_bar, kc, threshold, scale=scale, tokens=tokens, sink_start=0, sink_tokens=sink_tokens)

    n_q, n_kv = exact.shape[-2], exact.shape[-1]
    lo, hi = sink_block_range(tokens, 0, sink_tokens)
    assert (lo, hi) == (0, sink_tokens // SOL_BLOCK_SIZE)
    for i in range(n_q):
        for j in range(n_kv):
            expected = abs(i - j) <= 1 or lo <= j < hi
            assert bool(exact[0, 0, i, j]) == expected, f"({i}, {j})"


def test_large_tau_keeps_only_the_unconditional_rules():
    """A high tau collapses selection onto the local window; the rest is approximated."""
    tokens = 640
    q, k, v = _qkv(tokens, seed=5)
    result = sol_attention(q, k, v, tau=50.0)
    n_q, n_kv = result.exact.shape[-2], result.exact.shape[-1]

    qi = torch.arange(n_q).view(n_q, 1)
    ki = torch.arange(n_kv).view(1, n_kv)
    torch.testing.assert_close(result.exact[0, 0], ((qi - ki).abs() <= 1).expand(n_q, n_kv), rtol=0, atol=0)
    assert torch.isfinite(result.out).all(), "the approximate branch must still normalise"


def test_exact_threshold_matches_diag_when_dimensions_are_uncorrelated():
    """`diag` drops the pooled keys' cross-dimension covariance; with none, they agree."""
    tokens, heads, dim = 1024, 1, 16
    gen = torch.Generator().manual_seed(11)
    # independent per-dimension pooled keys: the second moment is diagonal up to
    # sampling noise, which is what `diag` assumes
    kc = torch.randn(1, heads, tokens // SOL_BLOCK_SIZE, dim, generator=gen)
    kc = kc * torch.tensor([2.0**i for i in range(dim)]).view(1, 1, 1, dim) / 2.0**dim
    q_bar = torch.randn(1, heads, 4, dim, generator=gen)
    scale = 1.0 / math.sqrt(dim)

    diag = routing_threshold(q_bar, kc, tau=SOL_DEFAULT_TAU, scale=scale, thresh_type="diag")
    exact = routing_threshold(q_bar, kc, tau=SOL_DEFAULT_TAU, scale=scale, thresh_type="exact")
    assert _pcc(diag, exact) > 0.99, f"diag/exact PCC {_pcc(diag, exact)}"


def test_bad_thresh_type_is_rejected(expect_error):
    q_bar = torch.zeros(1, 1, 1, HEAD_DIM)
    kc = torch.zeros(1, 1, 2, HEAD_DIM)
    with expect_error(ValueError, "thresh_type"):
        routing_threshold(q_bar, kc, scale=1.0, thresh_type="mean")


# H3's packed sequence at 1344x768: prefix rows plus the video grid, padded to the
# 64-token VSA tile. `test_vsa_attention_minimax_h3.py` carries the same geometry.
_H3_SHAPES = {
    "15s_768p": 115712,
    "10s_768p": 77824,
}


@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
@pytest.mark.parametrize("shape", list(_H3_SHAPES), ids=list(_H3_SHAPES))
def test_routing_density_at_h3_geometry(shape, thresh_type):
    """What the threshold selects at production size, against VSA's fixed 10%.

    Reports rather than asserts a target: this is the number S2 needs before a
    fixed-width device path can be sized. The only assertion is that the count is
    bounded and data-dependent -- if every row selected everything, threshold
    routing would buy nothing over dense.
    """
    tokens = _H3_SHAPES[shape]
    n_blocks = tokens // SOL_BLOCK_SIZE
    gen = torch.Generator().manual_seed(17)
    # the routing path never touches per-token K/V, only the pooled summaries, so the
    # oracle can run at real geometry by generating the summaries directly
    kc = torch.randn(1, 1, n_blocks, HEAD_DIM, generator=gen)
    q_bar = torch.randn(1, 1, n_blocks, HEAD_DIM, generator=gen)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    threshold = routing_threshold(q_bar, kc, tau=SOL_DEFAULT_TAU, scale=scale, thresh_type=thresh_type)
    exact = route(q_bar, kc, threshold, scale=scale, tokens=tokens)

    per_row = exact[0, 0].sum(dim=-1).float()
    density = float(per_row.mean()) / n_blocks
    logger.info(
        f"{shape} {thresh_type}: {n_blocks} blocks, tau={SOL_DEFAULT_TAU} -> "
        f"per-row selected min {int(per_row.min())} / mean {float(per_row.mean()):.1f} / "
        f"max {int(per_row.max())} ({density:.2%} density; VSA at sparsity 0.9 is 10.00%)"
    )

    assert per_row.min() >= 3, "the local window alone selects three blocks off the diagonal"
    assert density < 0.5, f"threshold routing selected {density:.2%}: no sparsity to exploit"
    assert float(per_row.max()) > float(per_row.min()), "the count must vary by row, or tau is inert"


# The gate a device Sol-Attn is held to: PCC against the fp32 oracle at the bf16 floor `vsa_sdpa`
# already demonstrates (0.99969, which VSA_README names as the floor). A tighter target is not
# reachable without an fp32-accumulating kernel, and a mathematical reference cannot support one
# either -- the emulated tier below lands ~20x tighter than the real kernel because it models the
# rounding points but not the per-visit accumulation chain. Correctness beyond the format is carried
# by determinism and by `tau -> -inf` reducing to dense, not by a tighter PCC.
DEVICE_VS_FP32_ORACLE_PCC = 0.9996
DEVICE_VS_BF16_ORACLE_PCC = 0.99999  # self-consistency of the emulated path only, not a device gate


@pytest.mark.parametrize("tokens", [320, 640])
def test_bf16_emulation_brackets_the_two_gates(tokens):
    """Calibrate the tiers: bf16 emulation must sit above the fp32 floor and below fp32 itself.

    Without this, `DEVICE_VS_FP32_ORACLE_PCC` is a guess. It measures how much of the eventual
    device-vs-fp32 gap is the number format rather than the kernel, which is the number to quote when
    a device run lands between the two tiers.
    """
    q, k, v = _qkv(tokens, seed=7)
    fp32 = sol_attention(q, k, v, tau=SOL_DEFAULT_TAU)
    bf16 = sol_attention(q, k, v, tau=SOL_DEFAULT_TAU, emulate_bf16=True)

    torch.testing.assert_close(fp32.exact, bf16.exact, rtol=0, atol=0)
    pcc = _pcc(fp32.out, bf16.out)
    logger.info(f"tokens={tokens}: bf16-emulated vs fp32 oracle PCC {pcc:.6f} (format-only loss)")
    assert (
        pcc > DEVICE_VS_FP32_ORACLE_PCC
    ), f"format-only loss {pcc} already below the fp32 gate; the gate is too tight to be meaningful"
    assert pcc < 1.0 - 1e-9, "bf16 emulation is a no-op; the rounding is not being applied"


def test_bf16_emulation_is_self_consistent():
    """The bf16 path must be deterministic, or the 99.999 % device gate has no fixed target."""
    q, k, v = _qkv(384, seed=9)
    a = sol_attention(q, k, v, tau=SOL_DEFAULT_TAU, emulate_bf16=True)
    b = sol_attention(q, k, v, tau=SOL_DEFAULT_TAU, emulate_bf16=True)
    torch.testing.assert_close(a.out, b.out, rtol=0, atol=0)
    assert _pcc(a.out, b.out) > DEVICE_VS_BF16_ORACLE_PCC
