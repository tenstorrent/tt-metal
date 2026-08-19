# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused MoE gate (`moe_grouped_topk`) vs MiniMax-M3's routing rule, at M3's exact shape.

Guards two things the fused gate depends on, both of which are easy to break from outside this file:

1. **The op supports M3's shape at all.** `moe_grouped_topk`'s DeepSeek constraints
   (n_groups==8, experts==256, n_activated_experts==8) sit behind its GROUPED branch; the
   single-group branch (n_groups=1, which is M3) accepts any tile-aligned expert count and any
   k <= 64. M3 is 128 experts / top-4 — the only one of the four models that is not top-8, so it is
   the one a future tightening of those asserts would break first.

2. **It computes M3's rule, and does so MORE faithfully than the legacy chain.** This is the
   justification for making it the default, and it is a claim about relative accuracy, so it needs a
   test rather than a comment. The legacy path runs sigmoid AND the bias add in bf16; around O(1)
   scores bf16's ~0.004 resolution makes near-ties at the top-k boundary resolve arbitrarily across
   128 experts. The op computes in fp32 internally.

Both paths are legitimate implementations of the rule — they simply disagree on genuinely tied
tokens — so this asserts on the FAITHFULNESS ORDERING (fused closer to fp32 than legacy), not on the
two paths agreeing with each other.

Single-device: the gate is per-token and replicated across TP, so the routing arithmetic is
device-count independent. See test_ep_moe_vs_ref.py for the mesh-level MoE check.
"""

import pytest
import torch
from loguru import logger

import ttnn

HIDDEN, E, TOPK, TOKENS = 6144, 128, 4, 640
ROUTED_SCALING_FACTOR = 2.0

# Measured floors, not aspirations. On this seed the fused gate agrees with an fp32 reference on
# 99.4 % of tokens and the legacy chain on 96.4 %; the gap is the whole point of the change. Set with
# ~1 pp of headroom so an SFPU sigmoid tweak does not fail the suite, while a real regression (the op
# silently falling back to bf16 arithmetic, say) collapses fused toward legacy and trips it.
MIN_FUSED_AGREEMENT_PCT = 98.5
MAX_LEGACY_AGREEMENT_PCT = 98.0


def _m3_reference(logits, bias, *, fp32_arithmetic):
    """MiniMax-M3's rule (MiniMaxM3SparseMoeBlock.route_tokens_to_experts).

    fp32_arithmetic=True models the fused kernel (bf16 only at the inputs); False models the legacy
    op chain, where every step runs in bf16. Selection uses biased scores; the returned weights are
    the UNBIASED sigmoid values, normalized, then scaled.
    """
    lg = logits.to(torch.bfloat16)
    if fp32_arithmetic:
        weights_all = torch.sigmoid(lg.float())
        scores_for_choice = weights_all + bias.float()
    else:
        weights_all = torch.sigmoid(lg)
        scores_for_choice = (weights_all + bias.to(torch.bfloat16)).float()
        weights_all = weights_all.float()
    _, idx = torch.topk(scores_for_choice, TOPK, dim=-1)
    w = weights_all.gather(-1, idx)
    return idx, w / w.sum(-1, keepdim=True) * ROUTED_SCALING_FACTOR


def _expert_set_agreement(a, b):
    """% of tokens whose selected expert SET matches. Order within the top-k is not meaningful."""
    return (a.sort(-1).values == b.sort(-1).values).all(-1).float().mean().item() * 100.0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_fused_gate_matches_m3_rule(device, reset_seeds):
    from models.demos.minimax_m3.tt.topk import route_tokens_to_experts_fused

    logits = torch.randn(1, 1, TOKENS, E, dtype=torch.float32) * 3.0
    bias_1d = torch.randn(E, dtype=torch.float32) * 0.5
    # The device holds the bias as bf16, and it dominates top-k selection, so an fp32 reference bias
    # would choose different experts for reasons that have nothing to do with the op under test.
    bias_as_stored = bias_1d.to(torch.bfloat16).float()

    tt_logits = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # The op requires bias.logical_shape() == scores.logical_shape() (TopKRouter builds this at init).
    tt_bias = ttnn.from_torch(
        bias_1d.reshape(1, -1).expand(TOKENS, E).reshape(1, 1, TOKENS, E).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    idx_tt, w_tt = route_tokens_to_experts_fused(tt_logits, TOPK, tt_bias, ROUTED_SCALING_FACTOR)
    assert idx_tt.dtype == ttnn.uint16, f"masked_bincount consumes uint16; got {idx_tt.dtype}"
    assert idx_tt.layout == ttnn.TILE_LAYOUT, f"expected TILE indices; got {idx_tt.layout}"

    got_idx = ttnn.to_torch(idx_tt).to(torch.int64).reshape(1, 1, TOKENS, TOPK)
    got_w = ttnn.to_torch(w_tt).float().reshape(1, 1, TOKENS, TOPK)

    assert got_idx.min() >= 0 and got_idx.max() < E, f"indices out of range: [{got_idx.min()}, {got_idx.max()}]"
    row_sum = got_w.sum(-1).mean().item()
    assert abs(row_sum - ROUTED_SCALING_FACTOR) < 0.02, (
        f"top-k weights must normalize to 1 then scale by routed_scaling_factor "
        f"({ROUTED_SCALING_FACTOR}); mean row sum is {row_sum:.4f}"
    )

    ref_idx, _ = _m3_reference(logits, bias_as_stored, fp32_arithmetic=True)
    legacy_idx, _ = _m3_reference(logits, bias_as_stored, fp32_arithmetic=False)

    fused_pct = _expert_set_agreement(got_idx, ref_idx)
    legacy_pct = _expert_set_agreement(legacy_idx, ref_idx)
    logger.info(
        f"expert-set agreement with the fp32 reference: fused {fused_pct:.2f}% / legacy {legacy_pct:.2f}%; "
        f"fused vs legacy {_expert_set_agreement(got_idx, legacy_idx):.2f}%"
    )

    assert fused_pct >= MIN_FUSED_AGREEMENT_PCT, (
        f"fused gate agrees with the fp32 reference on only {fused_pct:.2f}% of tokens "
        f"(floor {MIN_FUSED_AGREEMENT_PCT}%) — the op may no longer be computing in fp32"
    )
    assert legacy_pct <= MAX_LEGACY_AGREEMENT_PCT, (
        f"the legacy bf16 chain now agrees with the fp32 reference on {legacy_pct:.2f}% of tokens, "
        f"above the {MAX_LEGACY_AGREEMENT_PCT}% ceiling this test assumed. If the legacy path became "
        f"more accurate, the premise for defaulting to the fused gate needs re-checking."
    )
    assert fused_pct > legacy_pct, "the fused gate must be at least as faithful as the legacy chain"
