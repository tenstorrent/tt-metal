# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole-only reimplementation of the recurrent gated-delta-rule decode step.

Same contract as the shared ``recurrent_gated_delta_rule_decode_ttnn``
(models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py), except ``q`` is NOT
promoted to fp32 under ``high_precision``: q feeds only the read-only ``o = q @ h`` and is never
written back into the state, so promoting it cannot help the drift the flag exists to prevent.

Local to qwen36 rather than an edit to the shared module (see wh_compat.py). Reuses the shared
helpers verbatim and changes only the top-level orchestration. gdn/tp.py dispatches here on
Wormhole; Blackhole keeps calling the shared function unchanged.
"""

import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)

_OUTER_L1_BUDGET_BYTES = 8 << 20  # see wh_decode_fork_applies below


def wh_decode_fork_applies(B, Hq, K, V, gqa_repeat=1, high_precision=False):
    """Should this shape take the Wormhole fork rather than the shared upstream decode?

    The fork materializes the full [B,H,K,V] outer product in L1, so take it only while that fits;
    widen the budget only with a measurement. gdn/tp.py calls this once and uses the answer for
    BOTH the q/k fusion and the choice of decode function, so the two can never disagree -- a
    caller that fused while falling back to upstream would apply the gated-norm epsilon
    compensation to an uncompensated ``o``. Keep the budget evaluated only here.

    The arch gate is ``is_blackhole()``, NOT a narrower device check: for this model's shape the
    upstream kernel OVERFLOWS on Wormhole -- the recurrence output saturates, the following
    rms_norm underflows, and every GDN decode returns zeros -- so upstream is a correctness
    failure there, not just a slower path. That holds even at high_precision=False, where the two
    are otherwise equivalent. Shapes over the budget fall back to upstream anyway and rely on
    wh_compat's state-write override to keep them off L1."""
    if is_blackhole():
        return False
    itemsize = 4 if high_precision else 2
    return B * (Hq * gqa_repeat) * K * V * itemsize <= _OUTER_L1_BUDGET_BYTES


def _write_state_wh(h, k_row, delta, beta_t, outer_bf8=False):
    """Decode-only h = h + beta*(k(x)delta); bit-equivalent to fused_decay_and_write_ttnn's
    apply_decay=False branch, with upstream's dead `decay` reshape dropped.

    Rules, each one load-bearing:
    * k(x)delta is RANK-1, so it is a broadcast multiply, not a matmul -- and the multiply is also
      the more accurate of the two (no accumulate-then-round).
    * Build k_col with transpose(k_row, 2, 3), never reshape->reshape: those move the singleton
      between the two TILE-tiled dims and force a retile each time.
    * Scale by beta on d_row BEFORE the broadcast, not on the [B,H,K,V] result after.
    * ``outer_bf8`` MUST stay False unless h is already bf8: rounding the increment compounds over
      every decode step of every GDN layer and costs model-level logits PCC.
    * Do NOT fuse the multiply+add into one ttnn.mac. It is correct but much slower per op on this
      hardware; ttnn.mac is not a free fusion of a broadcast multiply and an add here."""
    B, H, V = h.shape[0], h.shape[1], h.shape[3]
    _L1 = ttnn.L1_MEMORY_CONFIG

    # [B,H] -> [B,H,1,1] via unsqueeze x2, not reshape -- see decay_bhkv in the caller.
    beta_expanded = beta_t
    # BOTH operands of the outer product must share a dtype: the broadcast multiply below is far
    # slower on mixed dtypes, so do not feed this the pre-cast bf16 k_row to save the transpose.
    k_col = ttnn.transpose(k_row, 2, 3, memory_config=_L1)
    d_row = delta  # already [B,H,1,V]

    d_scaled = ttnn.multiply(d_row, beta_expanded, memory_config=_L1)
    if outer_bf8:
        outer_dtype = None if h.dtype == ttnn.float32 else ttnn.bfloat8_b
    else:
        outer_dtype = ttnn.bfloat8_b if h.dtype == ttnn.bfloat8_b else None
    outer = ttnn.multiply(k_col, d_scaled, memory_config=_L1, dtype=outer_dtype)
    return ttnn.add(h, outer, memory_config=_L1)


def recurrent_gated_delta_rule_decode_wh(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    high_precision=False,
    gqa_repeat=1,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_fused=None,
    v_rowed=False,
):
    """Wormhole variant of recurrent_gated_delta_rule_decode_ttnn.

    Identical to upstream except: q is NOT typecast to fp32 at all. It stays in its native dtype
    through L2-norm, the scale multiply, the reshape to q_row (cheaper at bf16's data width) and
    into the final q @ h matmul, which takes the mixed (bf16, fp32) operands directly -- casting to
    match h would be pure cost. q feeds only the read-only o = q @ h and is never written back into
    the state, so leaving it narrow cannot cause the drift high_precision exists to prevent. Every
    other value (k, v, beta, g, h) follows the exact same fp32 path as upstream. When
    high_precision=False this is bit-identical to upstream (no casts happen either way).
    """
    # q/k arrive with Hq = H // gqa_repeat heads; the expansion happens after the L2-norm, so those ops run narrower.
    if qk_fused is not None:
        # fused [B,1,2*Hq,K]: q heads in rows [0,Hq), k heads in rows [Hq,2*Hq)
        B, Hq, K = qk_fused.shape[0], qk_fused.shape[2] // 2, qk_fused.shape[3]
    else:
        B, Hq, K = q.shape[0], q.shape[2], q.shape[3]
    H = Hq * gqa_repeat
    V = v.shape[3]

    if high_precision:
        pass  # g's cast moved down to g_t -- see the note there
        # ONLY g here (k is cast below); v/beta stay bf16 since delta is pinned fp32. Neither cast is removable -- the state increment would round to bf16 and compound.

    # L2 norm. q is intentionally left at its incoming dtype here -- see module docstring.
    if scale is None:
        scale = K**-0.5
    # q's L2 1/sqrt(K) folds into the attention scale and rides inside rms_norm; explicit multiplies remain for callers without fold weights.
    if qk_fused is not None:
        # q and k run as ONE tensor through norm, transpose and the GQA expansion -- see the block
        # comment at the split below.
        qk = ttnn.rms_norm(qk_fused, epsilon=1e-6 / K, weight=k_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    elif q_norm_weight is not None and k_norm_weight is not None:
        q = ttnn.rms_norm(q, epsilon=1e-6 / K, weight=q_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=1e-6 / K, weight=k_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        q = ttnn.rms_norm(q, epsilon=1e-6 / K)
        q = ttnn.multiply(q, scale * (K**-0.5), memory_config=ttnn.L1_MEMORY_CONFIG)
        k = l2_norm_ttnn(k, dim=-1)

    # All FOUR transposes stay: they bridge the COMPACT [B,1,H,K] the norms want and the PER-HEAD-BATCHED [B,H,1,K] the matmuls need.
    if qk_fused is not None:
        qk_row = ttnn.transpose(qk, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B,2*Hq,1,K]
        ttnn.deallocate(qk)
        if gqa_repeat > 1:
            # repeat_interleave repeats each head in place, so the q and k blocks stay contiguous and the split is one dim-1 cut.
            qk_row = ttnn.repeat_interleave(qk_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Split on dim 1, which is NOT one of the two dims a tile subdivides, so these are cheap
        # batch-style cuts -- unlike a dim-2 split, which would land mid-tile at Hq=4.
        q_row = ttnn.slice(qk_row, (0, 0, 0, 0), (B, H, 1, K), memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.slice(qk_row, (0, H, 0, 0), (B, 2 * H, 1, K), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(qk_row)
    else:
        q_row = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    if qk_fused is None and gqa_repeat > 1:
        # Expand Hq -> H HERE, not in the caller: only this layout clears repeat_interleave's TILE-native gate (dim < rank-2), and everything above runs narrower.
        q_row = ttnn.repeat_interleave(q_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.repeat_interleave(k_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    # k's cast sits HERE so everything upstream runs bf16; it is BIT-EXACT, and deleting it to pin the consumers instead is far slower.
    if high_precision:
        k_row = ttnn.typecast(k_row, ttnn.float32)
    # Keep delta on [B,H,V]; v_rowed folds this transpose into a reshape the caller had to do, and is only safe under the fork gate.
    v_t = v if v_rowed else ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # ->[B,H,1,V]
    beta_t = ttnn.reshape(beta, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_t = ttnn.reshape(g, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    if high_precision:
        # Reshape at bf16, widen after; the cast MUST still happen -- the fused exp below is only correct on an fp32 operand.
        g_t = ttnn.typecast(g_t, ttnn.float32)

    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)

    # NO explicit DRAM->L1 hoist: the decay multiply already writes to L1, so a to_memory_config here would pay twice.

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # Decay before read. [B,H] -> [B,H,1,1] via two unsqueeze calls, never one reshape (that crosses a pair of tiled dims).
    _L1 = ttnn.L1_MEMORY_CONFIG
    g_bhkv = g_t
    # exp(g) rides in as a fused activation -- ONLY CORRECT BECAUSE g IS FP32; the same fold on a bf16 operand silently computes garbage.
    h = ttnn.multiply(h, g_bhkv, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)

    # v_read = k @ h (decayed state)
    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # Delta + state write (no re-decay). k_row feeds _write_state_wh directly (transposed there);
    # no intermediate [B,H,K] reshape needed since k_row had no other consumer after this point.
    delta = ttnn.subtract(v_t, v_read, memory_config=_L1, dtype=ttnn.float32 if high_precision else None)
    # outer_bf8 stays False: rounding the increment to bf8 compounds across every decode step of every GDN layer.
    h = _write_state_wh(h=h, k_row=k_row, delta=delta, beta_t=beta_t, outer_bf8=False)

    # o = q @ h with q_row at its own dtype: ttnn.matmul takes mixed operands, so matching it to h first is pure cost.
    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # o stays in the matmul's native [B,H,1,V]; both layouts flatten to the same b,(h,v) order, so the upstream leg needs no fixup.
    return o_t, h
