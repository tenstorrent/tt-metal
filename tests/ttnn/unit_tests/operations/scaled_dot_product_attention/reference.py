# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference for scaled_dot_product_attention (Flash Attention).

Decomposed into the same Compute Phases as op_design.md — one function per
phase. Each function computes and PRINTS the checkpoint slice for that phase
so the implementer's TDD probe can compare device-side DPRINT/TSLICE output
against the reference.

Pinned shape: (1, 1, 128, 128) — S_q=S_kv=128, D=128, B=H=1.
This gives:
  - 4 Q-tiles (B_q_t=4) and 4 KV-tiles (B_kv_t=4) → exercises multi-tile paths.
  - D_t=4 (128/32) → multi-tile head dim.
  - num_q_blocks=1, num_kv_blocks=1 (one block iteration) → simplest TDD path.

Seed: torch.manual_seed(0) for reproducibility.

The reference implements the Flash Attention online-softmax recurrence in
fp32 torch — mathematically equivalent to torch.nn.functional.sdpa but
decomposed to expose every intermediate the kernel produces.
"""

from __future__ import annotations

import math
import torch

torch.manual_seed(0)

# Pinned shape and block params (must match op_design.md)
B, H, S_q, D = 1, 1, 128, 128
S_kv = S_q  # self-attention for the pinned shape
B_q_t = 4  # 4 Q-tiles (128 rows)
B_kv_t = 4  # 4 KV-tiles (128 cols)
D_t = D // 32  # 4 head-dim tiles
SCALE = 1.0 / math.sqrt(D)

# Generate pinned inputs
_Q = torch.randn(B, H, S_q, D, dtype=torch.float32)
_K = torch.randn(B, H, S_kv, D, dtype=torch.float32)
_V = torch.randn(B, H, S_kv, D, dtype=torch.float32)


def _print_slice(name, tensor, rows=4, cols=4):
    """Print a small slice of a tensor for checkpoint comparison."""
    sliced = tensor[:rows, :cols]
    print(f"[{name}] shape={list(tensor.shape)}, slice[:{rows},:{cols}]:")
    for r in range(min(rows, sliced.shape[0])):
        row_str = "  ".join(f"{s:10.6f}" for s in sliced[r])
        print(f"  [{row_str}]")
    print()


# =============================================================================
# Phase 0: Init — load Q-block, init m_i=-inf, l_i=0, O_i=0
# =============================================================================


def reference_phase_init():
    """Initialize running state for the first (and only) Q-block.

    m_i: (B_q_t * 32,) per-row running max, init to -inf
    l_i: (B_q_t * 32,) per-row running sum, init to 0
    O_i: (B_q_t * 32, D) running output, init to 0
    """
    Q_blk = _Q[0, 0, : B_q_t * 32, :]  # (128, 128)

    m_i = torch.full((B_q_t * 32,), float("-inf"), dtype=torch.float32)
    l_i = torch.zeros((B_q_t * 32,), dtype=torch.float32)
    O_i = torch.zeros((B_q_t * 32, D), dtype=torch.float32)

    _print_slice("m_i (init)", m_i.unsqueeze(-1))  # treat as column
    _print_slice("l_i (init)", l_i.unsqueeze(-1))
    _print_slice("O_i (init)", O_i)

    return Q_blk, m_i, l_i, O_i


# =============================================================================
# Phase 1: QK^T score matmul: S = Q @ K^T (pre-scale)
# =============================================================================


def reference_phase_qkt(Q_blk, m_i, l_i, O_i):
    """Compute S = Q_blk @ K_blk^T (before scaling).

    Score block: (B_q_t*32, B_kv_t*32) = (128, 128)
    """
    K_blk = _K[0, 0, : B_kv_t * 32, :]  # (128, 128)
    scores = Q_blk @ K_blk.T  # (128, 128)

    _print_slice("cb_scores (Q@K^T)", scores)
    return scores, m_i, l_i, O_i


# =============================================================================
# Phase 2: Scale: S *= scale
# =============================================================================


def reference_phase_scale(scores, m_i, l_i, O_i):
    """Scale scores by 1/sqrt(D)."""
    scores = scores * SCALE

    _print_slice("cb_scores (scaled)", scores)
    return scores, m_i, l_i, O_i


# =============================================================================
# Phase 3: Mask add (no mask for the pinned shape → passthrough)
# =============================================================================


def reference_phase_mask(scores, m_i, l_i, O_i):
    """Apply additive mask. Pinned shape uses no mask (mask_mode=none),
    so this is a passthrough (copy to cb_scores_masked).
    """
    scores_masked = scores.clone()  # passthrough

    _print_slice("cb_scores_masked (passthrough)", scores_masked)
    return scores_masked, m_i, l_i, O_i


# =============================================================================
# Phase 4: Row-max: m_blk = max(S, dim=-1)
# =============================================================================


def reference_phase_rowmax(scores_masked, m_i, l_i, O_i):
    """Compute row-wise max of the score block."""
    m_blk = scores_masked.max(dim=-1).values  # (128,)

    _print_slice("cb_max_new (row-max)", m_blk.unsqueeze(-1))
    return scores_masked, m_blk, m_i, l_i, O_i


# =============================================================================
# Phase 5: Compute alpha = exp(m_old - m_new)
# =============================================================================


def reference_phase_alpha(scores_masked, m_blk, m_i, l_i, O_i):
    """Compute alpha = exp(m_i_old - m_new) per row."""
    alpha = torch.exp(m_i - m_blk)  # (128,)

    _print_slice("cb_alpha (exp(m_old - m_new))", alpha.unsqueeze(-1))
    return scores_masked, m_blk, alpha, m_i, l_i, O_i


# =============================================================================
# Phase 6: Rescale O: O *= alpha
# =============================================================================


def reference_phase_rescale_o(scores_masked, m_blk, alpha, m_i, l_i, O_i):
    """Rescale running output: O_i = alpha * O_i."""
    O_i = alpha.unsqueeze(-1) * O_i  # (128, 128)

    _print_slice("cb_o (rescaled by alpha)", O_i)
    return scores_masked, m_blk, alpha, m_i, l_i, O_i


# =============================================================================
# Phase 7: Rescale l: l *= alpha
# =============================================================================


def reference_phase_rescale_l(scores_masked, m_blk, alpha, m_i, l_i, O_i):
    """Rescale running sum: l_i = alpha * l_i."""
    l_i = alpha * l_i  # (128,)

    _print_slice("cb_sum_old (rescaled by alpha)", l_i.unsqueeze(-1))
    return scores_masked, m_blk, alpha, m_i, l_i, O_i


# =============================================================================
# Phase 8: Subtract m_new: S -= m_new (broadcast Col)
# =============================================================================


def reference_phase_subtract_max(scores_masked, m_blk, alpha, m_i, l_i, O_i):
    """Subtract row max from scores: S = S - m_new (broadcast across columns)."""
    scores_masked = scores_masked - m_blk.unsqueeze(-1)  # (128, 128)

    _print_slice("cb_scores_masked (S - m_new)", scores_masked)
    return scores_masked, m_blk, alpha, m_i, l_i, O_i


# =============================================================================
# Phase 9: Exp: P = exp(S - m_new)
# =============================================================================


def reference_phase_exp(scores_masked, m_blk, alpha, m_i, l_i, O_i):
    """Element-wise exp of rescaled scores."""
    exp_scores = torch.exp(scores_masked)  # (128, 128)

    _print_slice("cb_exp_scores (exp(S - m_new))", exp_scores)
    return exp_scores, m_blk, alpha, m_i, l_i, O_i


# =============================================================================
# Phase 10: Row-sum: l_blk = sum(P, dim=-1)
# =============================================================================


def reference_phase_rowsum(exp_scores, m_blk, alpha, m_i, l_i, O_i):
    """Compute row-wise sum of exp scores."""
    l_blk = exp_scores.sum(dim=-1)  # (128,)

    _print_slice("cb_sum_new (row-sum of exp)", l_blk.unsqueeze(-1))
    return exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i


# =============================================================================
# Phase 11: Update l: l_i += l_blk
# =============================================================================


def reference_phase_update_l(exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i):
    """Update running sum: l_i = l_i + l_blk."""
    l_i = l_i + l_blk  # (128,)

    _print_slice("cb_sum_old (l_i + l_blk)", l_i.unsqueeze(-1))
    return exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i


# =============================================================================
# Phase 12: PV matmul: O += P @ V
# =============================================================================


def reference_phase_pv(exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i):
    """Accumulate output: O_i = O_i + P @ V_blk."""
    V_blk = _V[0, 0, : B_kv_t * 32, :]  # (128, 128)
    O_i = O_i + exp_scores @ V_blk  # (128, 128)

    _print_slice("cb_o (O + P@V)", O_i)
    return m_blk, alpha, l_blk, m_i, l_i, O_i


# =============================================================================
# Phase 13: Update m: m_i = m_new
# =============================================================================


def reference_phase_update_m(m_blk, alpha, l_blk, m_i, l_i, O_i):
    """Update running max: m_i = m_new."""
    m_i = m_blk  # (128,)

    _print_slice("cb_max_old (m_i = m_new)", m_i.unsqueeze(-1))
    return m_i, l_i, O_i


# =============================================================================
# Phase 14: Write output (final normalization)
# =============================================================================


def reference_phase_output(m_i, l_i, O_i):
    """Final output: O = O / l_i (normalize by row sum).

    The flash attention kernel produces unnormalized O_i; the final
    division by l_i happens at the end of all KV-blocks.
    """
    output = O_i / l_i.unsqueeze(-1)  # (128, 128)

    _print_slice("output (O / l_i)", output)

    # Also verify against torch's SDPA for sanity
    expected = torch.nn.functional.scaled_dot_product_attention(_Q, _K, _V, scale=SCALE)
    _print_slice("torch_sdpa_reference", expected[0, 0])

    max_diff = (output - expected[0, 0]).abs().max().item()
    print(f"Max diff vs torch SDPA: {max_diff:.2e}")
    print()

    return output


# =============================================================================
# Full reference (for the acceptance test)
# =============================================================================


def flash_attention_reference(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
    """Full Flash Attention reference — fp32, online softmax.

    Handles arbitrary shapes, masks, causal, and scale. Used by the
    acceptance test.
    """
    original_dtype = Q.dtype
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    if is_causal and attn_mask is not None:
        raise ValueError("is_causal and attn_mask are mutually exclusive")

    D = Qf.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    B, H_q, S_q, _ = Qf.shape
    _, H_kv, S_kv, _ = Kf.shape

    # GQA/MQA: replicate K/V heads
    if H_q != H_kv:
        if H_q % H_kv != 0:
            raise ValueError(f"H_q ({H_q}) must be multiple of H_kv ({H_kv})")
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    output = torch.empty_like(Qf)

    for b in range(B):
        for h in range(H_q):
            Q_bh = Qf[b, h]  # (S_q, D)
            K_bh = Kf[b, h]  # (S_kv, D)
            V_bh = Vf[b, h]  # (S_kv, D)

            # Flash attention with block size = full sequence (single block)
            # For the reference, we use one big block since torch handles it
            scores = Q_bh @ K_bh.T * scale  # (S_q, S_kv)

            if is_causal:
                causal_mask = torch.zeros(S_q, S_kv, dtype=torch.float32)
                causal_mask.masked_fill_(
                    torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
                    float("-inf"),
                )
                scores = scores + causal_mask
            elif attn_mask is not None:
                mask_bh = attn_mask[b, h] if attn_mask.shape[1] > 1 else attn_mask[b, 0]
                scores = scores + mask_bh.to(torch.float32)

            # Numerically stable softmax
            max_scores = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - max_scores)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)
            weights = exp_scores / sum_exp
            output[b, h] = weights @ V_bh

    return output.to(original_dtype)


# =============================================================================
# STAGES — machine-readable index for the staged drafter driver
# =============================================================================

STAGES = [
    {
        "index": 0,
        "name": "init",
        "implement": "Op file (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate, entry point, default_compute_kernel_config) + program descriptor + kernel scaffolding (reader/writer/compute stubs). Kernel must be runnable at the first checkpoint.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_init",
            "checkpoint": "cb_max_old tile 0, first 4×4 elements — all -inf (running max initialized)",
            "pass_criteria": "All elements are -inf; m_i and l_i initialized correctly",
        },
    },
    {
        "index": 1,
        "name": "qkt_matmul",
        "implement": "QK^T score matmul via matmul_block (transpose=true, NoWaitNoPop on Q, WaitAndPopPerKBlock on K). Reader streams Q-block and first K-block.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_qkt",
            "checkpoint": "cb_scores tile 0 (first Q-row × first K-col), first 4×4 elements — Q @ K^T values",
            "pass_criteria": "Score values match reference Q@K^T (pre-scale) within fp32 rounding",
        },
    },
    {
        "index": 2,
        "name": "scale",
        "implement": "Scale scores by 1/sqrt(D) via eltwise mul (scalar broadcast from cb_scale_factor).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_scale",
            "checkpoint": "cb_scores tile 0, first 4×4 — scaled by scale factor",
            "pass_criteria": "Score values = Q@K^T * scale, matching reference",
        },
    },
    {
        "index": 3,
        "name": "mask",
        "implement": "Mask add (if attn_mask) via eltwise add, or passthrough copy if no mask. Pinned shape uses passthrough.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_mask",
            "checkpoint": "cb_scores_masked tile 0, first 4×4 — mask applied or passthrough",
            "pass_criteria": "Values match reference (passthrough when no mask)",
        },
    },
    {
        "index": 4,
        "name": "rowmax",
        "implement": "Row-max reduce via reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop> — does NOT pop scores for reuse in phase 8.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rowmax",
            "checkpoint": "cb_max_new tile 0, first 4×4 — per-row max of first score rows",
            "pass_criteria": "Max values match reference row-max of scores",
        },
    },
    {
        "index": 5,
        "name": "alpha",
        "implement": "Compute alpha = exp(m_old - m_new) via eltwise_chain (BinaryFpu Sub + Exp + PackTile).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_alpha",
            "checkpoint": "cb_alpha tile 0, first 4×4 — exp(m_old - m_new)",
            "pass_criteria": "Alpha values match reference exp(m_old - m_new)",
        },
    },
    {
        "index": 6,
        "name": "rescale_o",
        "implement": "Rescale O by alpha via eltwise mul (Col broadcast). O_i = alpha * O_i.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rescale_o",
            "checkpoint": "cb_o tile 0, first 4×4 — O rescaled by alpha",
            "pass_criteria": "O values = alpha * O_old, matching reference (O_old=0 on first KV-block so O stays 0)",
        },
    },
    {
        "index": 7,
        "name": "rescale_l",
        "implement": "Rescale l by alpha via eltwise mul (Col broadcast). l_i = alpha * l_i.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rescale_l",
            "checkpoint": "cb_sum_old tile 0, first 4×4 — l rescaled by alpha",
            "pass_criteria": "l values = alpha * l_old, matching reference (l_old=0 so l stays 0 on first KV-block)",
        },
    },
    {
        "index": 8,
        "name": "subtract_max",
        "implement": "Subtract m_new from scores via eltwise sub (Col broadcast). Reads scores with Streaming (wait+pop after WaitUpfrontNoPop in phase 4).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_subtract_max",
            "checkpoint": "cb_scores_masked tile 0, first 4×4 — scores minus m_new",
            "pass_criteria": "Values = scores - m_new, matching reference",
        },
    },
    {
        "index": 9,
        "name": "exp",
        "implement": "Element-wise exp of rescaled scores via unary<Exp>.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_exp",
            "checkpoint": "cb_exp_scores tile 0, first 4×4 — exp(scores - m_new)",
            "pass_criteria": "Values = exp(scores - m_new), matching reference",
        },
    },
    {
        "index": 10,
        "name": "rowsum",
        "implement": "Row-sum reduce via reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rowsum",
            "checkpoint": "cb_sum_new tile 0, first 4×4 — per-row sum of exp scores",
            "pass_criteria": "Sum values match reference row-sum of exp scores",
        },
    },
    {
        "index": 11,
        "name": "update_l",
        "implement": "Update l_i += sum_new via eltwise add.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_update_l",
            "checkpoint": "cb_sum_old tile 0, first 4×4 — l_i + l_blk",
            "pass_criteria": "l values = l_old + sum_new, matching reference",
        },
    },
    {
        "index": 12,
        "name": "pv_matmul",
        "implement": "PV matmul via matmul_block (transpose=false, packer_l1_acc=true). Accumulates P @ V into cb_o. Reader streams V-block.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_pv",
            "checkpoint": "cb_o tile 0, first 4×4 — O after P @ V accumulation",
            "pass_criteria": "O values = O_rescaled + P @ V, matching reference",
        },
    },
    {
        "index": 13,
        "name": "update_m",
        "implement": "Update m_i = m_new via eltwise copy.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_update_m",
            "checkpoint": "cb_max_old tile 0, first 4×4 — m_i = m_new",
            "pass_criteria": "m values = m_new, matching reference",
        },
    },
    {
        "index": 14,
        "name": "output",
        "implement": "Final output: normalize O by l_i and write to DRAM via writer kernel. After all KV-blocks for a Q-block, drain cb_o to output buffer.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_output",
            "checkpoint": "output DRAM tile 0, first 4×4 — final O / l_i",
            "pass_criteria": "Output = O / l_i, matching torch.nn.functional.scaled_dot_product_attention within fp32 rounding (max diff < 1e-4)",
        },
    },
]
