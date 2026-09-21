# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared attention operations for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (not the llama variant).
No Meta-format weight conversion needed. No transformation matrices needed.

Handles:
- Per-head RMSNorm (q_norm, k_norm, v_norm) via reshape trick
- Partial RoPE for global layers (split, rotate, concat)
- K=V tying (fused Q+K+K weight, standard nlp_create_qkv_heads split)
- No bias on any projection
- scaling=1.0 (no 1/sqrt(d_k))
"""

import os

import ttnn

from .weights import AttentionWeights


def prefill_short_lived_memcfg() -> ttnn.MemoryConfig:
    """Choose DRAM or optional L1 storage for short-lived attention activations."""
    if os.environ.get("GEMMA4_PREFILL_L1_ACT", "0").lower() in ("1", "true", "yes"):
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _attn_mm_grid(t):  # DIAG: GEMMA4_ATTN_MM_CFG
    """Full compute grid for an attention projection, or None to keep ttnn's default.

    Same mechanism mmanzoor's MLP change uses: ttnn only builds a good matmul program
    config when it is handed a user grid. The attention projections never got it.

    MEASURED, AND IT IS NOT A WIN AT EVERY WIDTH -- keep it gated off by default.
    Isolated floor captures at ctx 256k (see
    tech_reports/Gemma4PrefillChunkSize/L1Activations/per_op_patched/README.md):

        chunk 8192  -4.1 ms   all four projections get faster (-27% to -37%)
        chunk 2048  +1.5 ms   three are flat; _x2048x5376 alone is +41%

    The sign flip is that one small matmul: at chunk 2048 it is half the MACs of the
    next smallest (2.82 G vs 5.6-6.3 G), and a full 120-core grid then costs more in
    dispatch than it saves on math. Same lesson as the RMSNorm block-sharding result --
    more cores is not faster. A size gate was written to keep the win at 8192 without
    the loss at 2048; it measured as a no-op (+/-0.5 ms) and was reverted rather than
    shipped on ambiguous data.
    """
    if os.environ.get("GEMMA4_ATTN_MM_CFG", "0").lower() not in ("1", "true", "yes"):
        return None
    try:
        g = t.device().compute_with_storage_grid_size()
        return ttnn.CoreGrid(y=g.y, x=g.x)
    except Exception:
        return None


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None, kv_tied: bool = False):
    """Project to QKV, or QK when kv_tied selects the narrow tied weight."""
    w_tensor = weights.wqk if kv_tied else weights.wqkv
    _cg = _attn_mm_grid(hidden_states)  # DIAG
    _kw = {"core_grid": _cg} if _cg is not None else {}
    return ttnn.linear(hidden_states, w_tensor, memory_config=memory_config, **_kw)


def split_qkv_heads_prefill(
    xqkv_fused,
    config,
    is_global: bool,
    tp: int = 1,
    kv_replicated: bool = False,
    kv_tied: bool = False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Split the local projection into Q, K and V head tensors.

    With kv_tied, K and V read the same projection columns but return separate tensors.
    memory_config selects storage for the resulting activations."""
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
        kv_tied=kv_tied,
    )


def apply_per_head_norm(tensor, eps, weight=None, memory_config=None):
    """Normalize each token and head independently along head_dim."""
    orig_shape = tensor.shape
    _, num_heads, seq_len, head_dim = orig_shape
    flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_len, head_dim))

    # Use HiFi4 and fp32 acc for greater accuracy
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tensor.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    normed = ttnn.rms_norm(
        flat,
        weight=weight,
        epsilon=eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.reshape(normed, orig_shape)
