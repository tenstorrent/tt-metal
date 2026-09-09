# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.quasar.transformer.scaled_dot_product_attention``  (prefill SDPA).

Model call site (modules/attention/attention_1d.py:566-575, prefill_forward):
    attn_output = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        q_heads_sdpa,          # [B, n_heads,    per_user_seq_len, head_dim]
        k_heads_cache_dtype,   # [B, n_kv_heads, per_user_seq_len, head_dim]
        v_heads_cache_dtype,   # [B, n_kv_heads, per_user_seq_len, head_dim]
        is_causal=True,
        sliding_window_size=cfg.sliding_window,   # None for Llama-3.2-1B
        scale=cfg.scale,                          # head_dim ** -0.5
        compute_kernel_config=cfg.sdpa_prefill_compute_kernel_cfg,
        program_config=cfg.prefill_sdpa_prg_config(per_user_seq_len, None),
    )

GQA: 32 query heads, 8 kv heads (head_dim 64). Reference is torch SDPA with the
kv heads expanded to 32 via repeat_interleave(n_heads // n_kv_heads).
Known-good direct-call pattern mirrored from
tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py:248-318.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# Single-user prefill (batch_size == 1 folds users into the sequence axis upstream).
BATCH = 1


def _torch_sdpa_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # Expand GQA kv heads to match the query head count.
    n_rep = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(n_rep, dim=1)
    v_rep = v.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k_rep.float(), v_rep.float(), is_causal=True, scale=scale
    )


@U.with_default_mesh()
@pytest.mark.parametrize("seq", U.PREFILL_SEQ_LENS, ids=[f"seq{s}" for s in U.PREFILL_SEQ_LENS])
def test_scaled_dot_product_attention(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device
    scale = U.HEAD_DIM**-0.5

    q_torch = U.torch_rand((BATCH, U.N_HEADS, seq, U.HEAD_DIM))
    k_torch = U.torch_rand((BATCH, U.N_KV_HEADS, seq, U.HEAD_DIM))
    v_torch = U.torch_rand((BATCH, U.N_KV_HEADS, seq, U.HEAD_DIM))

    q = U.to_tt(q_torch, mesh)
    k = U.to_tt(k_torch, mesh)
    v = U.to_tt(v_torch, mesh)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    ref = _torch_sdpa_causal(q_torch, k_torch, v_torch, scale)  # [B, n_heads, seq, head_dim]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
