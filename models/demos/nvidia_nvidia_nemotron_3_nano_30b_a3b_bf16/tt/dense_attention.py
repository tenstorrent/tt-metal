# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DenseAttention block — bringup on QB (device 0).

GQA (32 Q / 2 KV heads, head_dim=128).  No RoPE (HF has it
commented out with a TODO for this model variant).

Full block: pre-RMSNorm + QKV projections + GQA SDPA + O proj + residual.
Runs on device 0 of the QB mesh for initial bringup correctness.
TP parallelism is added in the optimization phase.
"""

import torch
import torch.nn.functional as F

import ttnn
from ttnn import MeshDevice

NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 2688
NORM_EPS = 1e-5

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def _to_dev(t, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.bfloat16(), dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=_R(mesh_device))


def _to_host(t, mesh_device):
    return ttnn.to_torch(t, mesh_composer=_C(mesh_device, dim=0)).bfloat16()


def dense_attention_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU
    norm_weight: torch.Tensor,  # [2688] bf16 CPU
    wq: torch.Tensor,  # [4096, 2688] bf16 CPU
    wk: torch.Tensor,  # [256,  2688] bf16 CPU
    wv: torch.Tensor,  # [256,  2688] bf16 CPU
    wo: torch.Tensor,  # [2688, 4096] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """DenseAttention with pre-RMSNorm and residual on device 0.

    Returns [B, S, 2688] bfloat16 (CPU).
    """
    residual = hidden_states
    B, S, _ = hidden_states.shape

    # 1. Pre-norm
    h_tt = _to_dev(hidden_states, mesh_device)
    w_tt = ttnn.from_torch(
        norm_weight.bfloat16().unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)

    # 2. QKV projections on device
    wq_tt = _to_dev(wq, mesh_device)
    wk_tt = _to_dev(wk, mesh_device)
    wv_tt = _to_dev(wv, mesh_device)
    q_tt = ttnn.linear(normed_tt, wq_tt, transpose_b=True)  # [B, S, 4096]
    k_tt = ttnn.linear(normed_tt, wk_tt, transpose_b=True)  # [B, S, 256]
    v_tt = ttnn.linear(normed_tt, wv_tt, transpose_b=True)  # [B, S, 256]

    q = _to_host(q_tt, mesh_device).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B,32,S,128]
    k = _to_host(k_tt, mesh_device).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [B,2,S,128]
    v = _to_host(v_tt, mesh_device).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [B,2,S,128]

    # 3. GQA expand KV heads and SDPA (on host for bringup)
    n_rep = NUM_HEADS // NUM_KV_HEADS
    k = k.unsqueeze(2).expand(B, NUM_KV_HEADS, n_rep, S, HEAD_DIM).reshape(B, NUM_HEADS, S, HEAD_DIM)
    v = v.unsqueeze(2).expand(B, NUM_KV_HEADS, n_rep, S, HEAD_DIM).reshape(B, NUM_HEADS, S, HEAD_DIM)

    is_causal = S > 1
    attn_out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=is_causal).to(
        torch.bfloat16
    )  # [B, 32, S, 128]

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, NUM_HEADS * HEAD_DIM)  # [B,S,4096]

    # 4. Output projection on device
    wo_tt = _to_dev(wo, mesh_device)
    attn_tt = _to_dev(attn_out, mesh_device)
    out_tt = ttnn.linear(attn_tt, wo_tt, transpose_b=True)
    out = _to_host(out_tt, mesh_device)

    return (residual + out).bfloat16()
