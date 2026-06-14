# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mamba2Layer — TP=4 on QB 4-chip Blackhole.

SSM recurrence (chunked SSD) runs on host (PyTorch reference) for the PCC
bringup test; in_proj and out_proj run on TTNN (replicated across all 4 devices).

The nemotron3_mamba2_decode_owned kernel has prohibitive per-step dispatch
latency for S=32 sequential calls; it is validated separately via
test_mamba2_decode_kernel_pcc.

Weight shapes (bfloat16 unless noted):
  norm.weight     : [2688]   (pre-block RMSNorm)
  in_proj.weight  : [10304, 2688]
  conv1d.weight   : [6144, 1, 4]
  conv1d.bias     : [6144]
  dt_bias         : [64]
  A_log           : [64]  float32 in ckpt
  mixer norm.w    : [4096]  (MambaRMSNormGated)
  D               : [64]  float32 in ckpt
  out_proj.weight : [2688, 4096]
"""

import torch
import torch.nn.functional as F

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _rep

NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE_SIZE = 128
CONV_KERNEL = 4
INTERMEDIATE_SIZE = NUM_HEADS * HEAD_DIM  # 4096
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * SSM_STATE_SIZE  # 6144
CHUNK_SIZE = 128
NORM_EPS = 1e-5


# ---------------------------------------------------------------------------
# Chunked SSD helpers  (exact copies from reference/functional.py)
# ---------------------------------------------------------------------------


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return input_tensor
    if input_tensor.ndim == 4:
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    t = _pad_tensor_by_size(input_tensor, pad_size)
    if t.ndim == 3:
        return t.reshape(t.shape[0], -1, chunk_size, t.shape[2])
    else:
        return t.reshape(t.shape[0], -1, chunk_size, t.shape[2], t.shape[3])


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    cs = input_tensor.size(-1)
    t = input_tensor[..., None].expand(*input_tensor.size(), cs)
    mask_lower = torch.tril(torch.ones(cs, cs, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    t = t.masked_fill(~mask_lower, 0)
    seg = torch.cumsum(t, dim=-2)
    mask_diag = torch.tril(torch.ones(cs, cs, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    seg = seg.masked_fill(~mask_diag, float("-inf"))
    return seg


def _mamba_rms_norm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int = 512,
) -> torch.Tensor:
    """Gate-first, then per-group RMSNorm, then scale (norm_before_gate=False)."""
    input_dtype = x.dtype
    B, S, D = x.shape
    gate = F.silu(z.float())
    xg = x.float() * gate
    xg_grouped = xg.view(B, S, -1, group_size)
    var = xg_grouped.pow(2).mean(-1, keepdim=True)
    xg_normed = (xg_grouped * torch.rsqrt(var + eps)).view(B, S, D)
    return (weight.float() * xg_normed).to(input_dtype)


def _ssm_chunked_ssd(
    x: torch.Tensor,  # [B, S, H*D]  after conv, float input
    B_vec: torch.Tensor,  # [B, S, G*N]
    C_vec: torch.Tensor,  # [B, S, G*N]
    dt: torch.Tensor,  # [B, S, H]
    dt_bias: torch.Tensor,  # [H]
    A_log: torch.Tensor,  # [H]
    D: torch.Tensor,  # [H]
    batch_size: int,
    seq_len: int,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    n_groups: int = N_GROUPS,
    ssm_state_size: int = SSM_STATE_SIZE,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Chunked SSD scan — exact reference algorithm from functional.py."""
    A = -torch.exp(A_log.float())
    dt_f = F.softplus(dt + dt_bias)  # [B, S, H]

    x_f = x.reshape(batch_size, seq_len, num_heads, head_dim).float()
    B_f = B_vec.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()
    C_f = C_vec.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()

    reps = num_heads // n_groups
    B_f = B_f.repeat_interleave(reps, dim=2)
    C_f = C_f.repeat_interleave(reps, dim=2)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    D_residual = D[..., None].float() * _pad_tensor_by_size(x_f, pad_size)

    x_f = x_f * dt_f[..., None]
    A_dt = A.to(x_f.dtype) * dt_f

    x_c = _reshape_into_chunks(x_f, pad_size, chunk_size)
    A_c = _reshape_into_chunks(A_dt, pad_size, chunk_size)
    B_c = _reshape_into_chunks(B_f, pad_size, chunk_size)
    C_c = _reshape_into_chunks(C_f, pad_size, chunk_size)

    A_c = A_c.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_c, dim=-1)
    L = torch.exp(_segment_sum(A_c))

    G_inter = C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]
    G = G_inter.sum(dim=-1)
    M_inter = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_inter.sum(dim=-1)
    Y_diag = (M[..., None] * x_c[:, :, None]).sum(dim=3)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B_c * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * x_c[..., None]).sum(dim=2)

    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    A_cumsum_last = A_cumsum[:, :, :, -1]
    decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum_last, (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, _ = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C_c[..., None, :] * states[:, :, None, ...]
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_perm[..., None]

    y = Y_diag + Y_off
    y = y.reshape(batch_size, -1, num_heads, head_dim)
    y = y + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    return y.reshape(batch_size, seq_len, -1)  # [B, S, 4096]


def mamba2_layer_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688]
    norm_weight: torch.Tensor,  # [2688]
    in_proj_weight: torch.Tensor,  # [10304, 2688]
    conv1d_weight: torch.Tensor,  # [6144, 1, 4]
    conv1d_bias: torch.Tensor,  # [6144]
    dt_bias: torch.Tensor,  # [64]  bf16
    A_log: torch.Tensor,  # [64]  fp32
    norm_mixer_weight: torch.Tensor,  # [4096]  bf16
    D: torch.Tensor,  # [64]  fp32
    out_proj_weight: torch.Tensor,  # [2688, 4096]
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """Returns [B, S, 2688] bfloat16 (CPU)."""
    residual = hidden_states
    B, S, H = hidden_states.shape

    def to_dev(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        t_cast = t.bfloat16() if dtype == ttnn.bfloat16 else t.float()
        return _rep(t_cast, mesh_device, layout=layout, dtype=dtype)

    def to_host(t):
        return _host_rep(t, mesh_device, B).bfloat16()

    # 1. Pre-block RMSNorm (TTNN, replicated)
    h_tt = to_dev(hidden_states)
    w_tt = _rep(norm_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)
    normed = to_host(normed_tt)  # [B, S, 2688]

    # 2. in_proj (TTNN, replicated): [B, S, 2688] → [B, S, 10304]
    ip_tt = to_dev(in_proj_weight)
    n_tt = to_dev(normed)
    proj_tt = ttnn.linear(n_tt, ip_tt, transpose_b=True)
    projected = to_host(proj_tt)  # [B, S, 10304]

    gate = projected[..., :INTERMEDIATE_SIZE]
    hBC = projected[..., INTERMEDIATE_SIZE : INTERMEDIATE_SIZE + CONV_DIM]
    dt = projected[..., INTERMEDIATE_SIZE + CONV_DIM :]

    # 3. Causal depthwise conv1d (host)
    hBC_t = hBC.float().transpose(1, 2)
    hBC_c = F.conv1d(hBC_t, conv1d_weight.float(), bias=conv1d_bias.float(), padding=CONV_KERNEL - 1, groups=CONV_DIM)[
        ..., :S
    ]
    hBC_s = F.silu(hBC_c).transpose(1, 2).bfloat16()

    x = hBC_s[..., :INTERMEDIATE_SIZE]
    B_vec = hBC_s[..., INTERMEDIATE_SIZE : INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE]
    C_vec = hBC_s[..., INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE :]

    # 4. SSM recurrence (chunked SSD on host — exact reference algorithm)
    y = _ssm_chunked_ssd(x, B_vec, C_vec, dt, dt_bias, A_log, D, B, S)  # [B, S, 4096]

    # 5. MambaRMSNormGated: gate-first, per-group RMSNorm, scale (host)
    group_size = INTERMEDIATE_SIZE // N_GROUPS  # 512
    scan_output = _mamba_rms_norm_gated(y, gate, norm_mixer_weight, eps=norm_eps, group_size=group_size)

    # 6. out_proj (TTNN, replicated)
    op_tt = to_dev(out_proj_weight)
    ys_tt = to_dev(scan_output)
    out_tt = ttnn.linear(ys_tt, op_tt, transpose_b=True)
    out = to_host(out_tt)

    # 7. Residual
    return (residual + out).bfloat16()
