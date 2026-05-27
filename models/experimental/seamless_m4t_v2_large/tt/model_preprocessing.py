# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import ttnn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, make_parameter_dict

from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import embed_scale_for_config
from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    create_dram_sharded_mem_config,
    dram_matmul_shard_cores,
)


def _replicate_mapper(device: ttnn.Device):
    """Return ``ReplicateTensorToMesh`` for multi-device mesh, else ``None``."""
    try:
        if hasattr(device, "get_num_devices") and int(device.get_num_devices()) > 1:
            return ttnn.ReplicateTensorToMesh(device)
    except Exception:
        pass
    return None


def _resolve_tp(device: ttnn.Device, tp: Optional[int]) -> int:
    """Resolve TP degree from explicit arg or device mesh size.

    ``tp=None`` means "auto": use ``device.get_num_devices()`` when available.
    """
    if tp is not None:
        if tp < 1:
            raise ValueError(f"tp must be >= 1, got {tp}")
        return tp
    try:
        if hasattr(device, "get_num_devices"):
            return max(1, int(device.get_num_devices()))
    except Exception:
        pass
    return 1


def _conv1d_weight(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> ttnn.Tensor:
    """Host ROW_MAJOR PyTorch-shaped weights (``[out, in/groups, K]``).

    ``ttnn.conv1d`` / conv2d expect either host tensors (prepared + uploaded per call) or
    device tensors that already pass ``is_valid_device_conv_weights`` (TILE, padded layout).
    Uploading raw ROW_MAJOR weights to device triggers a host round-trip and warnings in
    ``conv2d.cpp``; keeping weights on host avoids that.
    """
    _ = device  # kept for call-site symmetry with other preprocess helpers
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv1d_bias(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    _ = device
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_transpose1d_weight(conv: torch.nn.ConvTranspose1d, *, device: ttnn.Device) -> ttnn.Tensor:
    """PyTorch ``[in_c, out_c, K]`` -> TTNN conv_transpose2d-style ``[in_c, out_c, K, 1]``."""
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    w2 = w.unsqueeze(-1).contiguous()
    return ttnn.from_torch(
        w2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_transpose1d_bias_host(conv: torch.nn.ConvTranspose1d) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _embedding_weight(emb: torch.nn.Embedding, *, device: ttnn.Device) -> ttnn.Tensor:
    w = emb.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device),
    )


def _vocoder_embedding_weight_row_major(emb: torch.nn.Embedding, *, device: ttnn.Device) -> ttnn.Tensor:
    """ROW_MAJOR embedding table for [`TTSeamlessM4Tv2CodeHifiGan`] (matches text decoder / T2U decoder).

    ``ttnn.embedding`` still returns TILE_LAYOUT activations when ``layout=ttnn.TILE_LAYOUT`` is passed;
    TILE-stored tables can add a trailing untilize on lookup. Call sites unchanged in ``tt_code_hifigan``.
    """
    w = emb.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device),
    )


def _ln_to_device(param: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    x = param.detach().reshape(1, 1, -1).contiguous()
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device),
    )


def _linear_pair(
    linear: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    bias_dtype: Optional[ttnn.DataType] = None,
) -> dict:
    b_dtype = bias_dtype if bias_dtype is not None else weight_dtype
    mm = _replicate_mapper(device)
    # Weight: [out, in] -> transpose -> [in, out] padded to TILE
    w_torch = linear.weight.detach().T.contiguous().to(torch.bfloat16)
    b_torch = linear.bias.detach().to(torch.bfloat16).reshape(1, 1, 1, -1).contiguous()
    w_tt = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    b_tt = ttnn.from_torch(
        b_torch,
        dtype=b_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    return {"weight": w_tt, "bias": b_tt}


def _pointwise_conv1d_linear_pair(
    conv: torch.nn.Conv1d,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """1×1 Conv1d weights ``[out, in, 1]`` as TILE linear ``[in, out]`` for tuned matmul."""
    if int(conv.kernel_size[0]) != 1:
        raise ValueError(f"expected kernel_size=1 for pointwise conv, got {conv.kernel_size[0]}")
    mm = _replicate_mapper(device)
    w_torch = conv.weight.detach().squeeze(-1).T.contiguous().to(torch.bfloat16)  # [in, out]
    w_tt = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    out = {"weight": w_tt}
    if conv.bias is not None:
        b_torch = conv.bias.detach().to(torch.bfloat16).reshape(1, 1, 1, -1).contiguous()
        b_tt = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        out["bias"] = b_tt
    return out


def _linear_bias_dram_sharded(
    bias: torch.Tensor,
    *,
    device: ttnn.Device,
    k: int,
    n: int,
) -> ttnn.Tensor:
    """DRAM width-sharded bias for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.

    Bias is always ``[1, 1, TILE, N]`` (row-broadcast over M). Activation row count is
    controlled separately via ``matmul_token_rows`` on the encoder, not baked into bias.
    """
    _, padded_n = create_dram_sharded_mem_config(device, k, n)
    dram_cores = dram_matmul_shard_cores(device, k, n)
    bias_4d = bias.detach().reshape(1, 1, 1, int(bias.numel())).to(torch.bfloat16)
    bias_4d = torch.nn.functional.pad(bias_4d, (0, padded_n - int(bias.numel())))
    bias_4d = bias_4d.expand(1, 1, TILE, padded_n).contiguous()
    dram_grid = device.dram_grid_size()
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, dram_grid.y - 1))}
    )
    bias_shard_shape = [TILE, padded_n // dram_cores]
    bias_shard_spec = ttnn.ShardSpec(shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    bias_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec)
    return ttnn.from_torch(
        bias_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bias_mem_config,
    )


def _linear_pair_dram_sharded(
    linear: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    w_torch = linear.weight.detach().T.contiguous()
    k, n = int(w_torch.shape[0]), int(w_torch.shape[1])
    mem_config, padded_n = create_dram_sharded_mem_config(device, k, n)
    if padded_n > n:
        w_torch = torch.nn.functional.pad(w_torch, (0, padded_n - n))
    weight = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    bias = _linear_bias_dram_sharded(linear.bias.detach(), device=device, k=k, n=n)
    return {"weight": weight, "bias": bias}


def _fused_linear_weight_dram_sharded(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType,
) -> dict:
    w_torch = weight.T.contiguous()
    k, n = int(w_torch.shape[0]), int(w_torch.shape[1])
    mem_config, padded_n = create_dram_sharded_mem_config(device, k, n)
    if padded_n > n:
        w_torch = torch.nn.functional.pad(w_torch, (0, padded_n - n))
    w = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    b = _linear_bias_dram_sharded(bias, device=device, k=k, n=n)
    return {"weight": w, "bias": b}


def _fused_qkv_pair(
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    q_scale: float = 1.0,
) -> dict:
    """Concatenate Q/K/V projection weights into a single fused QKV linear.

    Produced tensor pairs feed ``ttnn.linear`` followed by
    ``ttnn.experimental.nlp_create_qkv_heads`` (fused QKV head path).
    Concatenation is along the output dimension so the fused matmul
    output is laid out as ``[..., 3 * hidden]`` (Q | K | V).

    Stage 8: when ``q_scale != 1.0``, the Q rows of the concatenated weight
    and bias are pre-multiplied by ``q_scale`` (typically ``1 / sqrt(head_dim)``).
    This folds the attention scale factor into the weights at preprocessing time,
    eliminating a runtime ``multiply`` per conformer self-attention layer.
    """
    q_w = q_proj.weight.detach()
    q_b = q_proj.bias.detach()
    if q_scale != 1.0:
        q_w = (q_w * q_scale).to(q_w.dtype)
        q_b = (q_b * q_scale).to(q_b.dtype)
    qkv_weight = torch.cat([q_w, k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    qkv_bias = torch.cat([q_b, k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    w = preprocess_linear_weight(qkv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


def _fused_qkv_pair_dram_sharded(
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    q_scale: float = 1.0,
) -> dict:
    q_w = q_proj.weight.detach()
    q_b = q_proj.bias.detach()
    if q_scale != 1.0:
        q_w = (q_w * q_scale).to(q_w.dtype)
        q_b = (q_b * q_scale).to(q_b.dtype)
    qkv_weight = torch.cat([q_w, k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    qkv_bias = torch.cat([q_b, k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    return _fused_linear_weight_dram_sharded(qkv_weight, qkv_bias, device=device, weight_dtype=weight_dtype)


def _fused_kv_pair(
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Concatenate K/V projection weights for one matmul over shared activations (cross-attn).

    Output layout is ``[..., 2 * hidden]`` (K | V) on the last dim; the decoder splits before
    ``_heads``.
    """
    kv_weight = torch.cat([k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
    kv_bias = torch.cat([k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
    w = preprocess_linear_weight(kv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(kv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


# ---------------------------------------------------------------------------
# Tensor-parallel (TP) weight-sharding helpers
# ---------------------------------------------------------------------------
# These helpers distribute weight matrices across ``tp`` devices using
# ``ShardTensorToMesh(device, dim=0)``.  Each device receives a different
# row-slice so all devices participate in the matrix multiply; a subsequent
# ``all_reduce_sum_replicate`` in the runtime accumulates the partial results.
#
# Convention used throughout (matches Megatron-LM column/row-parallel):
#   column-parallel: output dim split → each device: [in, out//tp]
#   row-parallel:    input dim split  → each device: [in//tp, out]
#
# For TP == 1, these helpers fall back to the existing single-device helpers.
# ---------------------------------------------------------------------------


def _tp_col_parallel_pair(
    weight_torch: torch.Tensor,
    bias_torch: torch.Tensor,
    *,
    device: ttnn.Device,
    tp: int,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Column-parallel linear: split output dim by ``tp``.

    ``weight_torch`` is the PyTorch weight ``[out, in]``.
    Each device receives ``[in, out//tp]`` (transposed for ``ttnn.linear``).
    Bias is split identically so each device adds its local share.
    """
    if tp == 1:
        w = preprocess_linear_weight(weight_torch, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        b = preprocess_linear_bias(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return {"weight": ttnn.to_device(w, device), "bias": ttnn.to_device(b, device)}

    out_dim, in_dim = weight_torch.shape
    local_out = out_dim // tp
    mapper = ttnn.ShardTensorToMesh(device, dim=0)

    weight_slices = []
    bias_slices = []
    for rank in range(tp):
        start, end = rank * local_out, (rank + 1) * local_out
        w_slice = weight_torch[start:end, :].T.contiguous().to(torch.bfloat16)  # [in, local_out]
        weight_slices.append(w_slice)
        bias_slices.append(bias_torch[start:end].to(torch.bfloat16))

    stacked_w = torch.stack(weight_slices, dim=0)  # [tp, in, local_out]
    stacked_b = torch.stack(bias_slices, dim=0)  # [tp, local_out]

    w_tt = ttnn.from_torch(stacked_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    b_4d = stacked_b.reshape(tp, 1, 1, local_out)
    b_tt = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    return {"weight": w_tt, "bias": b_tt}


def _tp_row_parallel_pair(
    weight_torch: torch.Tensor,
    bias_torch: torch.Tensor,
    *,
    device: ttnn.Device,
    tp: int,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Row-parallel linear: split input dim by ``tp``.

    ``weight_torch`` is the PyTorch weight ``[out, in]``.
    Each device receives ``[in//tp, out]`` (transposed for ``ttnn.linear``).
    Bias is divided by ``tp`` so that the ``all_reduce_sum_replicate`` across
    devices reconstructs the full bias (``tp × (bias/tp) = bias``).
    """
    if tp == 1:
        w = preprocess_linear_weight(weight_torch, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        b = preprocess_linear_bias(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return {"weight": ttnn.to_device(w, device), "bias": ttnn.to_device(b, device)}

    out_dim, in_dim = weight_torch.shape
    local_in = in_dim // tp
    mapper = ttnn.ShardTensorToMesh(device, dim=0)

    weight_slices = []
    for rank in range(tp):
        start, end = rank * local_in, (rank + 1) * local_in
        w_slice = weight_torch[:, start:end].T.contiguous().to(torch.bfloat16)  # [local_in, out]
        weight_slices.append(w_slice)

    stacked_w = torch.stack(weight_slices, dim=0)  # [tp, local_in, out]
    w_tt = ttnn.from_torch(stacked_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)

    # Bias divided by tp so the all_reduce sum gives the full bias.
    b_scaled = bias_torch.to(torch.bfloat16) / tp
    b_stacked = b_scaled.unsqueeze(0).expand(tp, -1).contiguous()  # [tp, out]
    b_4d = b_stacked.reshape(tp, 1, 1, out_dim)
    b_tt = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    return {"weight": w_tt, "bias": b_tt}


def _tp_fused_qkv_col_parallel(
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    tp: int,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    q_scale: float = 1.0,
) -> dict:
    """Column-parallel fused QKV: each device gets Q_local|K_local|V_local.

    For tp=1 this is identical to ``_fused_qkv_pair``.  For tp>1 the fused
    output dim ``3H`` is split *per-component* so device i gets:
        Q[i*H//tp : (i+1)*H//tp] | K[i*H//tp : (i+1)*H//tp] | V[i*H//tp : (i+1)*H//tp]
    Each device has a complete (Q, K, V) set for its ``num_heads//tp`` heads.
    """
    q_w = q_proj.weight.detach()
    q_b = q_proj.bias.detach()
    if q_scale != 1.0:
        q_w = (q_w * q_scale).to(q_w.dtype)
        q_b = (q_b * q_scale).to(q_b.dtype)

    if tp == 1:
        qkv_weight = torch.cat([q_w, k_proj.weight.detach(), v_proj.weight.detach()], dim=0).contiguous()
        qkv_bias = torch.cat([q_b, k_proj.bias.detach(), v_proj.bias.detach()], dim=0).contiguous()
        w = preprocess_linear_weight(qkv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        b = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return {"weight": ttnn.to_device(w, device), "bias": ttnn.to_device(b, device)}

    k_w = k_proj.weight.detach()
    v_w = v_proj.weight.detach()
    k_b = k_proj.bias.detach()
    v_b = v_proj.bias.detach()
    H_out = int(q_w.shape[0])  # output per-component = hidden_size
    local_out = H_out // tp
    mapper = ttnn.ShardTensorToMesh(device, dim=0)

    weight_slices = []
    bias_slices = []
    for rank in range(tp):
        start, end = rank * local_out, (rank + 1) * local_out
        # Interleaved: Q_rank | K_rank | V_rank
        qkv_slice = torch.cat([q_w[start:end, :], k_w[start:end, :], v_w[start:end, :]], dim=0)  # [3*local_out, in]
        w_t = qkv_slice.T.contiguous().to(torch.bfloat16)  # [in, 3*local_out]
        weight_slices.append(w_t)
        bias_slice = torch.cat([q_b[start:end], k_b[start:end], v_b[start:end]])
        bias_slices.append(bias_slice.to(torch.bfloat16))

    in_dim = int(q_w.shape[1])
    stacked_w = torch.stack(weight_slices, dim=0)  # [tp, in, 3*local_out]
    stacked_b = torch.stack(bias_slices, dim=0)  # [tp, 3*local_out]

    w_tt = ttnn.from_torch(stacked_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    b_4d = stacked_b.reshape(tp, 1, 1, 3 * local_out)
    b_tt = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    return {"weight": w_tt, "bias": b_tt}


def _tp_fused_kv_col_parallel(
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    *,
    device: ttnn.Device,
    tp: int,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Column-parallel fused KV for cross-attention: device i gets K_local|V_local."""
    if tp == 1:
        return _fused_kv_pair(k_proj, v_proj, device=device, weight_dtype=weight_dtype)

    k_w = k_proj.weight.detach()
    v_w = v_proj.weight.detach()
    k_b = k_proj.bias.detach()
    v_b = v_proj.bias.detach()
    H_out = int(k_w.shape[0])
    local_out = H_out // tp
    mapper = ttnn.ShardTensorToMesh(device, dim=0)

    weight_slices = []
    bias_slices = []
    for rank in range(tp):
        start, end = rank * local_out, (rank + 1) * local_out
        kv_slice = torch.cat([k_w[start:end, :], v_w[start:end, :]], dim=0)  # [2*local_out, in]
        weight_slices.append(kv_slice.T.contiguous().to(torch.bfloat16))
        bias_slices.append(torch.cat([k_b[start:end], v_b[start:end]]).to(torch.bfloat16))

    stacked_w = torch.stack(weight_slices, dim=0)
    stacked_b = torch.stack(bias_slices, dim=0)
    w_tt = ttnn.from_torch(stacked_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    b_4d = stacked_b.reshape(tp, 1, 1, 2 * local_out)
    b_tt = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
    return {"weight": w_tt, "bias": b_tt}


def _tp_linear_pair(
    linear: torch.nn.Linear,
    *,
    device: ttnn.Device,
    tp: int,
    parallel: str,  # "col" or "row"
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    """Dispatch to column- or row-parallel TP linear based on ``parallel``."""
    if parallel == "col":
        return _tp_col_parallel_pair(
            linear.weight.detach(), linear.bias.detach(), device=device, tp=tp, weight_dtype=weight_dtype
        )
    if parallel == "row":
        return _tp_row_parallel_pair(
            linear.weight.detach(), linear.bias.detach(), device=device, tp=tp, weight_dtype=weight_dtype
        )
    raise ValueError(f"_tp_linear_pair: parallel must be 'col' or 'row', got {parallel!r}")


def create_text_decoder_parameters(decoder, *, device: ttnn.Device, tp: int = 1) -> dict:
    """
    Convert [`SeamlessM4Tv2Decoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding``]).
    When ``tp > 1``, attention and FFN weights are sharded across devices using
    ``ShardTensorToMesh``; embeddings and layer-norms stay replicated.
    """
    cfg = decoder.config
    scale = embed_scale_for_config(cfg)
    mm = _replicate_mapper(device)

    # ROW_MAJOR embedding tables (matches text encoder / T2U decoder).
    # ``ttnn.embedding`` emits TILE_LAYOUT activations regardless; TILE-stored weights
    # can force a trailing ``UntilizeWithUnpaddingDeviceOperation`` per table lookup.
    scaled_emb = (decoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )

    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )

    layers = []
    for layer in decoder.layers:
        if tp > 1:
            # TP>1: column-parallel QKV, row-parallel O_proj/fc2, column-parallel fc1.
            # Cross-attn: column-parallel q/kv, row-parallel out_proj.
            self_attn_qkv = _tp_fused_qkv_col_parallel(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                device=device,
                tp=tp,
                weight_dtype=ttnn.bfloat8_b,
            )
            self_attn_out = _tp_linear_pair(
                layer.self_attn.out_proj, device=device, tp=tp, parallel="row", weight_dtype=ttnn.bfloat8_b
            )
            cross_q = _tp_linear_pair(
                layer.cross_attention.q_proj, device=device, tp=tp, parallel="col", weight_dtype=ttnn.bfloat8_b
            )
            cross_kv = _tp_fused_kv_col_parallel(
                layer.cross_attention.k_proj,
                layer.cross_attention.v_proj,
                device=device,
                tp=tp,
                weight_dtype=ttnn.bfloat8_b,
            )
            cross_out = _tp_linear_pair(
                layer.cross_attention.out_proj, device=device, tp=tp, parallel="row", weight_dtype=ttnn.bfloat8_b
            )
            fc1 = _tp_linear_pair(layer.ffn.fc1, device=device, tp=tp, parallel="col", weight_dtype=ttnn.bfloat8_b)
            fc2 = _tp_linear_pair(layer.ffn.fc2, device=device, tp=tp, parallel="row", weight_dtype=ttnn.bfloat8_b)
        else:
            self_attn_qkv = _fused_qkv_pair(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                device=device,
                weight_dtype=ttnn.bfloat8_b,
            )
            self_attn_out = _linear_pair(layer.self_attn.out_proj, device=device, weight_dtype=ttnn.bfloat8_b)
            cross_q = _linear_pair(layer.cross_attention.q_proj, device=device, weight_dtype=ttnn.bfloat8_b)
            cross_kv = _fused_kv_pair(
                layer.cross_attention.k_proj,
                layer.cross_attention.v_proj,
                device=device,
                weight_dtype=ttnn.bfloat8_b,
            )
            cross_out = _linear_pair(layer.cross_attention.out_proj, device=device, weight_dtype=ttnn.bfloat8_b)
            fc1 = _linear_pair(layer.ffn.fc1, device=device, weight_dtype=ttnn.bfloat8_b)
            fc2 = _linear_pair(layer.ffn.fc2, device=device, weight_dtype=ttnn.bfloat8_b)

        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            # Fused self-attn Q|K|V; cross-attn K|V fused (see ``cross_attention``).
            # Attention linear weights in bfloat8_b (bandwidth; biases stay bf16) — encoder pattern.
            "self_attn": {
                # Prefill: interleaved L1 matmul + ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (text encoder).
                # Decode: separate ``qkv_decode`` weights for KV-cache PCC.
                "qkv": self_attn_qkv,
                "qkv_decode": self_attn_qkv,  # same weights; decode path reuses prefill weights
                "out_proj": self_attn_out,
            },
            "cross_attention_layer_norm": {
                "weight": _ln_to_device(layer.cross_attention_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.cross_attention_layer_norm.bias, device=device),
            },
            # Fused K|V over encoder hidden states (one matmul vs two; Q stays separate).
            "cross_attention": {
                "q_proj": cross_q,
                "kv": cross_kv,
                "out_proj": cross_out,
            },
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": fc1,
                "fc2": fc2,
            },
        }
        layers.append(make_parameter_dict(layer_dict))

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _m4t_encoder_self_attn_ffn_layers(
    encoder,
    *,
    device: ttnn.Device,
    ffn_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    attn_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    fuse_qkv: bool = False,
    dram_width_sharded_weights: bool = False,
    prefill_token_rows: int = 32,
    tp: int = 1,
) -> list:
    """Layer parameter dicts shared by text encoder and text-to-unit encoder stacks.

    ``ffn_weight_dtype`` controls the storage dtype of ``fc1.weight`` /
    ``fc2.weight``. ``attn_weight_dtype`` controls the storage dtype of
    ``q_proj``/``k_proj``/``v_proj``/``out_proj`` weights. Pass
    ``ttnn.bfloat8_b`` for memory-bound matmuls (bandwidth optimization);
    biases and LayerNorm parameters always stay at ``bfloat16``.

    When ``fuse_qkv=True`` the per-layer ``self_attn`` dict exposes a single
    ``qkv`` entry with concatenated Q|K|V weights/biases (consumed by
    ``ttnn.experimental.nlp_create_qkv_heads`` in the encoder forward) instead
    of separate ``q_proj``/``k_proj``/``v_proj`` entries. ``out_proj`` is
    always exposed as a separate linear pair.

    ``dram_width_sharded_weights=True`` stores linear weights as DRAM width-sharded
    BFP8 tiles with DRAM-sharded bias for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.
    ``prefill_token_rows`` is the M dimension baked into DRAM-sharded bias tiles.

    When ``tp > 1``, TP sharding overrides DRAM-sharded weights: QKV is column-parallel,
    out_proj/fc2 are row-parallel, fc1 is column-parallel (Megatron-LM convention).
    """
    if tp > 1:
        # TP sharding is incompatible with DRAM-sharded per-device weights.
        # Use regular TILE layout with ShardTensorToMesh.
        layers = []
        for layer in encoder.layers:
            if fuse_qkv:
                self_attn = {
                    "qkv": _tp_fused_qkv_col_parallel(
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                        device=device,
                        tp=tp,
                        weight_dtype=attn_weight_dtype,
                    ),
                    "out_proj": _tp_linear_pair(
                        layer.self_attn.out_proj,
                        device=device,
                        tp=tp,
                        parallel="row",
                        weight_dtype=attn_weight_dtype,
                    ),
                }
            else:
                self_attn = {
                    "q_proj": _tp_linear_pair(
                        layer.self_attn.q_proj,
                        device=device,
                        tp=tp,
                        parallel="col",
                        weight_dtype=attn_weight_dtype,
                    ),
                    "k_proj": _tp_linear_pair(
                        layer.self_attn.k_proj,
                        device=device,
                        tp=tp,
                        parallel="col",
                        weight_dtype=attn_weight_dtype,
                    ),
                    "v_proj": _tp_linear_pair(
                        layer.self_attn.v_proj,
                        device=device,
                        tp=tp,
                        parallel="col",
                        weight_dtype=attn_weight_dtype,
                    ),
                    "out_proj": _tp_linear_pair(
                        layer.self_attn.out_proj,
                        device=device,
                        tp=tp,
                        parallel="row",
                        weight_dtype=attn_weight_dtype,
                    ),
                }
            layer_dict = {
                "self_attn_layer_norm": {
                    "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                    "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
                },
                "self_attn": self_attn,
                "ffn_layer_norm": {
                    "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                    "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
                },
                "ffn": {
                    "fc1": _tp_linear_pair(
                        layer.ffn.fc1, device=device, tp=tp, parallel="col", weight_dtype=ffn_weight_dtype
                    ),
                    "fc2": _tp_linear_pair(
                        layer.ffn.fc2, device=device, tp=tp, parallel="row", weight_dtype=ffn_weight_dtype
                    ),
                },
            }
            layers.append(make_parameter_dict(layer_dict))
        return layers

    # TP=1: use existing single-device paths (DRAM-sharded or regular).
    linear_pair = _linear_pair_dram_sharded if dram_width_sharded_weights else _linear_pair
    fused_qkv_pair = _fused_qkv_pair_dram_sharded if dram_width_sharded_weights else _fused_qkv_pair
    linear_kwargs: dict = {}

    layers = []
    for layer in encoder.layers:
        if fuse_qkv:
            self_attn = {
                "qkv": fused_qkv_pair(
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    device=device,
                    weight_dtype=attn_weight_dtype,
                    **linear_kwargs,
                ),
                "out_proj": linear_pair(
                    layer.self_attn.out_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
            }
        else:
            self_attn = {
                "q_proj": linear_pair(
                    layer.self_attn.q_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "k_proj": linear_pair(
                    layer.self_attn.k_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "v_proj": linear_pair(
                    layer.self_attn.v_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
                "out_proj": linear_pair(
                    layer.self_attn.out_proj, device=device, weight_dtype=attn_weight_dtype, **linear_kwargs
                ),
            }

        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": self_attn,
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": linear_pair(layer.ffn.fc1, device=device, weight_dtype=ffn_weight_dtype, **linear_kwargs),
                "fc2": linear_pair(layer.ffn.fc2, device=device, weight_dtype=ffn_weight_dtype, **linear_kwargs),
            },
        }
        layers.append(make_parameter_dict(layer_dict))
    return layers


def create_text_encoder_parameters(
    encoder,
    *,
    device: ttnn.Device,
    prefill_token_rows: int = 32,
    tp: Optional[int] = None,
) -> dict:
    """
    Convert [`SeamlessM4Tv2Encoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding`]).
    When ``tp == 1``: linear weights use L1 width-sharded activations + DRAM width-sharded
    BFP8 weights for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.
    When ``tp > 1``: TP column/row-parallel sharding via ``ShardTensorToMesh``.
    """
    tp = _resolve_tp(device, tp)
    cfg = encoder.config
    scale = embed_scale_for_config(cfg)
    mm = _replicate_mapper(device)

    scaled_emb = (encoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )

    pos_w = encoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )

    layers = _m4t_encoder_self_attn_ffn_layers(
        encoder,
        device=device,
        ffn_weight_dtype=ttnn.bfloat8_b,
        attn_weight_dtype=ttnn.bfloat8_b,
        fuse_qkv=True,
        dram_width_sharded_weights=(tp == 1),
        prefill_token_rows=prefill_token_rows,
        tp=tp,
    )

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _conv_padding_int(conv: torch.nn.Module) -> int:
    p = conv.padding
    if isinstance(p, int):
        return int(p)
    return int(p[0])


def _conv1d_like_padding_int(conv: torch.nn.Module) -> int:
    """
    TTNN paths need an integral symmetric padding. HF may use ``padding='same'`` on
    [`torch.nn.Conv1d`] (string); map that to PyTorch's stride-1 effective padding.
    """
    p = conv.padding
    if isinstance(p, str):
        if p == "same":
            k = int(conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size)
            d = int(conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation)
            s = int(conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride)
            if s != 1:
                raise ValueError("padding='same' with stride != 1 is not supported for TT export")
            return (k - 1) * d // 2
        if p in ("valid", "zeros"):
            return 0
        raise ValueError(f"Unsupported Conv1d padding mode {p!r}")
    if isinstance(p, (tuple, list)):
        return int(p[0])
    return int(p)


def _conformer_feed_forward_params(
    ffn: torch.nn.Module,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    out_scale: float = 1.0,
    dram_width_sharded_weights: bool = False,
    tp: int = 1,
) -> dict:
    """Preprocess conformer FFN weights.

    Stage 13a: when ``out_scale != 1.0`` (e.g. 0.5 for Macaron-style half-step residual),
    the ``output_dense`` weight and bias are pre-multiplied at load time.
    When ``tp > 1``: intermediate_dense is column-parallel, output_dense is row-parallel.
    """
    od_w = ffn.output_dense.weight.detach()
    od_b = ffn.output_dense.bias.detach()
    if out_scale != 1.0:
        od_w = (od_w * out_scale).to(od_w.dtype)
        od_b = (od_b * out_scale).to(od_b.dtype)

    if tp > 1:
        return make_parameter_dict(
            {
                "intermediate_dense": _tp_col_parallel_pair(
                    ffn.intermediate_dense.weight.detach(),
                    ffn.intermediate_dense.bias.detach(),
                    device=device,
                    tp=tp,
                    weight_dtype=weight_dtype,
                ),
                "output_dense": _tp_row_parallel_pair(
                    od_w,
                    od_b,
                    device=device,
                    tp=tp,
                    weight_dtype=weight_dtype,
                ),
            }
        )

    if dram_width_sharded_weights:
        return make_parameter_dict(
            {
                "intermediate_dense": _linear_pair_dram_sharded(
                    ffn.intermediate_dense, device=device, weight_dtype=weight_dtype
                ),
                "output_dense": _fused_linear_weight_dram_sharded(od_w, od_b, device=device, weight_dtype=weight_dtype),
            }
        )
    mm = _replicate_mapper(device)
    od_w_t = od_w.T.contiguous().to(torch.bfloat16)
    od_b_4d = od_b.to(torch.bfloat16).reshape(1, 1, 1, -1).contiguous()
    od_w_tt = ttnn.from_torch(
        od_w_t,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    od_b_tt = ttnn.from_torch(
        od_b_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    return make_parameter_dict(
        {
            "intermediate_dense": _linear_pair(ffn.intermediate_dense, device=device, weight_dtype=weight_dtype),
            "output_dense": {
                "weight": od_w_tt,
                "bias": od_b_tt,
            },
        }
    )


def _conformer_conv_module_params(
    conv_module: torch.nn.Module,
    *,
    device: ttnn.Device,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict:
    pc1_linear = _pointwise_conv1d_linear_pair(conv_module.pointwise_conv1, device=device, weight_dtype=weight_dtype)
    pc2_linear = _pointwise_conv1d_linear_pair(conv_module.pointwise_conv2, device=device, weight_dtype=weight_dtype)
    return make_parameter_dict(
        {
            "layer_norm": {
                "weight": _ln_to_device(conv_module.layer_norm.weight, device=device),
                "bias": _ln_to_device(conv_module.layer_norm.bias, device=device),
                "eps": float(conv_module.layer_norm.eps),
            },
            "pointwise_conv1": {
                **pc1_linear,
                "out_channels": conv_module.pointwise_conv1.out_channels,
            },
            "depthwise_conv": {
                "weight": _conv1d_weight(conv_module.depthwise_conv, device=device),
                "bias": _conv1d_bias(conv_module.depthwise_conv, device=device),
                "in_channels": conv_module.depthwise_conv.in_channels,
                "out_channels": conv_module.depthwise_conv.out_channels,
                "kernel_size": int(conv_module.depthwise_conv.kernel_size[0]),
                "padding": _conv_padding_int(conv_module.depthwise_conv),
                "stride": int(conv_module.depthwise_conv.stride[0]),
                "groups": conv_module.depthwise_conv.groups,
                "left_pad": int(conv_module.depthwise_conv.kernel_size[0]) - 1,
            },
            "depthwise_layer_norm": {
                "weight": _ln_to_device(conv_module.depthwise_layer_norm.weight, device=device),
                "bias": _ln_to_device(conv_module.depthwise_layer_norm.bias, device=device),
                "eps": float(conv_module.depthwise_layer_norm.eps),
            },
            "pointwise_conv2": {
                **pc2_linear,
                "out_channels": conv_module.pointwise_conv2.out_channels,
            },
        }
    )


def _conformer_self_attn_params(
    attn: torch.nn.Module,
    *,
    device: ttnn.Device,
    with_relative: bool,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    dram_width_sharded_weights: bool = False,
    tp: int = 1,
) -> dict:
    if with_relative:
        q_scale = 1.0 / math.sqrt(int(attn.head_size))
    else:
        q_scale = 1.0

    if tp > 1:
        out = {
            "qkv": _tp_fused_qkv_col_parallel(
                attn.linear_q,
                attn.linear_k,
                attn.linear_v,
                device=device,
                tp=tp,
                weight_dtype=weight_dtype,
                q_scale=q_scale,
            ),
            "linear_out": _tp_linear_pair(
                attn.linear_out,
                device=device,
                tp=tp,
                parallel="row",
                weight_dtype=weight_dtype,
            ),
        }
    else:
        linear_pair = _linear_pair_dram_sharded if dram_width_sharded_weights else _linear_pair
        fused_qkv_pair = _fused_qkv_pair_dram_sharded if dram_width_sharded_weights else _fused_qkv_pair
        out = {
            "qkv": fused_qkv_pair(
                attn.linear_q,
                attn.linear_k,
                attn.linear_v,
                device=device,
                weight_dtype=weight_dtype,
                q_scale=q_scale,
            ),
            "linear_out": linear_pair(attn.linear_out, device=device, weight_dtype=weight_dtype),
        }

    if with_relative and getattr(attn, "distance_embedding", None) is not None:
        out["distance_embedding"] = make_parameter_dict(
            {"weight": _embedding_weight(attn.distance_embedding, device=device)}
        )
        out["left_max_position_embeddings"] = int(attn.left_max_position_embeddings)
        out["right_max_position_embeddings"] = int(attn.right_max_position_embeddings)
    return make_parameter_dict(out)


def _conformer_encoder_layer_params(
    layer: torch.nn.Module,
    *,
    device: ttnn.Device,
    matmul_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    dram_width_sharded_weights: bool = False,
    tp: int = 1,
) -> dict:
    ffn_kwargs = {
        "device": device,
        "weight_dtype": matmul_weight_dtype,
        "dram_width_sharded_weights": dram_width_sharded_weights if tp == 1 else False,
        "tp": tp,
    }
    attn_kwargs = {
        "device": device,
        "with_relative": True,
        "weight_dtype": matmul_weight_dtype,
        "dram_width_sharded_weights": dram_width_sharded_weights if tp == 1 else False,
        "tp": tp,
    }
    return make_parameter_dict(
        {
            "ffn1_layer_norm": {
                "weight": _ln_to_device(layer.ffn1_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn1_layer_norm.bias, device=device),
            },
            "ffn1": _conformer_feed_forward_params(layer.ffn1, out_scale=0.5, **ffn_kwargs),
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": _conformer_self_attn_params(layer.self_attn, **attn_kwargs),
            "conv_module": _conformer_conv_module_params(
                layer.conv_module, device=device, weight_dtype=matmul_weight_dtype
            ),
            "ffn2_layer_norm": {
                "weight": _ln_to_device(layer.ffn2_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn2_layer_norm.bias, device=device),
            },
            "ffn2": _conformer_feed_forward_params(layer.ffn2, out_scale=0.5, **ffn_kwargs),
            "final_layer_norm": {
                "weight": _ln_to_device(layer.final_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.final_layer_norm.bias, device=device),
            },
        }
    )


def _speech_adapter_layer_params(
    layer: torch.nn.Module,
    *,
    device: ttnn.Device,
    matmul_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    dram_width_sharded_weights: bool = False,
    tp: int = 1,
) -> dict:
    ffn_kwargs = {
        "device": device,
        "weight_dtype": matmul_weight_dtype,
        "dram_width_sharded_weights": dram_width_sharded_weights if tp == 1 else False,
        "tp": tp,
    }
    attn_kwargs = {
        "device": device,
        "with_relative": False,
        "weight_dtype": matmul_weight_dtype,
        "dram_width_sharded_weights": dram_width_sharded_weights if tp == 1 else False,
        "tp": tp,
    }
    return make_parameter_dict(
        {
            "kernel_size": int(layer.kernel_size),
            "stride": int(layer.stride),
            "residual_layer_norm": {
                "weight": _ln_to_device(layer.residual_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.residual_layer_norm.bias, device=device),
            },
            "residual_conv": {
                "weight": _conv1d_weight(layer.residual_conv, device=device),
                "bias": _conv1d_bias(layer.residual_conv, device=device),
                "in_channels": layer.residual_conv.in_channels,
                "out_channels": layer.residual_conv.out_channels,
                "kernel_size": int(layer.residual_conv.kernel_size[0]),
                "padding": _conv1d_like_padding_int(layer.residual_conv),
                "stride": int(layer.residual_conv.stride[0]),
                "groups": layer.residual_conv.groups,
            },
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn_conv": {
                "weight": _conv1d_weight(layer.self_attn_conv, device=device),
                "bias": _conv1d_bias(layer.self_attn_conv, device=device),
                "in_channels": layer.self_attn_conv.in_channels,
                "out_channels": layer.self_attn_conv.out_channels,
                "kernel_size": int(layer.self_attn_conv.kernel_size[0]),
                "padding": _conv1d_like_padding_int(layer.self_attn_conv),
                "stride": int(layer.self_attn_conv.stride[0]),
                "groups": layer.self_attn_conv.groups,
            },
            "self_attn": _conformer_self_attn_params(layer.self_attn, **attn_kwargs),
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": _conformer_feed_forward_params(layer.ffn, **ffn_kwargs),
        }
    )


def create_speech_encoder_parameters(
    speech_encoder,
    *,
    device: ttnn.Device,
    dram_width_sharded_weights: bool = False,
    tp: int = 1,
) -> dict:
    """
    Convert [`SeamlessM4Tv2SpeechEncoder`] weights to TTNN tensors for [`TTSeamlessM4Tv2SpeechEncoder`].

    Default: interleaved DRAM ``bfloat8_b`` weights + L1 interleaved activations with 1D multicast
    matmul (best for prefill / seq > 32). Set ``dram_width_sharded_weights=True`` for decode-style
    L1/DRAM width-sharded matmul (M <= 32 only).
    When ``tp > 1``: conformer self-attn and FFN use TP column/row-parallel sharding;
    depthwise conv (groups=hidden_size) remains replicated (per-channel, not GEMM-bound).
    """
    matmul_bf8 = ttnn.bfloat8_b
    use_dram = dram_width_sharded_weights and (tp == 1)
    linear_pair = _linear_pair_dram_sharded if use_dram else _linear_pair
    layer_kwargs = {
        "device": device,
        "matmul_weight_dtype": matmul_bf8,
        "dram_width_sharded_weights": use_dram,
        "tp": tp,
    }
    fp = speech_encoder.feature_projection
    feature_projection = {
        "layer_norm": {
            "weight": _ln_to_device(fp.layer_norm.weight, device=device),
            "bias": _ln_to_device(fp.layer_norm.bias, device=device),
            "eps": float(fp.layer_norm.eps),
        },
        "projection": linear_pair(fp.projection, device=device, weight_dtype=matmul_bf8),
    }
    enc = speech_encoder.encoder
    enc_layers = [_conformer_encoder_layer_params(layer, **layer_kwargs) for layer in enc.layers]
    encoder = {
        "layers": enc_layers,
        "layer_norm": {
            "weight": _ln_to_device(enc.layer_norm.weight, device=device),
            "bias": _ln_to_device(enc.layer_norm.bias, device=device),
        },
    }
    im = speech_encoder.intermediate_ffn
    intermediate_ffn = _conformer_feed_forward_params(
        im,
        device=device,
        weight_dtype=matmul_bf8,
        out_scale=0.5,
        dram_width_sharded_weights=use_dram,
        tp=tp,
    )
    inner_layer_norm = {
        "weight": _ln_to_device(speech_encoder.inner_layer_norm.weight, device=device),
        "bias": _ln_to_device(speech_encoder.inner_layer_norm.bias, device=device),
    }
    out = {
        "feature_projection": make_parameter_dict(feature_projection),
        "encoder": make_parameter_dict(encoder),
        "intermediate_ffn": intermediate_ffn,
        "inner_layer_norm": make_parameter_dict(inner_layer_norm),
    }
    if speech_encoder.adapter is not None:
        adapter_layers = [
            _speech_adapter_layer_params(layer, **layer_kwargs) for layer in speech_encoder.adapter.layers
        ]
        out["adapter"] = make_parameter_dict({"layers": adapter_layers})
    return make_parameter_dict(out)


def create_text_to_unit_parameters(encoder, *, device: ttnn.Device, tp: int = 1) -> dict:
    """
    Convert the encoder submodule of Transformers
    [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] — ``model.encoder``, i.e.
    [`SeamlessM4Tv2Encoder`] with ``is_t2u_encoder=True`` — to TTNN tensors for
    [`TTSeamlessM4Tv2TextToUnitEncoder`].

    Expects ``encoder.layers`` as [`SeamlessM4Tv2EncoderLayer`] (self-attention + FFN only).
    Weights cover the transformer stack only (``inputs_embeds`` path; no token or position
    embeddings on this submodule).

    Store FFN ``fc1``/``fc2`` weights as ``bfloat8_b`` (block-float8).
    Halves DRAM bandwidth on the two largest matmuls per layer; the multiplier still
    runs at bf16 fidelity (``HiFi2`` + ``fp32_dest_acc_en``), so PCC is preserved.

    Extend the same ``bfloat8_b`` storage to the attention projections
    (fused ``qkv`` + ``out_proj`` via ``fuse_qkv=True``).  The underlying matmul
    still accumulates in fp32 with HiFi math fidelity, so the bf8 weight
    quantization is well below PCC headroom and we pick up an extra
    DRAM-bandwidth saving on the fused QKV and out projections per layer.
    When ``tp > 1``: TP column/row-parallel sharding overrides DRAM-sharded.
    """
    layers = _m4t_encoder_self_attn_ffn_layers(
        encoder,
        device=device,
        ffn_weight_dtype=ttnn.bfloat8_b,
        attn_weight_dtype=ttnn.bfloat8_b,
        fuse_qkv=True,
        dram_width_sharded_weights=(tp == 1),
        tp=tp,
    )
    out = {
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _t2u_variance_predictor_parameters(var_pred: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2VariancePredictor`] weights for TTNN text-to-unit duration path."""
    out = {
        "conv1": {
            "weight": _conv1d_weight(var_pred.conv1, device=device),
            "bias": _conv1d_bias(var_pred.conv1, device=device),
        },
        "ln1": {
            "weight": _ln_to_device(var_pred.ln1.weight, device=device),
            "bias": _ln_to_device(var_pred.ln1.bias, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(var_pred.conv2, device=device),
            "bias": _conv1d_bias(var_pred.conv2, device=device),
        },
        "ln2": {
            "weight": _ln_to_device(var_pred.ln2.weight, device=device),
            "bias": _ln_to_device(var_pred.ln2.bias, device=device),
        },
        # fp32 proj: ``log_dur`` rounding boundary must match HF; upload once here so
        # ``TTSeamlessM4Tv2TextToUnitForConditionalGeneration`` does not ``typecast`` at init.
        "proj": _linear_pair(
            var_pred.proj,
            device=device,
            weight_dtype=ttnn.float32,
            bias_dtype=ttnn.float32,
        ),
    }
    return make_parameter_dict(out)


def _t2u_decoder_layer_parameters(layer: torch.nn.Module, *, device: ttnn.Device, tp: int = 1) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoderLayer`] weights.

    When ``tp > 1``: self-attn QKV is column-parallel, out_proj is row-parallel.
    Convolutions (conv1, conv2) remain replicated.
    """
    if tp > 1:
        self_attn = {
            "qkv": _tp_fused_qkv_col_parallel(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                device=device,
                tp=tp,
                weight_dtype=ttnn.bfloat8_b,
            ),
            "out_proj": _tp_linear_pair(
                layer.self_attn.out_proj,
                device=device,
                tp=tp,
                parallel="row",
            ),
        }
    else:
        self_attn = {
            "qkv": _fused_qkv_pair_dram_sharded(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                device=device,
                weight_dtype=ttnn.bfloat8_b,
            ),
            "out_proj": _linear_pair_dram_sharded(layer.self_attn.out_proj, device=device),
        }

    layer_dict = {
        "self_attn": self_attn,
        "self_attn_layer_norm": {
            "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
        },
        "conv1": {
            "weight": _conv1d_weight(layer.conv1, device=device),
            "bias": _conv1d_bias(layer.conv1, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(layer.conv2, device=device),
            "bias": _conv1d_bias(layer.conv2, device=device),
        },
        "conv_layer_norm": {
            "weight": _ln_to_device(layer.conv_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.conv_layer_norm.bias, device=device),
        },
    }
    return make_parameter_dict(layer_dict)


def _t2u_decoder_parameters(decoder: torch.nn.Module, *, device: ttnn.Device, tp: int = 1) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoder`] weights (character + duration + conv decoder stack)."""
    cfg = decoder.config
    scale = embed_scale_for_config(cfg)
    mm = _replicate_mapper(device)
    # Upload embedding tables ROW-MAJOR (matches text encoder recipe).
    # ``ttnn.embedding`` produces a TILE_LAYOUT output regardless of how its weight is
    # stored; uploading the weight in ROW_MAJOR_LAYOUT avoids the trailing
    # ``UntilizeWithUnpaddingDeviceOperation`` that the embedding kernel emits when the
    # weight is already tile-padded.  Numerically identical, ~3 ops cheaper per forward.
    scaled_char = (decoder.embed_char.weight.detach() * scale).contiguous()
    embed_char_weight = ttnn.from_torch(
        scaled_char,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    char_pos_w = decoder.embed_char_positions.weights.detach()
    if char_pos_w.dtype != torch.bfloat16:
        char_pos_w = char_pos_w.to(dtype=torch.bfloat16)
    embed_char_positions_weight = ttnn.from_torch(
        char_pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    pos_emb_alpha_char = ttnn.from_torch(
        decoder.pos_emb_alpha_char.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    pos_emb_alpha = ttnn.from_torch(
        decoder.pos_emb_alpha.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    layers = [_t2u_decoder_layer_parameters(layer, device=device, tp=tp) for layer in decoder.layers]
    out = {
        "embed_char": make_parameter_dict({"weight": embed_char_weight}),
        "embed_char_positions": make_parameter_dict({"weight": embed_char_positions_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "pos_emb_alpha_char": pos_emb_alpha_char,
        "pos_emb_alpha": pos_emb_alpha,
        "duration_predictor": _t2u_variance_predictor_parameters(decoder.duration_predictor, device=device),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def create_text_to_unit_condgen_parameters(
    t2u: torch.nn.Module,
    *,
    device: ttnn.Device,
    tp: int = 1,
) -> dict:
    """
    Full [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] weights for TTNN:
    ``model.encoder``, ``model.decoder``, and ``lm_head``.
    """
    mm = _replicate_mapper(device)
    w_lm_torch = t2u.lm_head.weight.detach().T.contiguous().to(torch.bfloat16)
    w_lm = ttnn.from_torch(
        w_lm_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    out = {
        "encoder": create_text_to_unit_parameters(t2u.model.encoder, device=device, tp=tp),
        "decoder": _t2u_decoder_parameters(t2u.model.decoder, device=device, tp=tp),
        "lm_head": make_parameter_dict({"weight": w_lm}),
    }
    return make_parameter_dict(out)


def create_code_hifigan_parameters(vocoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2CodeHifiGan`] (``model.vocoder``) to TTNN tensors for
    [`TTSeamlessM4Tv2CodeHifiGan`].
    """
    dp = vocoder.dur_predictor
    dur_predictor = {
        "conv1": {
            "weight": _conv1d_weight(dp.conv1, device=device),
            "bias": _conv1d_bias(dp.conv1, device=device),
            "in_channels": dp.conv1.in_channels,
            "out_channels": dp.conv1.out_channels,
            "kernel_size": int(dp.conv1.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv1),
        },
        "ln1": {
            "weight": _ln_to_device(dp.ln1.weight, device=device),
            "bias": _ln_to_device(dp.ln1.bias, device=device),
            "eps": float(dp.ln1.eps),
        },
        "conv2": {
            "weight": _conv1d_weight(dp.conv2, device=device),
            "bias": _conv1d_bias(dp.conv2, device=device),
            "in_channels": dp.conv2.in_channels,
            "out_channels": dp.conv2.out_channels,
            "kernel_size": int(dp.conv2.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv2),
        },
        "ln2": {
            "weight": _ln_to_device(dp.ln2.weight, device=device),
            "bias": _ln_to_device(dp.ln2.bias, device=device),
            "eps": float(dp.ln2.eps),
        },
        "proj": _linear_pair(dp.proj, device=device),
    }

    hg = vocoder.hifi_gan
    upsampler_layers = []
    for layer in hg.upsampler:
        assert isinstance(layer, torch.nn.ConvTranspose1d)
        k = int(layer.kernel_size[0])
        s = int(layer.stride[0])
        p = _conv1d_like_padding_int(layer)
        upsampler_layers.append(
            {
                "weight": _conv_transpose1d_weight(layer, device=device),
                "bias": _conv_transpose1d_bias_host(layer),
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
            }
        )

    resblock_layers = []
    for rb in hg.resblocks:
        c1 = []
        c2 = []
        for c in rb.convs1:
            c1.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        for c in rb.convs2:
            c2.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        resblock_layers.append(make_parameter_dict({"convs1": c1, "convs2": c2}))

    out = {
        "unit_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.unit_embedding, device=device)}
        ),
        "speaker_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.speaker_embedding, device=device)}
        ),
        "language_embedding": make_parameter_dict(
            {"weight": _vocoder_embedding_weight_row_major(vocoder.language_embedding, device=device)}
        ),
        "dur_predictor": make_parameter_dict(dur_predictor),
        "hifi_gan": make_parameter_dict(
            {
                "conv_pre": {
                    "weight": _conv1d_weight(hg.conv_pre, device=device),
                    "bias": _conv1d_bias(hg.conv_pre, device=device),
                    "kernel_size": int(hg.conv_pre.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_pre),
                    "in_channels": hg.conv_pre.in_channels,
                    "out_channels": hg.conv_pre.out_channels,
                },
                "upsampler": upsampler_layers,
                "resblocks": resblock_layers,
                "conv_post": {
                    "weight": _conv1d_weight(hg.conv_post, device=device),
                    "bias": _conv1d_bias(hg.conv_post, device=device),
                    "kernel_size": int(hg.conv_post.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_post),
                    "in_channels": hg.conv_post.in_channels,
                    "out_channels": hg.conv_post.out_channels,
                },
            }
        ),
    }
    return make_parameter_dict(out)


def create_seamless_m4t_v2_model_parameters(model: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """
    Full [`SeamlessM4Tv2Model`] weights for TTNN: ``text_encoder``, ``text_decoder``, ``speech_encoder``,
    main ``lm_head``, ``t2u_model``, and ``vocoder`` (same submodules as Hugging Face).

    Automatically detects ``tp`` from ``device.get_num_devices()`` — 1 on P150, 4 on BH QB.
    TP sharding is applied to text encoder, text decoder, speech encoder, and T2U;
    vocoder (HiFiGAN) is always replicated.
    """
    tp = 1
    try:
        if hasattr(device, "get_num_devices"):
            tp = max(1, int(device.get_num_devices()))
    except Exception:
        tp = 1

    mm = _replicate_mapper(device)
    w_lm_torch = model.lm_head.weight.detach().T.contiguous().to(torch.bfloat16)
    w_lm = ttnn.from_torch(
        w_lm_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mm,
    )
    out = {
        "text_encoder": create_text_encoder_parameters(model.text_encoder, device=device, tp=tp),
        "text_decoder": create_text_decoder_parameters(model.text_decoder, device=device, tp=tp),
        "speech_encoder": create_speech_encoder_parameters(model.speech_encoder, device=device, tp=tp),
        "lm_head": make_parameter_dict({"weight": w_lm}),
        "t2u": create_text_to_unit_condgen_parameters(model.t2u_model, device=device, tp=tp),
        "vocoder": create_code_hifigan_parameters(model.vocoder, device=device),
    }
    return make_parameter_dict(out)
