# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger

import ttnn

from models.common.rmsnorm import RMSNorm
from models.demos.glm4_moe.tt.config import Glm4MoeHParams


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tp_axis_and_size(device: Any) -> tuple[int | None, int]:
    """Return (cluster_axis, tp_size) for a mesh.

    Galaxy TG Mesh(8,4): TP=axis 0 (rows=8), DP=axis 1 (cols=4).
    T3K Mesh(1,8): TP=axis 1 (cols=8).
    """
    if not _is_mesh_device(device):
        return (None, 1)
    mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
    if mesh_rows > 1 and mesh_cols > 1:
        # 2D mesh (TG): TP is axis 0 (rows=8), DP is axis 1 (cols=4)
        return (0, mesh_rows)
    if mesh_cols > 1:
        return (1, mesh_cols)
    if mesh_rows > 1:
        return (0, mesh_rows)
    return (None, 1)


def _tp_mesh_mapper(device: Any, *, shard_dim: int) -> Any | None:
    """Shard a tensor across the TP axis, replicate across the other axis.

    For Galaxy TG mesh (8, 4): TP=8 is axis 0 (rows), DP=4 is axis 1 (cols).
    For T3K mesh (1, 8): TP=8 is axis 1 (cols).
    """
    if not _is_mesh_device(device):
        return None
    mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
    if mesh_rows > 1 and mesh_cols > 1:
        # 2D mesh (TG): TP is axis 0 (rows=8), DP is axis 1 (cols=4)
        return ttnn.ShardTensor2dMesh(device, dims=(int(shard_dim), None), mesh_shape=list(device.shape))
    if mesh_cols > 1:
        return ttnn.ShardTensor2dMesh(device, dims=(None, int(shard_dim)), mesh_shape=list(device.shape))
    if mesh_rows > 1:
        return ttnn.ShardTensor2dMesh(device, dims=(int(shard_dim), None), mesh_shape=list(device.shape))
    return ttnn.ReplicateTensorToMesh(device)


def _replicate_mapper(device: Any) -> Any | None:
    if not _is_mesh_device(device):
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _env_experts_dtype() -> ttnn.DataType:
    """Return TT dtype for routed expert weights.

    Default is BF8 for memory efficiency; override via GLM4_MOE_EXPERTS_TT_DTYPE.
    """
    override = os.environ.get("GLM4_MOE_EXPERTS_TT_DTYPE", "").strip().lower()
    if not override:
        return ttnn.bfloat8_b
    if override in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if override in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if override in {"f32", "fp32", "float32"}:
        return ttnn.float32
    raise ValueError(f"Invalid GLM4_MOE_EXPERTS_TT_DTYPE={override!r}")


def _env_dense_dtype() -> ttnn.DataType:
    """Return TT dtype for dense projection weights (attention Q/O, dense MLP).

    Default is BF8 for memory efficiency; override via GLM4_MOE_DENSE_TT_DTYPE.
    """
    override = os.environ.get("GLM4_MOE_DENSE_TT_DTYPE", "").strip().lower()
    if not override:
        return ttnn.bfloat8_b
    if override in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if override in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if override in {"f32", "fp32", "float32"}:
        return ttnn.float32
    raise ValueError(f"Invalid GLM4_MOE_DENSE_TT_DTYPE={override!r}")


def _linear_weight_tt(
    *,
    device,
    torch_weight_out_in: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    mesh_mapper: Any | None = None,
) -> ttnn.Tensor:
    """Convert a torch Linear weight in HF layout [out, in] into TT layout [1, 1, in, out]."""
    if torch_weight_out_in.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape={tuple(torch_weight_out_in.shape)}")
    w = torch_weight_out_in.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
    is_mesh = _is_mesh_device(device)
    if is_mesh and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper if is_mesh else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _vector_weight_tt(
    *,
    device,
    torch_vector: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    mesh_mapper: Any | None = None,
) -> ttnn.Tensor:
    """Convert a torch vector into TT [1,1,1,E] ROW_MAJOR."""
    if torch_vector.ndim != 1:
        raise ValueError(f"expected 1D vector, got shape={tuple(torch_vector.shape)}")
    v = torch_vector.contiguous().view(1, 1, 1, -1)
    is_mesh = _is_mesh_device(device)
    if is_mesh and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.as_tensor(
        v,
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper if is_mesh else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _experts_weight_tt(
    *,
    device,
    torch_weights: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> ttnn.Tensor:
    """Convert stacked expert weights for sparse MoE matmuls.

    For a MeshDevice with EP=32, expert weights are sharded across ALL 32 devices
    (3 experts per device for 96 total).

    Host tensor shape: [num_devices, 1, experts_per_device, in, out]
    mesh_mapper: ShardTensorToMesh(dim=0)
    """
    if torch_weights.ndim != 3:
        raise ValueError(f"expected [E,in,out] weights, got shape={tuple(torch_weights.shape)}")
    is_mesh = _is_mesh_device(device)
    if is_mesh:
        num_devices = int(device.get_num_devices())
        num_experts = int(torch_weights.shape[0])
        if num_experts % num_devices != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by num_devices={num_devices} for expert sharding"
            )
        experts_per_device = num_experts // num_devices
        w = (
            torch_weights.contiguous()
            .view(num_devices, experts_per_device, int(torch_weights.shape[1]), int(torch_weights.shape[2]))
            .unsqueeze(1)
            .contiguous()
        )
        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
    else:
        w = torch_weights.unsqueeze(0).contiguous()
        mesh_mapper = None
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


@dataclass(frozen=True)
class MoELayerTTWeights:
    """Per-layer routed MoE weights (gate + experts).

    Shapes (after conversion, TT layout):
    - w_gate: [1,1,hidden,n_routed_experts]
    - e_score_correction_bias: [1,1,1,n_routed_experts] row-major
    - w1_experts (gate_proj): per-device [1,experts_per_device,hidden,moe_intermediate]
    - w3_experts (up_proj):   per-device [1,experts_per_device,hidden,moe_intermediate]
    - w2_experts (down_proj): per-device [1,experts_per_device,moe_intermediate,hidden]
    """

    w_gate: ttnn.Tensor
    e_score_correction_bias: ttnn.Tensor
    w1_experts: ttnn.Tensor
    w2_experts: ttnn.Tensor
    w3_experts: ttnn.Tensor
    e_score_correction_bias_tile: Optional[ttnn.Tensor] = None


@dataclass(frozen=True)
class DecoderLayerTTWeights:
    """Per-layer weights for GLM-4.7-REAP decoder with standard GQA attention.

    Attention uses fused QKV (following tt_transformers pattern):
    - w_qkv: [1,1,hidden, (q_local+k_local+v_local)*num_devices] sharded across TP
    - w_qkv_bias: fused bias for QKV, sharded same as w_qkv
    - w_o: [1,1,total_head_dim/tp, hidden] row-parallel across TP

    MLP: dense layers (0-2) or shared expert + routed MoE (layers 3-91).
    """

    layer_idx: int

    # Layer norms
    input_layernorm: Any
    post_attention_layernorm: Any

    # Attention: fused QKV weight + bias
    w_qkv: ttnn.Tensor
    w_qkv_bias: ttnn.Tensor
    w_o: ttnn.Tensor

    # QK norm weights (per-head RMSNorm, dim=head_dim=128, replicated)
    q_norm: Any
    k_norm: Any

    # MLP projections (dense layers 0-2, or shared expert for MoE layers)
    w_mlp_gate: ttnn.Tensor
    w_mlp_up: ttnn.Tensor
    w_mlp_down: ttnn.Tensor

    # Fused gate+up weight: [1,1,hidden,2*inter/TP] for single-matmul shared expert
    w_mlp_gate_up: Optional[ttnn.Tensor] = None

    # Optional routed MoE weights (layers >= first_k_dense_replace)
    moe: Optional[MoELayerTTWeights] = None


def convert_decoder_layer_weights(
    *,
    device,
    state,
    layer_idx: int,
    hparams: Glm4MoeHParams,
    cache_dir: Optional[Path] = None,
    enable_moe: bool = True,
) -> DecoderLayerTTWeights:
    """Convert weights for a single decoder layer from HF safetensors to TT format.

    Handles:
    - Fused QKV with TP=8 sharding (chunk Q/K/V per head, fuse per device)
    - QKV bias fusion (same TP chunking)
    - QK norm replication
    - O projection row-parallel TP sharding
    - Dense MLP (layers 0-2) with column/row parallel TP
    - Shared expert MLP with column/row parallel TP
    - Routed expert weights with EP=32 sharding (3 experts/device for 96 total)
    """
    cache_dir = None if cache_dir is None else Path(cache_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    layer_idx = int(layer_idx)

    def c(name: str, variant: str = "") -> Optional[Path]:
        suffix = variant.strip()
        suffix = f"_{suffix}" if suffix and not suffix.startswith("_") else suffix
        return None if cache_dir is None else cache_dir / f"layer{layer_idx}_{name}{suffix}"

    dense_dtype = _env_dense_dtype()
    tp_axis, tp_size = _tp_axis_and_size(device)
    num_devices = int(device.get_num_devices()) if _is_mesh_device(device) else 1

    tp_variant = f"tp{tp_size}" if tp_size > 1 else ""

    # ---- Layer Norms (replicated) ----
    input_layernorm = RMSNorm(
        device=device,
        dim=hparams.hidden_size,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.",
        weight_key="input_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    post_attention_layernorm = RMSNorm(
        device=device,
        dim=hparams.hidden_size,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.",
        weight_key="post_attention_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )

    # ---- QK Norm (replicated, per-head dim=128) ----
    q_norm = RMSNorm(
        device=device,
        dim=hparams.head_dim,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.self_attn.",
        weight_key="q_norm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    k_norm = RMSNorm(
        device=device,
        dim=hparams.head_dim,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.self_attn.",
        weight_key="k_norm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    logger.info("  [DEBUG L{}] norms done", layer_idx)

    # ---- Fused QKV Weights (following tt_transformers/attention.py pattern) ----
    # Load Q [12288, 5120], K [1024, 5120], V [1024, 5120]
    wq_full = state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]  # [12288, 5120]
    wk_full = state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]  # [1024, 5120]
    wv_full = state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]  # [1024, 5120]

    # Chunk by TP=8 along output dim (head dim), transpose, and fuse per device
    # Q: [12288, 5120] -> 8 chunks of [1536, 5120] -> transpose -> [5120, 1536]
    # K: [1024, 5120]  -> 8 chunks of [128, 5120]  -> transpose -> [5120, 128]
    # V: [1024, 5120]  -> 8 chunks of [128, 5120]  -> transpose -> [5120, 128]
    # Fused per device: [5120, 1536+128+128] = [5120, 1792]
    wq_chunks = torch.chunk(wq_full, tp_size, dim=0)
    wk_chunks = torch.chunk(wk_full, tp_size, dim=0)
    wv_chunks = torch.chunk(wv_full, tp_size, dim=0)

    qkv_list = []
    for i in range(tp_size):
        wq_t = wq_chunks[i].transpose(-2, -1)  # [5120, 1536]
        wk_t = wk_chunks[i].transpose(-2, -1)  # [5120, 128]
        wv_t = wv_chunks[i].transpose(-2, -1)  # [5120, 128]
        qkv = torch.cat([wq_t, wk_t, wv_t], dim=-1)  # [5120, 1792]
        qkv_list.append(qkv)

    # Concatenate all TP chunks along last dim -> [5120, 1792*tp_size]
    # Then shard across mesh via ShardTensor2dMesh
    qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, 5120, 1792*tp_size]

    if _is_mesh_device(device):
        # Shard along dim=3 across TP, replicate across DP
        qkv_mapper = _tp_mesh_mapper(device, shard_dim=3)
    else:
        qkv_mapper = None

    w_qkv = ttnn.as_tensor(
        qkv_cat,
        dtype=dense_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=qkv_mapper,
        cache_file_name=c("w_qkv", tp_variant),
    )
    logger.info("  [DEBUG L{}] w_qkv done", layer_idx)

    # ---- Fused QKV Bias ----
    bq_full = state[f"model.layers.{layer_idx}.self_attn.q_proj.bias"]  # [12288]
    bk_full = state[f"model.layers.{layer_idx}.self_attn.k_proj.bias"]  # [1024]
    bv_full = state[f"model.layers.{layer_idx}.self_attn.v_proj.bias"]  # [1024]

    bq_chunks = torch.chunk(bq_full, tp_size, dim=0)
    bk_chunks = torch.chunk(bk_full, tp_size, dim=0)
    bv_chunks = torch.chunk(bv_full, tp_size, dim=0)

    bias_list = []
    for i in range(tp_size):
        b_fused = torch.cat([bq_chunks[i], bk_chunks[i], bv_chunks[i]], dim=-1)  # [1792]
        bias_list.append(b_fused)

    # Concatenate all TP chunks -> [1792*tp_size], reshape to [1, 1, 1, 1792*tp_size]
    bias_cat = torch.cat(bias_list, dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    w_qkv_bias = ttnn.as_tensor(
        bias_cat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=qkv_mapper,
        cache_file_name=c("w_qkv_bias", tp_variant),
    )
    logger.info("  [DEBUG L{}] w_qkv_bias done", layer_idx)

    # ---- O Projection (row-parallel: shard input dim by TP) ----
    # HF: [5120, 12288] -> TT: [1, 1, 12288, 5120], shard dim=2 (input) by TP
    # Each device gets [1, 1, 1536, 5120]
    wo_mapper = _tp_mesh_mapper(device, shard_dim=2) if tp_size > 1 else _replicate_mapper(device)
    w_o = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
        cache_file=c("w_o", tp_variant),
        dtype=dense_dtype,
        mesh_mapper=wo_mapper,
    )
    logger.info("  [DEBUG L{}] w_o done", layer_idx)

    # ---- MLP Weights ----
    dense_layer = layer_idx < int(hparams.first_k_dense_replace)

    if dense_layer:
        # Dense MLP (layers 0-2): intermediate_size=12288
        mlp_prefix = f"model.layers.{layer_idx}.mlp."
    else:
        # Shared expert MLP: moe_intermediate_size=1536
        mlp_prefix = f"model.layers.{layer_idx}.mlp.shared_experts."

    # Column-parallel: gate/up shard output dim (dim=3 in TT layout after transpose)
    # Row-parallel: down shard input dim (dim=2 in TT layout after transpose)
    mlp_gate_mapper = _tp_mesh_mapper(device, shard_dim=3) if tp_size > 1 else _replicate_mapper(device)
    mlp_down_mapper = _tp_mesh_mapper(device, shard_dim=2) if tp_size > 1 else _replicate_mapper(device)

    mlp_variant = tp_variant

    _gate_w = state[f"{mlp_prefix}gate_proj.weight"]
    _up_w = state[f"{mlp_prefix}up_proj.weight"]

    if dense_layer:
        # Dense layers (0-2): keep separate gate/up for _dense_mlp_forward.
        w_mlp_gate_up = None
        w_mlp_gate = _linear_weight_tt(
            device=device,
            torch_weight_out_in=_gate_w,
            cache_file=c("w_mlp_gate", mlp_variant),
            dtype=dense_dtype,
            mesh_mapper=mlp_gate_mapper,
        )
        w_mlp_up = _linear_weight_tt(
            device=device,
            torch_weight_out_in=_up_w,
            cache_file=c("w_mlp_up", mlp_variant),
            dtype=dense_dtype,
            mesh_mapper=mlp_gate_mapper,
        )
    else:
        # MoE layers (3-91): fuse gate+up, skip separate weights to save DRAM.
        # Interleave gate/up chunks so column-parallel TP sharding gives each
        # device [gate_shard, up_shard] instead of a contiguous slice of the
        # concatenated [gate_full; up_full] which would be wrong.
        inter = _gate_w.shape[0]  # 1536
        chunk = inter // tp_size  # 192 per device
        _gate_chunks = _gate_w.reshape(tp_size, chunk, -1)  # [TP, chunk, hidden]
        _up_chunks = _up_w.reshape(tp_size, chunk, -1)
        # Stack [gate_i, up_i] for each TP rank, then flatten:
        # [TP, 2, chunk, hidden] → [TP*2*chunk, hidden] = [2*inter, hidden]
        _gate_up_fused = torch.stack([_gate_chunks, _up_chunks], dim=1).reshape(-1, _gate_w.shape[1])
        w_mlp_gate_up = _linear_weight_tt(
            device=device,
            torch_weight_out_in=_gate_up_fused,
            cache_file=c("w_mlp_gate_up_v2", mlp_variant),
            dtype=dense_dtype,
            mesh_mapper=mlp_gate_mapper,
        )
        w_mlp_gate = w_mlp_gate_up  # placeholder (unused when w_gate_up is set)
        w_mlp_up = w_mlp_gate_up    # placeholder (unused when w_gate_up is set)
    logger.info("  [DEBUG L{}] mlp gate/up done (dense={})", layer_idx, dense_layer)
    w_mlp_down = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"{mlp_prefix}down_proj.weight"],
        cache_file=c("w_mlp_down", mlp_variant),
        dtype=dense_dtype,
        mesh_mapper=mlp_down_mapper,
    )
    logger.info("  [DEBUG L{}] mlp down done", layer_idx)

    # ---- Routed MoE (layers >= first_k_dense_replace) ----
    moe: Optional[MoELayerTTWeights] = None
    if enable_moe and not dense_layer:
        import sys as _sys
        _dbg_sync = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
        def _msync(label):
            if _dbg_sync:
                print(f"  [DEBUG MOE L{layer_idx}] {label} ...", flush=True, file=_sys.stderr)
                ttnn.synchronize_device(device)
                print(f"  [DEBUG MOE L{layer_idx}] {label} OK", flush=True, file=_sys.stderr)

        # Gate: [96, 5120] -> [1, 1, 5120, 96] replicated
        w_gate = _linear_weight_tt(
            device=device,
            torch_weight_out_in=state[f"model.layers.{layer_idx}.mlp.gate.weight"],
            cache_file=c("w_moe_gate"),
            dtype=ttnn.bfloat16,
        )
        _msync("after gate weight")

        # e_score_correction_bias: center before BF16 cast to preserve ordering
        e_bias_torch = state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"].to(dtype=torch.float32)
        e_bias_centered = e_bias_torch - float(e_bias_torch.min().item())
        e_bias = _vector_weight_tt(
            device=device,
            torch_vector=e_bias_centered,
            cache_file=c("e_score_correction_bias_centered_v1"),
            dtype=ttnn.bfloat16,
        )
        _msync("after e_bias vector")
        # Create tile-layout version directly via as_tensor (ttnn.to_layout hangs on TG mesh)
        is_mesh = _is_mesh_device(device)
        e_bias_tile = ttnn.as_tensor(
            e_bias_centered.contiguous().view(1, 1, 1, -1),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )
        _msync("after e_bias_tile as_tensor")

        # Stack and shard expert weights across all 32 devices (EP=32)
        experts_dtype = _env_experts_dtype()
        num_experts = int(hparams.n_routed_experts)
        moe_intermediate = int(hparams.moe_intermediate_size)
        hidden = int(hparams.hidden_size)
        experts_variant = f"ep{num_devices}_v1"

        w1_list: list[torch.Tensor] = []
        w3_list: list[torch.Tensor] = []
        w2_list: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            w1 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"]
            w3 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"]
            w2 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"]
            if tuple(w1.shape) != (moe_intermediate, hidden):
                raise ValueError(
                    f"Unexpected gate_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w1.shape)}, "
                    f"expected ({moe_intermediate}, {hidden})"
                )
            if tuple(w3.shape) != (moe_intermediate, hidden):
                raise ValueError(
                    f"Unexpected up_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w3.shape)}, "
                    f"expected ({moe_intermediate}, {hidden})"
                )
            if tuple(w2.shape) != (hidden, moe_intermediate):
                raise ValueError(
                    f"Unexpected down_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w2.shape)}, "
                    f"expected ({hidden}, {moe_intermediate})"
                )
            # Transpose to TT convention: [in, out]
            w1_list.append(w1.transpose(-2, -1).contiguous())  # [hidden, moe_intermediate]
            w3_list.append(w3.transpose(-2, -1).contiguous())  # [hidden, moe_intermediate]
            w2_list.append(w2.transpose(-2, -1).contiguous())  # [moe_intermediate, hidden]

        w1_stacked = torch.stack(w1_list, dim=0)  # [96, hidden, moe_intermediate]
        w3_stacked = torch.stack(w3_list, dim=0)  # [96, hidden, moe_intermediate]
        w2_stacked = torch.stack(w2_list, dim=0)  # [96, moe_intermediate, hidden]

        _msync("after expert stacking")
        w1_experts = _experts_weight_tt(
            device=device,
            torch_weights=w1_stacked,
            cache_file=c("w1_experts", experts_variant),
            dtype=experts_dtype,
        )
        _msync("after w1_experts")
        w3_experts = _experts_weight_tt(
            device=device,
            torch_weights=w3_stacked,
            cache_file=c("w3_experts", experts_variant),
            dtype=experts_dtype,
        )
        _msync("after w3_experts")
        w2_experts = _experts_weight_tt(
            device=device,
            torch_weights=w2_stacked,
            cache_file=c("w2_experts", experts_variant),
            dtype=experts_dtype,
        )
        _msync("after w2_experts")

        moe = MoELayerTTWeights(
            w_gate=w_gate,
            e_score_correction_bias=e_bias,
            w1_experts=w1_experts,
            w2_experts=w2_experts,
            w3_experts=w3_experts,
            e_score_correction_bias_tile=e_bias_tile,
        )

    return DecoderLayerTTWeights(
        layer_idx=layer_idx,
        input_layernorm=input_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        w_qkv=w_qkv,
        w_qkv_bias=w_qkv_bias,
        w_o=w_o,
        q_norm=q_norm,
        k_norm=k_norm,
        w_mlp_gate=w_mlp_gate,
        w_mlp_up=w_mlp_up,
        w_mlp_down=w_mlp_down,
        w_mlp_gate_up=w_mlp_gate_up,
        moe=moe,
    )
