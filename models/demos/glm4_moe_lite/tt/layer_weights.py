# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

import ttnn

from models.common.rmsnorm import RMSNorm
from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.weights import LazyStateDict


def _env_tp_enabled() -> bool:
    return os.environ.get("GLM4_MOE_LITE_TP", "").strip() == "1"


def _env_fuse_qkv_a() -> bool:
    return os.environ.get("GLM4_MOE_LITE_FUSE_QKV_A", "").strip() == "1"


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tp_axis_and_size(device: Any) -> tuple[int | None, int]:
    """Return (cluster_axis, tp_size) for a mesh.

    Convention:
    - Prefer sharding along mesh cols when available.
    - Otherwise shard along mesh rows.
    """
    if not _is_mesh_device(device):
        return (None, 1)
    mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
    if mesh_cols > 1:
        return (1, mesh_cols)
    if mesh_rows > 1:
        return (0, mesh_rows)
    return (None, 1)


def _tp_mesh_mapper(device: Any, *, shard_dim: int) -> Any | None:
    """Shard a tensor across the TP axis, replicate across the other axis if present."""
    if not _is_mesh_device(device):
        return None
    mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
    if mesh_cols > 1:
        return ttnn.ShardTensor2dMesh(device, dims=(None, int(shard_dim)), mesh_shape=list(device.shape))
    if mesh_rows > 1:
        return ttnn.ShardTensor2dMesh(device, dims=(int(shard_dim), None), mesh_shape=list(device.shape))
    return ttnn.ReplicateTensorToMesh(device)


def _env_experts_dtype() -> ttnn.DataType:
    """Return TT dtype for routed expert weights.

    Default is BF8 for memory efficiency; override via `GLM4_MOE_LITE_EXPERTS_TT_DTYPE`.
    """
    override = os.environ.get("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "").strip().lower()
    if not override:
        return ttnn.bfloat8_b
    if override in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if override in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if override in {"f16", "fp16", "float16"}:
        return ttnn.float16
    if override in {"f32", "fp32", "float32"}:
        return ttnn.float32
    raise ValueError(f"Invalid GLM4_MOE_LITE_EXPERTS_TT_DTYPE={override!r}")


def _env_dense_dtype() -> ttnn.DataType:
    """Return TT dtype for dense/non-router projection weights.

    Default is BF16 for correctness. BF8 is exposed as an opt-in experiment
    for throughput tuning.
    """
    override = os.environ.get("GLM4_MOE_LITE_DENSE_TT_DTYPE", "").strip().lower()
    if not override:
        return ttnn.bfloat16
    if override in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if override in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if override in {"f16", "fp16", "float16"}:
        return ttnn.float16
    if override in {"f32", "fp32", "float32"}:
        return ttnn.float32
    raise ValueError(f"Invalid GLM4_MOE_LITE_DENSE_TT_DTYPE={override!r}")


def _env_attn_dp() -> bool:
    """When enabled, replicate attention projection weights (no TP sharding).

    This removes the per-projection all_reduce calls for w_q_kv_a, w_q_a,
    w_kv_a, w_q_b, and w_kv_b2. w_o remains row-parallel (still needs all_reduce).
    """
    return os.environ.get("GLM4_MOE_LITE_ATTN_DP", "").strip() == "1"


def _env_dram_sharded_weights() -> bool:
    return os.environ.get("GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS", "").strip() == "1"


def _env_dram_sharded_attn() -> bool:
    """Separate flag for attention DRAM sharding (off by default — resharding overhead + trace issues).

    Enable via GLM4_MOE_LITE_DRAM_SHARDED_ATTN=1 to experiment.
    Requires GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 to be set.
    """
    return _env_dram_sharded_weights() and os.environ.get("GLM4_MOE_LITE_DRAM_SHARDED_ATTN", "").strip() == "1"


def _env_dram_sharded_mlp() -> bool:
    """MLP DRAM sharding (ON by default when main flag is set).

    Disable via GLM4_MOE_LITE_DRAM_SHARDED_MLP=0 to opt out.
    Requires GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 to be set.
    """
    return _env_dram_sharded_weights() and os.environ.get("GLM4_MOE_LITE_DRAM_SHARDED_MLP", "1").strip() != "0"


def _env_sharded_mlp() -> bool:
    """Standalone sharded MLP flag (no dependency on DRAM_SHARDED_WEIGHTS).

    Enable via GLM4_MOE_LITE_SHARDED_MLP=1 to use DRAM-sharded weights + L1
    WIDTH_SHARDED activations for the shared MLP decode path, following the
    DeepSeek V3 pattern.
    """
    return os.environ.get("GLM4_MOE_LITE_SHARDED_MLP", "").strip() == "1"


def _maybe_dram_shard_linear_weight(weight: ttnn.Tensor, device, force: bool = False) -> ttnn.Tensor:
    """Convert a [1,1,K,N] linear weight to DRAM-sharded format for decode perf.

    DRAM-sharded storage distributes the weight's N dimension across all DRAM banks,
    giving full DRAM bandwidth utilization for M=1 decode matmuls.
    """
    if not force and not _env_dram_sharded_weights():
        return weight
    from models.demos.deepseek_v3.utils.config_helpers import dram_sharded_weight_config

    K = int(weight.shape[2])
    N = int(weight.shape[3])
    dram_mc = dram_sharded_weight_config(K, N, device.dram_grid_size())
    return ttnn.to_memory_config(weight, dram_mc)


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
    is_mesh_device = _is_mesh_device(device)
    if is_mesh_device and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper if is_mesh_device else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _per_head_weight_tt(
    *,
    device,
    torch_weight: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    mesh_mapper: Any | None = None,
) -> ttnn.Tensor:
    """Convert a per-head torch weight [H, in, out] into TT weight [1, H, in, out]."""
    if torch_weight.ndim != 3:
        raise ValueError(f"expected [H,in,out] weight, got shape={tuple(torch_weight.shape)}")
    w = torch_weight.unsqueeze(0).contiguous()
    is_mesh_device = _is_mesh_device(device)
    if is_mesh_device and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper if is_mesh_device else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _vector_weight_tt(
    *,
    device,
    torch_vector: torch.Tensor,  # [E]
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    mesh_mapper: Any | None = None,
) -> ttnn.Tensor:
    """Convert a torch vector into TT [1,1,1,E] ROW_MAJOR."""
    if torch_vector.ndim != 1:
        raise ValueError(f"expected 1D vector, got shape={tuple(torch_vector.shape)}")
    v = torch_vector.contiguous().view(1, 1, 1, -1)
    is_mesh_device = _is_mesh_device(device)
    if is_mesh_device and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.as_tensor(
        v,
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper if is_mesh_device else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _experts_weight_tt(
    *,
    device,
    torch_weights: torch.Tensor,  # [num_experts, in, out]
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> ttnn.Tensor:
    """Convert stacked expert weights for sparse MoE matmuls.

    For a MeshDevice, `ttnn.moe_expert_token_remap` produces per-device sparsity with
    last-dim = `experts_per_device`. `ttnn.sparse_matmul` validates that sparsity
    volume matches the product of the *broadcasted batch dims* of (A, B).

    To keep that validation local (experts_per_device, not global num_experts), we
    represent expert weights as an implicitly sharded mesh tensor:
    - Host tensor shape: [num_devices, 1, experts_per_device, in, out]
    - mesh_mapper: ShardTensorToMesh(dim=0) so the leading device dimension is
      distributed across the mesh and is not part of the logical tensor shape.
    """
    if torch_weights.ndim != 3:
        raise ValueError(f"expected [E,in,out] weights, got shape={tuple(torch_weights.shape)}")
    is_mesh_device = _is_mesh_device(device)
    if is_mesh_device:
        num_devices = int(device.get_num_devices())
        num_experts = int(torch_weights.shape[0])
        if num_experts % max(1, num_devices) != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by num_devices={num_devices} for expert sharding"
            )
        experts_per_device = num_experts // max(1, num_devices)
        # [E,in,out] -> [D,1,E_local,in,out] where dim0 is implicitly sharded across mesh.
        w = (
            torch_weights.contiguous()
            .view(num_devices, experts_per_device, int(torch_weights.shape[1]), int(torch_weights.shape[2]))
            .unsqueeze(1)
            .contiguous()
        )
        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
    else:
        w = torch_weights.unsqueeze(0).contiguous()  # [1,E,in,out]
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

    Shapes (global):
    - w_gate: [1,1,hidden,n_routed_experts]
    - e_score_correction_bias: [1,1,1,n_routed_experts] row-major
    - w1_experts (gate_proj): [1,n_routed_experts,hidden,moe_intermediate]
    - w3_experts (up_proj):   [1,n_routed_experts,hidden,moe_intermediate]
    - w2_experts (down_proj): [1,n_routed_experts,moe_intermediate,hidden]
    - w1w3_experts (fused gate+up): [1,n_routed_experts,hidden,2*moe_intermediate] (optional)
    """

    w_gate: ttnn.Tensor
    e_score_correction_bias: ttnn.Tensor
    w1_experts: ttnn.Tensor
    w2_experts: ttnn.Tensor
    w3_experts: ttnn.Tensor
    w1w3_experts: Optional[ttnn.Tensor] = None


@dataclass(frozen=True)
class DecoderLayerTTWeights:
    """Minimal per-layer weights for GLM-4.7-Flash decoder execution on TT.

    Notes:
    - This is an unoptimized bring-up representation: weights are replicated and
      stored in DRAM as BF16 unless overridden. Later phases will shard and/or
      quantize.
    - For layers >= `first_k_dense_replace`, the MLP weights are taken from
      `mlp.shared_experts.*` when running in dense-only bring-up mode.
    """

    layer_idx: int

    # Layer norms
    input_layernorm: Any
    q_a_layernorm: Any
    kv_a_layernorm: Any
    post_attention_layernorm: Any

    # Attention projections
    w_q_a: ttnn.Tensor
    w_q_b: ttnn.Tensor
    w_kv_a: ttnn.Tensor
    w_kv_b1: ttnn.Tensor
    w_kv_b2: ttnn.Tensor
    w_o: ttnn.Tensor

    # MLP projections (dense or shared-expert-as-dense)
    w_mlp_gate: ttnn.Tensor
    w_mlp_up: ttnn.Tensor
    w_mlp_down: ttnn.Tensor

    # Optional fused attention projection (q_a + kv_a) for one matmul.
    w_q_kv_a: Optional[ttnn.Tensor] = None

    # Optional routed MoE weights (layers >= first_k_dense_replace)
    moe: Optional[MoELayerTTWeights] = None


def convert_decoder_layer_weights(
    *,
    device,
    state: LazyStateDict,
    layer_idx: int,
    hparams: Glm4MoeLiteHParams,
    cache_dir: Optional[Path] = None,
    force_shared_expert_dense: bool = False,
    enable_moe: bool = False,
) -> DecoderLayerTTWeights:
    """Convert weights for a single decoder layer.

    bring-up mode:
    - Layer 0 uses the dense MLP weights under `model.layers.0.mlp.*`.
    - Layers >= first_k_dense_replace use `mlp.shared_experts.*` as a temporary
      dense MLP (until routed experts are implemented).
    """
    cache_dir = None if cache_dir is None else Path(cache_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    layer_idx = int(layer_idx)

    def c(name: str, variant: str = "") -> Optional[Path]:
        suffix = variant.strip()
        suffix = f"_{suffix}" if suffix and not suffix.startswith("_") else suffix
        return None if cache_dir is None else cache_dir / f"layer{layer_idx}_{name}{suffix}"

    # ---- Norms ----
    dense_dtype = _env_dense_dtype()
    tp_enabled = _env_tp_enabled()
    tp_axis, tp_size = _tp_axis_and_size(device)

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
    q_a_layernorm = RMSNorm(
        device=device,
        dim=hparams.q_lora_rank,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.self_attn.",
        weight_key="q_a_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    kv_a_layernorm = RMSNorm(
        device=device,
        dim=hparams.kv_lora_rank,
        eps=hparams.rms_norm_eps,
        state_dict=state,
        state_dict_prefix=f"model.layers.{layer_idx}.self_attn.",
        weight_key="kv_a_layernorm",
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

    # ---- Attention projections ----
    attn_row_mapper = None
    attn_variant = ""
    attn_dp = _env_attn_dp()
    if tp_enabled and tp_size > 1:
        hidden = int(hparams.hidden_size)
        q_lora = int(hparams.q_lora_rank)
        kv_lora = int(hparams.kv_lora_rank)
        qk_nope = int(hparams.qk_nope_head_dim)
        in_o = int(hparams.num_attention_heads) * int(hparams.v_head_dim)
        if hidden % int(tp_size) != 0:
            raise ValueError(f"TP enabled but hidden_size={hidden} not divisible by tp_size={tp_size}")
        if q_lora % int(tp_size) != 0:
            raise ValueError(f"TP enabled but q_lora_rank={q_lora} not divisible by tp_size={tp_size}")
        if kv_lora % int(tp_size) != 0:
            raise ValueError(f"TP enabled but kv_lora_rank={kv_lora} not divisible by tp_size={tp_size}")
        if qk_nope % int(tp_size) != 0:
            raise ValueError(f"TP enabled but qk_nope_head_dim={qk_nope} not divisible by tp_size={tp_size}")
        if in_o % int(tp_size) != 0:
            raise ValueError(
                f"TP enabled but attention out in_dim={in_o} (num_heads*v_head_dim) not divisible by tp_size={tp_size}"
            )
        # Attention projection weights use row-parallel sharding (shard input dim).
        attn_variant = f"tp{tp_size}"
        attn_row_mapper = _tp_mesh_mapper(device, shard_dim=2)

    # When ATTN_DP=1, replicate attention projection weights (w_q_a, w_q_b, w_kv_a,
    # w_q_kv_a, w_kv_b2) so each device has the full weight. This removes per-projection
    # all_reduce calls. w_o stays row-parallel (needs all_reduce for correctness).
    attn_proj_mapper = None if attn_dp else attn_row_mapper
    attn_proj_variant = f"{attn_variant}_attndp" if attn_dp and attn_variant else attn_variant

    w_q_a = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"model.layers.{layer_idx}.self_attn.q_a_proj.weight"],
        cache_file=c("w_q_a", attn_proj_variant),
        dtype=dense_dtype,
        mesh_mapper=attn_proj_mapper,
    )
    w_q_b = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"model.layers.{layer_idx}.self_attn.q_b_proj.weight"],
        cache_file=c("w_q_b", attn_proj_variant),
        dtype=dense_dtype,
        mesh_mapper=attn_proj_mapper,
    )
    w_kv_a = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"],
        cache_file=c("w_kv_a", attn_proj_variant),
        dtype=dense_dtype,
        mesh_mapper=attn_proj_mapper,
    )
    w_q_kv_a: Optional[ttnn.Tensor] = None
    if _env_fuse_qkv_a():
        q_a_torch = state[f"model.layers.{layer_idx}.self_attn.q_a_proj.weight"]
        kv_a_torch = state[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"]
        if q_a_torch.ndim != 2 or kv_a_torch.ndim != 2:
            raise ValueError(
                f"unexpected q_a/kv_a ranks: q_a={tuple(q_a_torch.shape)} kv_a={tuple(kv_a_torch.shape)}"
            )
        if int(q_a_torch.shape[1]) != int(kv_a_torch.shape[1]):
            raise ValueError(
                f"q_a and kv_a must share input dim; got q_a_in={int(q_a_torch.shape[1])} kv_a_in={int(kv_a_torch.shape[1])}"
            )
        fused_out_in = torch.cat([q_a_torch, kv_a_torch], dim=0)
        fused_base = f"fused_{attn_proj_variant}_v1" if attn_proj_variant else "fused_v1"
        w_q_kv_a = _linear_weight_tt(
            device=device,
            torch_weight_out_in=fused_out_in,
            cache_file=c("w_q_kv_a", fused_base),
            dtype=dense_dtype,
            mesh_mapper=attn_proj_mapper,
        )
        if _env_dram_sharded_attn():
            w_q_kv_a = _maybe_dram_shard_linear_weight(w_q_kv_a, device)

    kv_b = state[f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"]  # [num_heads*(qk_nope+v), kv_lora]
    kv_b = kv_b.view(hparams.num_attention_heads, hparams.qk_nope_head_dim + hparams.v_head_dim, hparams.kv_lora_rank)
    w_kv_b1_torch = kv_b[:, : hparams.qk_nope_head_dim, :].contiguous()  # [H, qk_nope, kv_lora]
    w_kv_b2_torch = kv_b[:, -hparams.v_head_dim :, :].transpose(1, 2).contiguous()  # [H, kv_lora, v]

    # `ttnn.mesh_partition` (used for row-parallel activation sharding) requires tile-aligned
    # slice boundaries on tilized tensors. Some attention per-head dims (e.g., qk_nope=192)
    # are divisible by tp_size=8 but not by TILE_SIZE*tp_size, so row-parallel sharding
    # would fail at runtime. Fall back to replication for those weights.
    # Note: w_kv_b1 is already replicated — don't touch it regardless of ATTN_DP.
    w_kv_b1_mapper = attn_row_mapper
    w_kv_b1_variant = attn_variant
    if tp_enabled and tp_size > 1:
        qk_nope_per_shard = int(hparams.qk_nope_head_dim) // int(tp_size)
        if qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            w_kv_b1_mapper = None  # replicate
            w_kv_b1_variant = f"{attn_variant}_rep"

    w_kv_b1 = _per_head_weight_tt(
        device=device,
        torch_weight=w_kv_b1_torch,
        cache_file=c("w_kv_b1", w_kv_b1_variant),
        dtype=dense_dtype,
        mesh_mapper=w_kv_b1_mapper,
    )
    # w_kv_b2 uses attn_proj_mapper (replicated when ATTN_DP=1).
    w_kv_b2 = _per_head_weight_tt(
        device=device,
        torch_weight=w_kv_b2_torch,
        cache_file=c("w_kv_b2", attn_proj_variant),
        dtype=dense_dtype,
        mesh_mapper=attn_proj_mapper,
    )

    # w_o stays row-parallel even when ATTN_DP=1 (it MUST have all_reduce for correctness).
    w_o = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
        cache_file=c("w_o", attn_variant),
        dtype=dense_dtype,
        mesh_mapper=attn_row_mapper,
    )
    if _env_dram_sharded_attn():
        w_q_a = _maybe_dram_shard_linear_weight(w_q_a, device)
        w_q_b = _maybe_dram_shard_linear_weight(w_q_b, device)
        w_kv_a = _maybe_dram_shard_linear_weight(w_kv_a, device)
        w_o = _maybe_dram_shard_linear_weight(w_o, device)

    # ---- MLP (dense or shared expert as dense) ----
    dense_layer = layer_idx < int(hparams.first_k_dense_replace)
    if dense_layer and not force_shared_expert_dense:
        mlp_prefix = f"model.layers.{layer_idx}.mlp."
    else:
        mlp_prefix = f"model.layers.{layer_idx}.mlp.shared_experts."

    mlp_gate_mapper = None
    mlp_down_mapper = None
    mlp_variant = ""
    if tp_enabled and tp_size > 1:
        # MLP uses Megatron-style column/row parallel:
        # - gate/up: shard output (dim=3) across TP
        # - down: shard input (dim=2) across TP and all-reduce after matmul
        gate_shape = tuple(state[f"{mlp_prefix}gate_proj.weight"].shape)
        up_shape = tuple(state[f"{mlp_prefix}up_proj.weight"].shape)
        down_shape = tuple(state[f"{mlp_prefix}down_proj.weight"].shape)
        # HF layout: [out, in]
        if int(gate_shape[0]) % int(tp_size) != 0 or int(up_shape[0]) % int(tp_size) != 0:
            raise ValueError(
                f"TP enabled but MLP out dim not divisible by tp_size={tp_size}: "
                f"gate_proj={gate_shape} up_proj={up_shape}"
            )
        if int(down_shape[1]) % int(tp_size) != 0:
            raise ValueError(
                f"TP enabled but MLP in dim not divisible by tp_size={tp_size}: down_proj={down_shape}"
            )
        mlp_variant = f"tp{tp_size}"
        mlp_gate_mapper = _tp_mesh_mapper(device, shard_dim=3)
        mlp_down_mapper = _tp_mesh_mapper(device, shard_dim=2)

    w_mlp_gate = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"{mlp_prefix}gate_proj.weight"],
        cache_file=c("w_mlp_gate", mlp_variant),
        dtype=dense_dtype,
        mesh_mapper=mlp_gate_mapper,
    )
    w_mlp_up = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"{mlp_prefix}up_proj.weight"],
        cache_file=c("w_mlp_up", mlp_variant),
        dtype=dense_dtype,
        mesh_mapper=mlp_gate_mapper,
    )
    w_mlp_down = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state[f"{mlp_prefix}down_proj.weight"],
        cache_file=c("w_mlp_down", mlp_variant),
        dtype=dense_dtype,
        mesh_mapper=mlp_down_mapper,
    )
    if _env_dram_sharded_mlp() or _env_sharded_mlp():
        w_mlp_gate = _maybe_dram_shard_linear_weight(w_mlp_gate, device, force=_env_sharded_mlp())
        w_mlp_up = _maybe_dram_shard_linear_weight(w_mlp_up, device, force=_env_sharded_mlp())
        w_mlp_down = _maybe_dram_shard_linear_weight(w_mlp_down, device, force=_env_sharded_mlp())

    # ---- Routed MoE (layers >= first_k_dense_replace) ----
    moe: Optional[MoELayerTTWeights] = None
    if enable_moe and not dense_layer:
        # Gate: hidden -> n_routed_experts
        w_gate = _linear_weight_tt(
            device=device,
            torch_weight_out_in=state[f"model.layers.{layer_idx}.mlp.gate.weight"],
            cache_file=c("w_moe_gate"),
            dtype=ttnn.bfloat16,
        )
        # IMPORTANT: `e_score_correction_bias` is float32 in the checkpoint and has very small
        # expert-to-expert deltas (~1e-3) around a large constant offset (~9). The TT `topk`
        # kernel currently requires BF16/BF8 inputs, and casting a ~9-valued float32 bias to
        # BF16 collapses those deltas (BF16 step size near 9 is ~7e-2), scrambling expert
        # ordering and destabilizing routing.
        #
        # The fix is to *center* the bias by subtracting a constant scalar offset before
        # casting to BF16. Adding/subtracting a constant from every expert does not change
        # top-k selection, but it brings values close to 0 where BF16 has much finer
        # resolution (~1e-3 near 0.1), preserving the intended ordering.
        e_bias_torch = state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"].to(dtype=torch.float32)
        e_bias_centered = e_bias_torch - float(e_bias_torch.min().item())
        e_bias = _vector_weight_tt(
            device=device,
            torch_vector=e_bias_centered,
            # Cache key bump: old BF16-cast bias caches are not compatible with centered bias.
            cache_file=c("e_score_correction_bias_centered_v1"),
            dtype=ttnn.bfloat16,
        )

        experts_dtype = _env_experts_dtype()
        num_experts = int(hparams.n_routed_experts)
        moe_intermediate = int(hparams.moe_intermediate_size)
        hidden = int(hparams.hidden_size)
        num_devices = int(device.get_num_devices()) if _is_mesh_device(device) else 1
        experts_variant = f"localE_d{num_devices}_v1"

        # Stack experts: [E, in, out] with TT linear conventions.
        # gate/up: HF is [out, in] == [moe_intermediate, hidden] -> transpose to [hidden, moe_intermediate]
        # down: HF is [out, in] == [hidden, moe_intermediate] -> transpose to [moe_intermediate, hidden]
        w1_list: list[torch.Tensor] = []
        w3_list: list[torch.Tensor] = []
        w2_list: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            w1 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"]
            w3 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"]
            w2 = state[f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"]
            if tuple(w1.shape) != (moe_intermediate, hidden):
                raise ValueError(
                    f"Unexpected gate_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w1.shape)}"
                )
            if tuple(w3.shape) != (moe_intermediate, hidden):
                raise ValueError(f"Unexpected up_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w3.shape)}")
            if tuple(w2.shape) != (hidden, moe_intermediate):
                raise ValueError(
                    f"Unexpected down_proj shape for layer{layer_idx} expert{expert_id}: {tuple(w2.shape)}"
                )
            w1_list.append(w1.transpose(-2, -1).contiguous())
            w3_list.append(w3.transpose(-2, -1).contiguous())
            w2_list.append(w2.transpose(-2, -1).contiguous())

        w1_stacked = torch.stack(w1_list, dim=0)  # [E, hidden, moe_intermediate]
        w3_stacked = torch.stack(w3_list, dim=0)  # [E, hidden, moe_intermediate]
        w2_stacked = torch.stack(w2_list, dim=0)  # [E, moe_intermediate, hidden]

        w1_experts = _experts_weight_tt(
            device=device,
            torch_weights=w1_stacked,
            # Cache key must include mesh size because expert weights are sharded.
            cache_file=c("w1_experts", experts_variant),
            dtype=experts_dtype,
        )
        w3_experts = _experts_weight_tt(
            device=device,
            torch_weights=w3_stacked,
            cache_file=c("w3_experts", experts_variant),
            dtype=experts_dtype,
        )
        w2_experts = _experts_weight_tt(
            device=device,
            torch_weights=w2_stacked,
            cache_file=c("w2_experts", experts_variant),
            dtype=experts_dtype,
        )

        # Optional fused gate+up (w1+w3) tensor for single sparse_matmul.
        w1w3_experts_tt: Optional[ttnn.Tensor] = None
        fuse_gate_up = os.environ.get("GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP", "").strip() == "1"
        if fuse_gate_up:
            w1w3_stacked = torch.cat([w1_stacked, w3_stacked], dim=2)  # [E, hidden, 2*moe_intermediate]
            w1w3_experts_tt = _experts_weight_tt(
                device=device,
                torch_weights=w1w3_stacked,
                cache_file=c("w1w3_experts", experts_variant),
                dtype=experts_dtype,
            )

        moe = MoELayerTTWeights(
            w_gate=w_gate,
            e_score_correction_bias=e_bias,
            w1_experts=w1_experts,
            w2_experts=w2_experts,
            w3_experts=w3_experts,
            w1w3_experts=w1w3_experts_tt,
        )

    return DecoderLayerTTWeights(
        layer_idx=layer_idx,
        input_layernorm=input_layernorm,
        q_a_layernorm=q_a_layernorm,
        kv_a_layernorm=kv_a_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        w_q_a=w_q_a,
        w_q_b=w_q_b,
        w_kv_a=w_kv_a,
        w_q_kv_a=w_q_kv_a,
        w_kv_b1=w_kv_b1,
        w_kv_b2=w_kv_b2,
        w_o=w_o,
        w_mlp_gate=w_mlp_gate,
        w_mlp_up=w_mlp_up,
        w_mlp_down=w_mlp_down,
        moe=moe,
    )
