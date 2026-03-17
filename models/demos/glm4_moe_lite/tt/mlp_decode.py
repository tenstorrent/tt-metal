# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MLP decode path for GLM-4.7-Flash (dense SwiGLU and MoE).

Extracted from decoder_layer_tt.py lines 1326-1576. Two entry points:
  1. dense_mlp_forward — SwiGLU for dense layers (layer 0)
  2. moe_mlp_forward — shared expert + routed experts for MoE layers
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import ttnn
from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.linear_helpers import _DS_BATCH, dram_sharded_mlp, mlp_linear
from models.demos.glm4_moe_lite.tt.runtime_config import Glm4RuntimeConfig, mesh_shape

_SIGNPOST_ENABLED = os.environ.get("GLM4_MOE_LITE_SIGNPOST", "").strip() == "1"
if _SIGNPOST_ENABLED:
    from tracy import signpost


def _profile_add(profile: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(elapsed_s)


def _moe_sparse_tokens_multiple(*, device: Any, moe_runtime: Any) -> int:
    """Minimum per-device token multiple required by sparse MoE."""
    block = max(1, int(getattr(moe_runtime, "sparsity_block_size", 32)))
    dispatch_axis = int(getattr(moe_runtime, "dispatch_cluster_axis", 0))
    mesh_rows, mesh_cols = mesh_shape(device)
    dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_axis])
    dispatch_devices = max(1, dispatch_devices)
    return max(1, block // math.gcd(block, dispatch_devices))


def _run_dram_sharded_swiglu(
    x: ttnn.Tensor,
    w_gate: ttnn.Tensor,
    w_up: ttnn.Tensor,
    w_down: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
) -> ttnn.Tensor:
    """DRAM-sharded SwiGLU with padding to _DS_BATCH if needed."""
    tokens = int(x.shape[2])
    if tokens == _DS_BATCH:
        return dram_sharded_mlp(x, w_gate, w_up, w_down, device=device, cfg=cfg)
    pad = _DS_BATCH - tokens
    x_padded = ttnn.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
    out_padded = dram_sharded_mlp(x_padded, w_gate, w_up, w_down, device=device, cfg=cfg)
    ttnn.deallocate(x_padded, force=False)
    return out_padded[:, :, :tokens, :]


def _run_standard_swiglu(
    x: ttnn.Tensor,
    w: Any,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    fuse_gate_up: bool,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Standard (non-DRAM-sharded) SwiGLU: gate * silu -> up -> down."""
    if fuse_gate_up and getattr(w, "w_mlp_gate_up", None) is not None:
        gate_up = mlp_linear(x, w.w_mlp_gate_up, device=device, cfg=cfg)
        _batch = int(gate_up.shape[2])
        _inter_tp = int(gate_up.shape[3]) // 2
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, _batch, _inter_tp])
        up = ttnn.slice(gate_up, [0, 0, 0, _inter_tp], [1, 1, _batch, _inter_tp * 2])
        ttnn.deallocate(gate_up, force=False)
    else:
        gate = mlp_linear(x, w.w_mlp_gate, device=device, cfg=cfg)
        up = mlp_linear(x, w.w_mlp_up, device=device, cfg=cfg)

    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate, force=False)
    ttnn.deallocate(up, force=False)
    out = mlp_linear(x_ff, w.w_mlp_down, device=device, cfg=cfg, memory_config=memory_config)
    ttnn.deallocate(x_ff, force=False)
    return out


def dense_mlp_forward(
    x: ttnn.Tensor,
    w: Any,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor:
    """Dense SwiGLU MLP for non-MoE layers. Returns mlp_out [1,1,B,hidden]."""
    t0 = time.perf_counter() if profile is not None else 0.0

    if cfg.dram_sharded_mlp:
        mlp_out = _run_dram_sharded_swiglu(x, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down, device=device, cfg=cfg)
        ttnn.deallocate(x, force=False)
    else:
        mlp_out = _run_standard_swiglu(x, w, device=device, cfg=cfg, fuse_gate_up=False)
        ttnn.deallocate(x, force=False)

    if cfg.tp_enabled:
        mlp_out_reduced = ttnn.all_reduce(
            mlp_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=cfg.tp_axis,
            memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mlp_out, force=False)
        mlp_out = mlp_out_reduced

    _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
    return mlp_out


def moe_mlp_forward(
    x: ttnn.Tensor,
    w: Any,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    hparams: Glm4MoeLiteHParams,
    moe_runtime: Any,
    profile: dict[str, float] | None = None,
    layer_idx: int = -1,
    use_signpost: bool = False,
) -> ttnn.Tensor:
    """MoE MLP: shared expert (dense) + routed experts. Returns mlp_out [1,1,B,hidden]."""
    from models.demos.glm4_moe_lite.tt.moe_tt import (
        moe_dense_experts_forward_decode_tt,
        moe_dense_experts_forward_prefill_tt,
        moe_packed_experts_forward_prefill_tt,
        moe_sparse_experts_forward_tt,
        moe_topk_cpu_reference,
        moe_topk_tt,
    )

    tokens = int(x.shape[2])
    use_dense_decode = cfg.moe_experts_impl in {"dense_decode", "dense-decode"} and tokens == 1
    _PACKED_PREFILL_MIN_TOKENS = 33
    use_packed_prefill = cfg.moe_packed_prefill and tokens >= _PACKED_PREFILL_MIN_TOKENS
    use_dense_prefill = cfg.moe_dense_prefill and not use_packed_prefill and tokens >= 33
    moe_decode_mc = getattr(moe_runtime, "decode_memory_config", ttnn.DRAM_MEMORY_CONFIG)
    mlp_compute_kernel_config = cfg.mlp_compute_kernel_config()
    _skip_shared_reduce = cfg.fuse_mlp_moe_reduce and cfg.tp_enabled

    # Pad tokens for sparse expert kernels if needed
    pad_tokens = 0
    if not use_dense_decode and not use_dense_prefill and not use_packed_prefill:
        sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
        pad_tokens = (-tokens) % sparse_multiple
        if pad_tokens:
            if cfg.skip_defensive_clones:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            else:
                x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
                x = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # --- Shared expert (dense SwiGLU) ---
    if use_signpost:
        signpost(f"L{layer_idx}_shared_expert-start")
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.dram_sharded_mlp:
        shared_out = _run_dram_sharded_swiglu(x, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down, device=device, cfg=cfg)
    else:
        shared_out = _run_standard_swiglu(
            x,
            w,
            device=device,
            cfg=cfg,
            fuse_gate_up=cfg.fuse_shared_gate_up,
            memory_config=moe_decode_mc,
        )
    if cfg.tp_enabled and not _skip_shared_reduce:
        shared_out_reduced = ttnn.all_reduce(
            shared_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=cfg.tp_axis,
            memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(shared_out, force=False)
        shared_out = shared_out_reduced
    _profile_add(profile, "moe_shared_s", time.perf_counter() - t0 if profile is not None else 0.0)
    if use_signpost:
        signpost(f"L{layer_idx}_shared_expert-end")

    # --- Routing ---
    if use_signpost:
        signpost(f"L{layer_idx}_router-start")
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.moe_router_impl == "cpu":
        topk_weights, topk_indices = moe_topk_cpu_reference(device=device, x=x, moe_w=w.moe, hparams=hparams)
    else:
        topk_weights, topk_indices = moe_topk_tt(
            x=x,
            moe_w=w.moe,
            hparams=hparams,
            compute_kernel_config=mlp_compute_kernel_config,
        )
    _profile_add(profile, "moe_router_s", time.perf_counter() - t0 if profile is not None else 0.0)
    if use_signpost:
        signpost(f"L{layer_idx}_router-end")

    # --- Routed experts ---
    if use_signpost:
        signpost(f"L{layer_idx}_routed_experts-start")
    t0 = time.perf_counter() if profile is not None else 0.0
    expert_kwargs = dict(
        device=device,
        hidden_states=x,
        topk_expert_indices=topk_indices,
        topk_expert_weights=topk_weights,
        moe_w=w.moe,
        memory_config=moe_decode_mc,
    )
    if use_dense_decode:
        if int(x.shape[2]) > 1:
            routed_out = moe_dense_experts_forward_prefill_tt(
                **expert_kwargs,
                rt=moe_runtime,
                compute_kernel_config=mlp_compute_kernel_config,
                skip_defensive_clones=cfg.skip_defensive_clones,
            )
        else:
            routed_out = moe_dense_experts_forward_decode_tt(
                **expert_kwargs,
                hparams=hparams,
                compute_kernel_config=mlp_compute_kernel_config,
                skip_defensive_clones=cfg.skip_defensive_clones,
            )
    elif use_packed_prefill:
        routed_out = moe_packed_experts_forward_prefill_tt(
            **expert_kwargs,
            hparams=hparams,
            compute_kernel_config=mlp_compute_kernel_config,
            skip_defensive_clones=cfg.skip_defensive_clones,
        )
    elif use_dense_prefill:
        routed_out = moe_dense_experts_forward_prefill_tt(
            **expert_kwargs,
            hparams=hparams,
            compute_kernel_config=mlp_compute_kernel_config,
            skip_defensive_clones=cfg.skip_defensive_clones,
        )
    else:
        routed_out = moe_sparse_experts_forward_tt(
            **expert_kwargs,
            rt=moe_runtime,
            skip_defensive_clones=cfg.skip_defensive_clones,
            skip_final_reduce=_skip_shared_reduce,
        )
    _profile_add(profile, "moe_experts_s", time.perf_counter() - t0 if profile is not None else 0.0)
    if use_signpost:
        signpost(f"L{layer_idx}_routed_experts-end")

    # --- Merge shared + routed ---
    if use_signpost:
        signpost(f"L{layer_idx}_merge-start")
    t0 = time.perf_counter() if profile is not None else 0.0
    mlp_out = ttnn.add(shared_out, routed_out, memory_config=moe_decode_mc)
    ttnn.deallocate(shared_out, force=False)
    ttnn.deallocate(routed_out, force=False)

    if _skip_shared_reduce:
        mlp_out_reduced = ttnn.all_reduce(
            mlp_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=cfg.tp_axis,
            memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mlp_out, force=False)
        mlp_out = mlp_out_reduced

    # Unpad if we padded for sparse kernels
    if pad_tokens:
        if cfg.skip_defensive_clones:
            mlp_out = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
        else:
            mlp_out_view = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
            mlp_out = ttnn.clone(mlp_out_view, memory_config=moe_decode_mc)

    _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)
    if use_signpost:
        signpost(f"L{layer_idx}_merge-end")
    return mlp_out
