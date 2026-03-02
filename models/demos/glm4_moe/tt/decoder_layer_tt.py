# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standard pre-norm transformer decoder layer for GLM-4.7-REAP-218B.

Layers 0-2: dense MLP (intermediate_size=12288)
Layers 3-91: MoE (96 routed experts EP=32 + 1 shared expert TP=8)

Attention uses standard GQA (96Q/8KV heads, NOT MLA).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

import ttnn

from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.layer_weights import DecoderLayerTTWeights
from models.demos.glm4_moe.tt.moe_tt import (
    Glm4MoeMoERuntime,
    moe_sparse_experts_forward_tt,
    moe_topk_tt,
    shared_expert_forward_tt,
    _env_bool,
    _get_mesh_shape,
    _moe_sparse_tokens_multiple,
    _parse_math_fidelity,
)
from models.demos.glm4_moe.tt.attention_tt import Glm4MoeAttention, _simple_all_reduce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tp_cluster_axis(device: Any) -> int | None:
    """Return the mesh axis used for TP sharding.

    Galaxy TG Mesh(8,4): TP=axis 0 (rows=8).
    T3K Mesh(1,8): TP=axis 1 (cols=8).
    """
    if device.__class__.__name__ != "MeshDevice":
        return None
    mesh_rows, mesh_cols = _get_mesh_shape(device)
    if mesh_rows > 1 and mesh_cols > 1:
        # 2D mesh (TG): TP is axis 0 (rows=8)
        return 0
    if mesh_cols > 1:
        return 1
    if mesh_rows > 1:
        return 0
    return None


def _sharded_rms_norm(x: ttnn.Tensor, norm_module: Any, hidden_size: int, num_cores: int = 8) -> ttnn.Tensor:
    """Width-sharded multi-core RMSNorm to avoid L1 overflow for large hidden sizes.

    For hidden_size=5120, single-core RMSNorm overflows L1 by ~13.6 KB (1.51 MB vs 1.43 MB limit).
    Spreading across 8 cores reduces per-core memory to ~100 KB — well within L1.
    """
    h_logical = int(x.shape[-2])  # logical height (may NOT be tile-padded)
    h = ((h_logical + 31) // 32) * 32  # round up to tile boundary (32)
    tile_h = h // 32  # number of tile rows
    tiles_w = hidden_size // 32  # total tiles in width
    tiles_per_core = tiles_w // num_cores  # tiles per core

    # Subblock width: largest value <= 4 that evenly divides tiles_per_core
    subblock_w = 4
    while subblock_w > 0:
        if tiles_per_core % subblock_w == 0:
            break
        subblock_w -= 1

    # Width-sharded memory config: each core gets [h, hidden/num_cores] elements
    shard_w = hidden_size // num_cores
    core_range = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))
    ])
    shard_spec = ttnn.ShardSpec(core_range, [h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
    )

    # Matching program config
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores, 1],
        subblock_w=subblock_w,
        block_h=tile_h,
        block_w=tiles_per_core,
        inplace=False,
    )

    # Shard input from DRAM interleaved to L1 width-sharded
    x_sharded = ttnn.interleaved_to_sharded(x, sharded_mem_cfg)

    # Run sharded RMSNorm
    result = ttnn.rms_norm(
        x_sharded,
        epsilon=norm_module.eps,
        weight=norm_module.weight,
        program_config=program_config,
        compute_kernel_config=norm_module.compute_kernel_config_hifi2,
    )
    ttnn.deallocate(x_sharded, force=False)

    # Convert back to interleaved DRAM
    result = ttnn.sharded_to_interleaved(result, ttnn.DRAM_MEMORY_CONFIG)
    return result


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Glm4MoeDecoderLayer:
    """Single decoder layer for GLM-4.7-REAP-218B.

    Pre-norm architecture:
    1. RMSNorm -> GQA Attention -> residual
    2. RMSNorm -> MLP (dense or MoE) -> residual
    """

    def __init__(
        self,
        *,
        mesh_device: Any,
        tt_ccl: Any | None,
        hparams: Glm4MoeHParams,
        layer_weights: DecoderLayerTTWeights,
        configuration: Any | None = None,
        paged_attention_config: Any | None = None,
        moe_runtime: Glm4MoeMoERuntime | None = None,
    ):
        self.device = mesh_device
        self.tt_ccl = tt_ccl
        self.hparams = hparams
        self.layer_weights = layer_weights
        self.configuration = configuration
        self.paged_attention_config = paged_attention_config
        self.moe_runtime = moe_runtime
        self.layer_idx = int(layer_weights.layer_idx)

        self.is_dense_layer = self.layer_idx < int(hparams.first_k_dense_replace)
        self.has_moe = not self.is_dense_layer and layer_weights.moe is not None

        # Attention module (GQA).
        self.attention = Glm4MoeAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            layer_weights=layer_weights,
            hparams=hparams,
            configuration=configuration,
            paged_attention_config=paged_attention_config,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        mode: str = "decode",
    ) -> ttnn.Tensor:
        """Forward pass for one decoder layer.

        Args:
            x: Hidden states [1,1,B,hidden] (decode) or [1,1,S,hidden] (prefill).
            current_pos: Position indices for KV cache update.
            rot_mats: RoPE rotation matrices (cos, sin, trans_mat or tuple).
            page_table: Paged KV cache page table.
            kv_cache: [cache_k, cache_v] for this layer.
            mode: "decode" or "prefill".

        Returns:
            Updated hidden states, same shape as input.
        """
        if mode == "decode":
            return self._forward_decode(x, current_pos, rot_mats, page_table, kv_cache)
        else:
            return self._forward_prefill(x, current_pos, rot_mats, page_table, kv_cache)

    def _forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Decode forward: batch in dim=2, seq_len=1 per token."""
        w = self.layer_weights
        device = self.device
        hparams = self.hparams

        tp_axis = _tp_cluster_axis(device)
        tp_enabled = tp_axis is not None

        # Compute kernel config for MLP.
        mlp_fidelity = _parse_math_fidelity(
            os.environ.get("GLM4_MOE_MLP_FIDELITY", ""),
            default=ttnn.MathFidelity.LoFi,
        )
        mlp_approx = os.environ.get("GLM4_MOE_MLP_APPROX", "1").strip() != "0"
        _fp32_acc_decode = os.environ.get("GLM4_MOE_FP32_ACC", "0").strip() != "0"
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=mlp_fidelity,
            math_approx_mode=mlp_approx,
            fp32_dest_acc_en=_fp32_acc_decode,
            packer_l1_acc=True,
        )

        # ---- Pre-attention norm (sharded to avoid L1 overflow with hidden=5120) ----
        h = _sharded_rms_norm(x, w.input_layernorm, hparams.hidden_size)

        # ---- GQA Attention ----
        attn_out = self.attention.forward(
            h, current_pos, rot_mats, mode="decode", page_table=page_table, kv_cache=kv_cache,
        )

        # ---- Residual ----
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, force=False)

        # ---- Pre-MLP norm (sharded to avoid L1 overflow with hidden=5120) ----
        residual = x
        h = _sharded_rms_norm(x, w.post_attention_layernorm, hparams.hidden_size)

        # ---- MLP ----
        if self.is_dense_layer:
            mlp_out = self._dense_mlp_forward(
                h, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down,
                compute_kernel_config=mlp_compute_kernel_config,
            )
            ttnn.deallocate(h, force=False)
            # TP reduce for dense MLP (host-side — ttnn.all_reduce broken on TG 2D mesh).
            if tp_enabled:
                mlp_out = _simple_all_reduce(
                    mlp_out, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl,
                )
        else:
            mlp_out = self._moe_forward(
                h, compute_kernel_config=mlp_compute_kernel_config,
                tp_axis=tp_axis, tp_enabled=tp_enabled,
            )

        # ---- Residual ----
        res_shape = [int(d) for d in residual.shape]
        mlp_shape = [int(d) for d in mlp_out.shape]
        if mlp_shape != res_shape:
            res_vol = 1
            mlp_vol = 1
            for d in res_shape:
                res_vol *= d
            for d in mlp_shape:
                mlp_vol *= d
            if mlp_vol == res_vol:
                mlp_out = ttnn.reshape(mlp_out, res_shape, mlp_shape)
            else:
                mlp_out = ttnn.slice(mlp_out, starts=[0] * len(res_shape), ends=res_shape)

        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out, force=False)
        ttnn.deallocate(residual, force=False)
        return x

    def _forward_prefill(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: Any,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Prefill forward: batch=1, seq_len in dim=2."""
        w = self.layer_weights
        device = self.device
        hparams = self.hparams

        tp_axis = _tp_cluster_axis(device)
        tp_enabled = tp_axis is not None

        # FP32 dest accumulation reduces BF8 rounding error through 92 layers
        _fp32_acc = os.environ.get("GLM4_MOE_FP32_ACC", "0").strip() != "0"
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=_fp32_acc,
            packer_l1_acc=True,
        )

        # ---- Pre-attention norm (sharded to avoid L1 overflow with hidden=5120) ----
        h = _sharded_rms_norm(x, w.input_layernorm, hparams.hidden_size)

        # ---- GQA Attention ----
        attn_out = self.attention.forward(
            h, current_pos, rot_mats, mode="prefill", page_table=page_table, kv_cache=kv_cache,
        )

        # ---- Residual ----
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, force=False)

        # ---- Pre-MLP norm (sharded to avoid L1 overflow with hidden=5120) ----
        residual = x
        h = _sharded_rms_norm(x, w.post_attention_layernorm, hparams.hidden_size)

        # ---- MLP ----
        if self.is_dense_layer:
            mlp_out = self._dense_mlp_forward(
                h, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down,
                compute_kernel_config=mlp_compute_kernel_config,
            )
            ttnn.deallocate(h, force=False)
            if tp_enabled:
                mlp_out = _simple_all_reduce(
                    mlp_out, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl,
                )
        else:
            mlp_out = self._moe_forward(
                h, compute_kernel_config=mlp_compute_kernel_config,
                tp_axis=tp_axis, tp_enabled=tp_enabled,
            )

        # ---- Residual (prefill path) ----
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out, force=False)
        ttnn.deallocate(residual, force=False)
        return x

    # -----------------------------------------------------------------------
    # Dense MLP (layers 0-2): gate_up_SiLU_down, TP=8 column/row parallel
    # -----------------------------------------------------------------------

    def _dense_mlp_forward(
        self,
        x: ttnn.Tensor,
        w_gate: ttnn.Tensor,
        w_up: ttnn.Tensor,
        w_down: ttnn.Tensor,
        compute_kernel_config: Any | None = None,
    ) -> ttnn.Tensor:
        """Dense MLP: gate_proj * SiLU(up_proj) -> down_proj.

        gate/up are column-parallel (sharded output dim by TP=8).
        down is row-parallel (sharded input dim by TP=8, needs all_reduce).
        """
        kwargs: dict[str, Any] = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config

        gate = ttnn.linear(x, w_gate, **kwargs)
        up = ttnn.linear(x, w_up, **kwargs)
        x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate, force=False)
        ttnn.deallocate(up, force=False)
        out = ttnn.linear(x_ff, w_down, **kwargs)
        ttnn.deallocate(x_ff, force=False)
        return out

    # -----------------------------------------------------------------------
    # MoE (layers 3-91): routing + shared expert + routed experts
    # -----------------------------------------------------------------------

    def _moe_forward(
        self,
        x: ttnn.Tensor,
        compute_kernel_config: Any | None = None,
        tp_axis: int | None = None,
        tp_enabled: bool = False,
    ) -> ttnn.Tensor:
        """MoE forward: shared expert + routed experts -> sum.

        Shared expert: TP=8 gate-up-SiLU-down MLP.
        Routed experts: top-8 routing + sparse_matmul across EP=32.
        """
        w = self.layer_weights
        device = self.device
        hparams = self.hparams
        moe_runtime = self.moe_runtime
        moe_w = w.moe

        tokens = int(x.shape[2])

        # Pad tokens for sparse expert alignment.
        pad_tokens = 0
        if moe_runtime is not None:
            sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
            pad_tokens = (-tokens) % sparse_multiple
            if pad_tokens:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)

        moe_decode_mc = getattr(moe_runtime, "decode_memory_config", ttnn.DRAM_MEMORY_CONFIG) if moe_runtime else ttnn.DRAM_MEMORY_CONFIG

        # ---- Shared expert (dense MLP, runs on all tokens) ----
        shared_out_partial = shared_expert_forward_tt(
            x=x,
            w_gate=w.w_mlp_gate,
            w_up=w.w_mlp_up,
            w_down=w.w_mlp_down,
            w_gate_up=w.w_mlp_gate_up,
            memory_config=moe_decode_mc,
            compute_kernel_config=compute_kernel_config,
        )
        # NOTE: No TP reduce here — fused with EP reduce below

        # ---- Routed experts (skip EP reduce — will fuse with shared TP reduce) ----
        topk_weights, topk_indices = moe_topk_tt(
            x=x,
            moe_w=moe_w,
            hparams=hparams,
            compute_kernel_config=compute_kernel_config,
        )

        routed_out_partial = moe_sparse_experts_forward_tt(
            device=device,
            hidden_states=x,
            topk_expert_indices=topk_indices,
            topk_expert_weights=topk_weights,
            moe_w=moe_w,
            rt=moe_runtime,
            memory_config=moe_decode_mc,
            skip_final_reduce=True,
        )

        # ---- Fused reduce: scale shared by 1/DP, combine, single reduce ----
        fuse_reduce = _env_bool("GLM4_MOE_FUSE_SHARED_EP_REDUCE", default=True)
        if fuse_reduce and tp_enabled and tp_axis is not None:
            mesh_shape = _get_mesh_shape(device)
            num_dp = mesh_shape[1]  # DP = cols = 4 for Galaxy TG
            if num_dp > 1:
                shared_out_partial = ttnn.mul(shared_out_partial, 1.0 / num_dp,
                                              memory_config=moe_decode_mc)

            combined = ttnn.add(shared_out_partial, routed_out_partial,
                                memory_config=moe_decode_mc)
            ttnn.deallocate(shared_out_partial, force=False)
            ttnn.deallocate(routed_out_partial, force=False)

            # 2-step all_reduce: axis=0 (TP=8) then axis=1 (DP=4) = full 32-way
            mlp_out = _simple_all_reduce(combined, device, cluster_axis=0,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                         ccl=self.tt_ccl)
            ttnn.deallocate(combined, force=False)
            mlp_out = _simple_all_reduce(mlp_out, device, cluster_axis=1,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                         ccl=self.tt_ccl)
        else:
            # Fallback: separate reduces (original behavior)
            if tp_enabled and tp_axis is not None:
                shared_out_partial = _simple_all_reduce(
                    shared_out_partial, device, cluster_axis=tp_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ccl=self.tt_ccl,
                )
            # EP reduce for routed experts
            if device.__class__.__name__ == "MeshDevice":
                num_devices = device.get_num_devices()
                if num_devices > 1:
                    routed_out_partial = _simple_all_reduce(
                        routed_out_partial, device, cluster_axis=0,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, ccl=self.tt_ccl)
                    routed_out_partial = _simple_all_reduce(
                        routed_out_partial, device, cluster_axis=1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, ccl=self.tt_ccl)
            mlp_out = ttnn.add(shared_out_partial, routed_out_partial,
                               memory_config=moe_decode_mc)
            ttnn.deallocate(shared_out_partial, force=False)
            ttnn.deallocate(routed_out_partial, force=False)

        # Slice back to real token count if padded.
        if pad_tokens:
            mlp_out = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])

        return mlp_out
