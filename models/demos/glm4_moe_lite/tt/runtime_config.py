# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Centralized runtime configuration for GLM-4.7-Flash.

All GLM4_MOE_LITE_* environment variables are parsed once into a frozen
dataclass at model init time. No other module should read os.environ for
GLM4_MOE_LITE_* knobs directly.

For new model bring-ups: copy this file, rename the dataclass, and adjust
the env var names and defaults for the new model's knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import ttnn


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _env_str(name: str, *, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_math_fidelity(value: str, *, default: ttnn.MathFidelity) -> ttnn.MathFidelity:
    raw = value.strip().lower()
    if not raw:
        return default
    table = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    return table.get(raw, default)


def mesh_shape(device: Any) -> tuple[int, int]:
    if device.__class__.__name__ != "MeshDevice":
        return (1, 1)
    return (int(device.shape[0]), int(device.shape[1]))


def tp_cluster_axis(device: Any) -> int | None:
    """Return the mesh axis used for TP-style sharding (preferred: cols)."""
    if device.__class__.__name__ != "MeshDevice":
        return None
    mesh_rows, mesh_cols = mesh_shape(device)
    if mesh_cols > 1:
        return 1
    if mesh_rows > 1:
        return 0
    return None


@dataclass(frozen=True)
class Glm4RuntimeConfig:
    """All GLM4_MOE_LITE_* runtime knobs, parsed once from env vars.

    Pass this to decoder layer functions instead of having them read
    os.environ on every call. Immutable after creation.
    """

    # --- Precision ---
    moe_fp32_acc: bool
    mlp_fidelity: ttnn.MathFidelity
    mlp_approx: bool
    mla_fidelity: ttnn.MathFidelity
    mla_approx: bool
    mla_fp32_acc: bool
    mla_scale_mode: str
    mla_k_chunk_size: int
    packer_l1_acc: bool
    skip_typecast: bool

    # --- Memory layout ---
    decode_l1_act: bool
    dram_sharded_weights: bool
    dram_sharded_attn: bool
    dram_sharded_mlp: bool
    sharded_mlp: bool

    # --- Matmul config ---
    explicit_prog_cfg: bool

    # --- Attention ---
    concat_heads: bool
    attn_dp: bool
    head_parallel_kvb2: bool
    use_v_cache_slice: bool
    shard_q: bool

    # --- MLP / MoE ---
    fuse_mlp_moe_reduce: bool
    fuse_shared_gate_up: bool
    moe_experts_impl: str
    moe_router_impl: str
    moe_dense_prefill: bool
    moe_packed_prefill: bool

    # --- Defensive copies ---
    skip_defensive_clones: bool

    # --- TP ---
    tp_enabled: bool
    tp_axis: int | None
    tp_size: int

    # --- Debug ---
    layer_identity: bool
    skip_kv_update: bool
    disable_mlp: bool
    disable_flash_mla_decode: bool
    sync_after_kv_update: bool

    @classmethod
    def from_env(cls, *, device: Any) -> "Glm4RuntimeConfig":
        """Parse all GLM4_MOE_LITE_* env vars once. Call at model init."""
        tp_ax = tp_cluster_axis(device)
        tp_on = tp_ax is not None and _env_bool("GLM4_MOE_LITE_TP")
        _, mesh_cols = mesh_shape(device)
        mesh_rows, _ = mesh_shape(device)
        tp_sz = int((mesh_rows, mesh_cols)[tp_ax]) if tp_ax is not None else 1

        dram_sharded = _env_bool("GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS")
        sharded_mlp_standalone = _env_bool("GLM4_MOE_LITE_SHARDED_MLP")
        dram_sharded_mlp_val = (
            dram_sharded and _env_str("GLM4_MOE_LITE_DRAM_SHARDED_MLP", default="1") != "0"
        ) or sharded_mlp_standalone

        mla_fp32_req = _env_bool("GLM4_MOE_LITE_MLA_FP32_ACC")
        mla_fp32 = mla_fp32_req
        if mla_fp32_req and not _env_bool("GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA"):
            mla_fp32 = False

        return cls(
            # Precision
            moe_fp32_acc=_env_bool("GLM4_MOE_LITE_MOE_FP32_ACC"),
            mlp_fidelity=parse_math_fidelity(_env_str("GLM4_MOE_LITE_MLP_FIDELITY"), default=ttnn.MathFidelity.LoFi),
            mlp_approx=_env_str("GLM4_MOE_LITE_MLP_APPROX", default="1") != "0",
            mla_fidelity=parse_math_fidelity(_env_str("GLM4_MOE_LITE_MLA_FIDELITY"), default=ttnn.MathFidelity.HiFi4),
            mla_approx=_env_str("GLM4_MOE_LITE_MLA_APPROX", default="0") != "0",
            mla_fp32_acc=mla_fp32,
            mla_scale_mode=_env_str("GLM4_MOE_LITE_MLA_SCALE_MODE", default="qk").lower(),
            mla_k_chunk_size=_env_int("GLM4_MOE_LITE_MLA_K_CHUNK_SIZE", default=64),
            packer_l1_acc=_env_bool("GLM4_MOE_LITE_PACKER_L1_ACC"),
            skip_typecast=_env_bool("GLM4_MOE_LITE_SKIP_TYPECAST"),
            # Memory layout
            decode_l1_act=_env_bool("GLM4_MOE_LITE_DECODE_L1_ACT"),
            dram_sharded_weights=dram_sharded,
            dram_sharded_attn=dram_sharded and _env_bool("GLM4_MOE_LITE_DRAM_SHARDED_ATTN"),
            dram_sharded_mlp=dram_sharded_mlp_val,
            sharded_mlp=sharded_mlp_standalone,
            # Matmul config
            explicit_prog_cfg=_env_bool("GLM4_MOE_LITE_EXPLICIT_PROG_CFG"),
            # Attention
            concat_heads=_env_bool("GLM4_MOE_LITE_CONCAT_HEADS"),
            attn_dp=_env_bool("GLM4_MOE_LITE_ATTN_DP"),
            head_parallel_kvb2=(_env_bool("GLM4_MOE_LITE_HEAD_PARALLEL_KVB2") and tp_on and tp_sz > 1),
            use_v_cache_slice=_env_bool("GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE"),
            shard_q=_env_bool("GLM4_MOE_LITE_MLA_SHARD_Q"),
            # MLP / MoE
            fuse_mlp_moe_reduce=_env_bool("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE"),
            fuse_shared_gate_up=_env_bool("GLM4_MOE_LITE_FUSE_SHARED_GATE_UP"),
            moe_experts_impl=_env_str("GLM4_MOE_LITE_MOE_EXPERTS_IMPL", default="sparse").lower(),
            moe_router_impl=_env_str("GLM4_MOE_LITE_MOE_ROUTER_IMPL", default="tt").lower(),
            moe_dense_prefill=_env_bool("GLM4_MOE_LITE_MOE_DENSE_PREFILL"),
            moe_packed_prefill=_env_bool("GLM4_MOE_LITE_MOE_PACKED_PREFILL"),
            # Defensive copies
            skip_defensive_clones=_env_bool("GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES"),
            # TP
            tp_enabled=tp_on,
            tp_axis=tp_ax,
            tp_size=tp_sz,
            # Debug
            layer_identity=_env_bool("GLM4_MOE_LITE_LAYER_IDENTITY"),
            skip_kv_update=_env_bool("GLM4_MOE_LITE_SKIP_KV_UPDATE"),
            disable_mlp=_env_bool("GLM4_MOE_LITE_DISABLE_MLP"),
            disable_flash_mla_decode=_env_bool("GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE"),
            sync_after_kv_update=_env_bool("GLM4_MOE_LITE_SYNC_AFTER_KV_UPDATE"),
        )

    @property
    def decode_act_mc(self) -> ttnn.MemoryConfig | None:
        """L1 memory config for decode activations, or None for DRAM default."""
        return ttnn.L1_MEMORY_CONFIG if self.decode_l1_act else None

    def mlp_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """Compute kernel config for MLP/router matmuls."""
        if self.moe_fp32_acc:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.mlp_fidelity,
            math_approx_mode=self.mlp_approx,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def mla_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """Compute kernel config for FlashMLA attention."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.mla_fidelity,
            math_approx_mode=self.mla_approx,
            fp32_dest_acc_en=self.mla_fp32_acc,
            packer_l1_acc=self.packer_l1_acc,
        )
