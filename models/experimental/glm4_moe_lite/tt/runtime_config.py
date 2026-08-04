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
from loguru import logger


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


# ---------------------------------------------------------------------------
# Cluster / architecture capabilities
#
# These exist because the model's hardware assumptions were WH-Galaxy literals. Blackhole
# Galaxy is a *different ClusterType* (BLACKHOLE_GALAXY, not GALAXY), so every
# `get_cluster_type() == GALAXY` test silently took the non-Galaxy branch on it.
# ---------------------------------------------------------------------------


def _cluster_type() -> Any:
    """Current ClusterType, or None when the cluster cannot be queried."""
    try:
        return ttnn.cluster.get_cluster_type()
    except Exception:  # pragma: no cover - no driver / no devices
        return None


def is_ubb_galaxy() -> bool:
    """True on a 32-chip Galaxy of either architecture.

    Mirrors `Cluster::is_ubb_galaxy` (tt_metal/llrt/tt_cluster.hpp:323-327), which covers
    GALAXY and BLACKHOLE_GALAXY, plus legacy TG so this stays a superset of the check it
    replaces. Use this instead of comparing against ClusterType.GALAXY: BH Galaxy reports
    BLACKHOLE_GALAXY (cluster.hpp:44) and would otherwise be treated as a desktop card.
    """
    ct = _cluster_type()
    if ct is None:
        return False
    return ct in {
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    }


def is_blackhole_galaxy() -> bool:
    ct = _cluster_type()
    return ct is not None and ct == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY


def hw_max_ccl_links() -> int:
    """Maximum ethernet links per direction usable by one CCL op on this cluster.

    Requesting more than the hardware has is not a graceful degradation -- it aborts in
    `TT_FATAL @ tt_metal/fabric/fabric.cpp:163: link_idx < candidate_eth_chans.size()`.
    Measured on the 32-chip BH Galaxy: 1 and 2 links work on both cluster axes, 3 aborts.

    Values follow `models/common/modules/tt_ccl.get_num_links()`, which already tabulates
    this as TG/6U-Galaxy = 4 versus BHGLX = 2. BH Galaxy has only two local connections
    per direction because eth channels 8 and 9 are internally-connected Z links that the
    topology mapper skips (tt_metal/fabric/topology_mapper.cpp:1539).
    """
    if is_blackhole_galaxy():
        return 2
    if is_ubb_galaxy():
        # WH 6U Galaxy has 4; a 4U TG has 3 (see moe_tt._detect_galaxy_ccl, which
        # distinguishes them by PCIe device count).
        return 4
    return 1


def galaxy_fabric_config() -> Any:
    """FabricConfig to set before opening the mesh, or None to leave it alone.

    Ring versus line is not a free choice on Blackhole Galaxy -- it decides which mesh
    shapes exist. `get_fabric_type()` maps FABRIC_1D_RING to TORUS_XY on any UBB galaxy
    (tt_metal/fabric/fabric_host_utils.cpp:36-47), and on a BH Galaxy cabled as a 16x2
    torus that makes SystemMesh `[16, 2]`, so `MeshShape(4, 8)` fails outright with
    "Requested mesh is too big and is not rotatable". Under FABRIC_1D the same machine
    presents `[8, 4]` and both (4,8) and (8,4) open.

    We keep the (4,8) mesh, because that is what the fused collective epilogue, the
    buffered MoE all-reduce and the 2-experts-per-device layout are built around, and
    because ring buys nothing measurable here: `CCL_TOPOLOGY=linear` was measured at
    0.0 ms versus ring on WH. So BH Galaxy gets line fabric, WH Galaxy keeps the ring it
    was tuned with.
    """
    if is_blackhole_galaxy():
        return ttnn.FabricConfig.FABRIC_1D
    if is_ubb_galaxy():
        return ttnn.FabricConfig.FABRIC_1D_RING
    return ttnn.FabricConfig.FABRIC_1D


def default_ccl_topology() -> Any:
    """CCL Topology matching whatever `galaxy_fabric_config()` selected.

    A Ring-topology CCL op on line fabric is not a legal combination, so these two must
    be decided together.
    """
    if galaxy_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
        return ttnn.Topology.Ring
    return ttnn.Topology.Linear


def dispatch_core_config() -> Any:
    """DispatchCoreConfig for this cluster, from the canonical ttnn defaults.

    Hardcoding is a trap on Blackhole, which rejects both of the combinations the WH path
    used: `DispatchCoreAxis.ROW` raises "ROW dispatch core axis is not supported for
    blackhole arch unless fabric tensix MUX is enabled", and `DispatchCoreType.ETH` is
    rejected alongside the COL axis BH requires (ttnn/ttnn/device.py:114-123). Deferring
    to the ttnn helpers gives WORKER+ROW on WH and WORKER+COL on BH.
    """
    return ttnn.DispatchCoreConfig(
        ttnn.device.get_default_dispatch_core_type(),
        ttnn.device.get_default_dispatch_core_axis(ttnn.FabricTensixConfig.DISABLED),
    )


_CCL_CLAMP_WARNED: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log a warning the first time only. These are read per-layer at build time, so an
    unguarded warning would print dozens of identical lines per run."""
    if key not in _CCL_CLAMP_WARNED:
        _CCL_CLAMP_WARNED.add(key)
        logger.warning(message, *args)


def ccl_settings_from_env() -> tuple[int, Any]:
    """`(num_links, topology)` from the env, clamped to what this cluster can express.

    THIS is the only correct way to read GLM4_MOE_LITE_CCL_NUM_LINKS /
    GLM4_MOE_LITE_CCL_TOPOLOGY. Reading `os.environ` directly is a Blackhole bug: the
    winning flag set is a WH-Galaxy artifact that `perf_defaults` setdefaults to 4 links
    and ring topology, and on a BH Galaxy (2 links, line fabric) asking for either one
    ABORTS -- `TT_FATAL @ fabric.cpp:163` for the links, and a ring CCL op on line fabric
    is not a legal pairing. Neither degrades gracefully.

    `Glm4RuntimeConfig.from_env` calls this, but several call sites build their CCL args
    without a Glm4RuntimeConfig in scope (the TP row-parallel all_reduce in
    decoder_layer_tt, the prefill lm_head all_gather in model_tt, the MoE runtime in
    moe_tt), so it is exposed as a function rather than living inside from_env.
    """
    max_links = hw_max_ccl_links()
    requested_links = _env_int("GLM4_MOE_LITE_CCL_NUM_LINKS", default=1)
    links = min(requested_links, max_links) if requested_links > 0 else requested_links
    if requested_links > max_links:
        _warn_once(
            "links",
            "GLM4_MOE_LITE_CCL_NUM_LINKS={} exceeds the {} link(s) this cluster has; using {}. "
            "Requesting more aborts in fabric.cpp rather than degrading.",
            requested_links,
            max_links,
            links,
        )

    hw_topo = default_ccl_topology()
    raw_topo = _env_str("GLM4_MOE_LITE_CCL_TOPOLOGY", default="").lower()
    if not raw_topo:
        return links, hw_topo
    topo = ttnn.Topology.Ring if raw_topo == "ring" else ttnn.Topology.Linear
    if topo == ttnn.Topology.Ring and hw_topo != ttnn.Topology.Ring:
        # Ring CCL ops on line fabric are not a legal pairing, and this cluster's fabric
        # was chosen to keep the (4,8) mesh reachable -- see galaxy_fabric_config().
        _warn_once(
            "topology",
            "GLM4_MOE_LITE_CCL_TOPOLOGY=ring is not available with this cluster's fabric " "config; using Linear.",
        )
        topo = hw_topo
    return links, topo


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

    # --- GlobalCB DRAM weight prefetch ---
    # Off by default: requires the SubDevice split, which confines every decode op to
    # worker columns 0-5. The live prefetcher state (GlobalCB, sub-device id, ring
    # program/memory configs) is carried on the mutable Glm4MoeLitePrefetcherSetup
    # object, since this dataclass is frozen.
    prefetch: bool

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
    fused_collective_epilogue: bool
    buffered_moe_all_reduce: bool
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

    # --- CCL ---
    ccl_num_links: int
    ccl_topology: ttnn.Topology

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

        # CCL settings are clamped to what the cluster physically has -- see
        # ccl_settings_from_env for why reading the env directly is a Blackhole bug.
        ccl_links, ccl_topo = ccl_settings_from_env()

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
            prefetch=_env_bool("GLM4_MOE_LITE_PREFETCH"),
            # Matmul config. The helper only applies this to one-tile, non-batched
            # matmuls; validated on Galaxy B1 at 58.1 -> 54.2 ms/token.
            explicit_prog_cfg=_env_bool("GLM4_MOE_LITE_EXPLICIT_PROG_CFG", default=True),
            # Attention
            concat_heads=_env_bool("GLM4_MOE_LITE_CONCAT_HEADS"),
            attn_dp=_env_bool("GLM4_MOE_LITE_ATTN_DP"),
            head_parallel_kvb2=(_env_bool("GLM4_MOE_LITE_HEAD_PARALLEL_KVB2") and tp_on and tp_sz > 1),
            use_v_cache_slice=_env_bool("GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE"),
            shard_q=_env_bool("GLM4_MOE_LITE_MLA_SHARD_Q"),
            # MLP / MoE
            fuse_mlp_moe_reduce=_env_bool("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE"),
            # On by default (validated: coherent + cross-device-equality gate passes on
            # 4x8; falls back to the safe path on unsupported configs). Disable with =0.
            fused_collective_epilogue=_env_bool("GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE", default=True),
            buffered_moe_all_reduce=_env_bool("GLM4_MOE_LITE_BUFFERED_MOE_ALL_REDUCE", default=True),
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
            # CCL
            ccl_num_links=ccl_links,
            ccl_topology=ccl_topo,
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
