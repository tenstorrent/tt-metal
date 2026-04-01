# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GQA Attention for GLM-4.7-REAP on TT hardware.

Standard Grouped Query Attention (96 Q heads, 8 KV heads, head_dim=128)
with QK norm, partial RoPE (partial_rotary_factor=0.5), and QKV bias.

Reference: tt_transformers/tt/attention.py (Llama/Qwen GQA pattern).
NOT derived from glm4_moe_lite (which uses MLA).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.glm4_moe.tt.ccl import glm4_moe_ccl_num_links_for_axis, glm4_moe_ccl_topology_for_collectives
from models.experimental.glm4_moe.tt.config import Glm4MoeHParams
from models.experimental.glm4_moe.tt.layer_weights import DecoderLayerTTWeights

_REDUCE_IMPL = os.environ.get("GLM4_MOE_REDUCE_IMPL", "host").strip().lower()
_REDUCE_IMPL_AXIS1 = os.environ.get("GLM4_MOE_REDUCE_IMPL_AXIS1", "").strip().lower()
_REDUCE_LOG_ONCE = set()
_QK_L1 = os.environ.get("GLM4_MOE_QK_L1", "1").strip() != "0"
_ROPE_PAD = os.environ.get("GLM4_MOE_ROPE_PAD", "1").strip() != "0"
_FUSED_QK_ROPE = os.environ.get("GLM4_MOE_FUSED_QK_ROPE", "0").strip() == "1"
_FUSED_KV_UPDATE = os.environ.get("GLM4_MOE_FUSED_KV_UPDATE", "0").strip() == "1"
_SDPA_L1 = os.environ.get("GLM4_MOE_SDPA_L1", "1").strip() != "0"

logger = logging.getLogger(__name__)


def _simple_all_reduce_host(tensor, mesh_device, cluster_axis, memory_config=None):
    """Host-side all-reduce fallback for TG 2D mesh.

    Reads device tensors to CPU, sums, and replicates back. Correct but slow.
    """
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    orig_layout = tensor.layout
    orig_dtype = tensor.dtype

    mesh_rows, mesh_cols = list(mesh_device.shape)
    dev_tensors = ttnn.get_device_tensors(tensor)
    num_devs = len(dev_tensors)

    if cluster_axis == 0:
        col0_indices = list(range(0, num_devs, mesh_cols))
        host_sum = ttnn.to_torch(dev_tensors[col0_indices[0]].cpu())
        for idx in col0_indices[1:]:
            host_sum = host_sum + ttnn.to_torch(dev_tensors[idx].cpu())
    elif cluster_axis == 1:
        row0_indices = list(range(0, mesh_cols))
        host_sum = ttnn.to_torch(dev_tensors[row0_indices[0]].cpu())
        for idx in row0_indices[1:]:
            host_sum = host_sum + ttnn.to_torch(dev_tensors[idx].cpu())
    else:
        raise ValueError(f"cluster_axis must be 0 or 1, got {cluster_axis}")

    ttnn.deallocate(tensor, force=False)
    result = ttnn.from_torch(
        host_sum,
        device=mesh_device,
        dtype=orig_dtype,
        layout=orig_layout,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return result


def _simple_all_reduce(tensor, mesh_device, cluster_axis, memory_config=None, ccl=None, impl=None, subdevice_id=None):
    """All-reduce with configurable implementation via GLM4_MOE_REDUCE_IMPL env var.

    Implementations:
      "host"     — Host-side CPU fallback (default, proven correct)
      "native"   — ttnn.all_reduce(cluster_axis=...) on-device
      "rs_ag"    — ttnn.reduce_scatter + ttnn.all_gather 2-step decomposition

    Args:
        subdevice_id: Optional SubDeviceId for prefetcher SubDevice dispatch.
            When set, CCL ops are restricted to the worker SubDevice.
    """
    if mesh_device.__class__.__name__ != "MeshDevice":
        return tensor
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1]:
        return tensor

    # Save input logical shape — CCL ops (reduce_scatter/all_gather) can lose it,
    # returning physical-padded dims instead of the true logical shape.
    input_logical_shape = [int(d) for d in tensor.shape]

    impl = impl or (_REDUCE_IMPL_AXIS1 if (_REDUCE_IMPL_AXIS1 and cluster_axis == 1) else _REDUCE_IMPL)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG

    # Log once per (impl, axis) combination
    log_key = (impl, cluster_axis)
    if log_key not in _REDUCE_LOG_ONCE:
        _REDUCE_LOG_ONCE.add(log_key)
        logger.info("_simple_all_reduce: impl=%s, cluster_axis=%d, mesh=%s", impl, cluster_axis, mesh_shape)

    result = None

    # Build optional subdevice_id kwarg for CCL ops
    _sd_kw = {"subdevice_id": subdevice_id} if subdevice_id is not None else {}

    # Resolve num_links / topology from env vars once per call.
    _ccl_topo = glm4_moe_ccl_topology_for_collectives()
    _ccl_nl = (
        glm4_moe_ccl_num_links_for_axis(int(cluster_axis))
        if cluster_axis is not None
        else max(1, int(os.environ.get("GLM4_MOE_CCL_NUM_LINKS", "1").strip() or "1"))
    )

    if impl == "native":
        result = ttnn.all_reduce(
            tensor,
            cluster_axis=cluster_axis,
            memory_config=mc,
            num_links=_ccl_nl,
            topology=_ccl_topo,
            **_sd_kw,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "native_auto":
        result = ttnn.all_reduce(
            tensor,
            cluster_axis=cluster_axis,
            memory_config=mc,
            num_links=_ccl_nl,
            topology=_ccl_topo,
            **_sd_kw,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "full_ar":
        result = ttnn.all_reduce(
            tensor,
            memory_config=mc,
            num_links=_ccl_nl,
            topology=_ccl_topo,
            **_sd_kw,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "rs_ag":
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=3,
            cluster_axis=cluster_axis,
            memory_config=mc,
            num_links=_ccl_nl,
            topology=_ccl_topo,
            **_sd_kw,
        )
        ttnn.deallocate(tensor, force=False)
        result = ttnn.all_gather(
            scattered,
            dim=3,
            cluster_axis=cluster_axis,
            memory_config=mc,
            num_links=_ccl_nl,
            topology=_ccl_topo,
            **_sd_kw,
        )
        ttnn.deallocate(scattered, force=False)

    elif impl == "rs_ag_async":
        if ccl is None:
            scattered = ttnn.reduce_scatter(
                tensor,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                num_links=_ccl_nl,
                topology=_ccl_topo,
                **_sd_kw,
            )
            ttnn.deallocate(tensor, force=False)
            result = ttnn.all_gather(
                scattered,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                num_links=_ccl_nl,
                topology=_ccl_topo,
                **_sd_kw,
            )
            ttnn.deallocate(scattered, force=False)
        else:
            _topo_async = glm4_moe_ccl_topology_for_collectives()
            rs_params = ccl.get_ccl_params_for_reduce_scatter(axis=cluster_axis)
            scattered = ttnn.experimental.reduce_scatter_minimal_async(
                tensor,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                topology=_topo_async,
                **{**rs_params, **_sd_kw},
            )
            ttnn.deallocate(tensor, force=False)

            ag_params = ccl.get_ccl_params_for_all_gather(axis=cluster_axis)
            result = ttnn.experimental.all_gather_async(
                scattered,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                topology=_topo_async,
                **{**ag_params, **_sd_kw},
            )
            ttnn.deallocate(scattered, force=False)

    else:
        # Default: host-side fallback (proven correct)
        result = _simple_all_reduce_host(tensor, mesh_device, cluster_axis, memory_config)

    # CCL ops (reduce_scatter, all_gather, all_reduce) can lose the logical shape,
    # returning physical-padded dims (e.g., TILE_LAYOUT pads batch to 32) instead of
    # the true logical shape. Restore it so downstream ops (ttnn.add, etc.) don't fail
    # with "Invalid subtile broadcast type" due to mismatched batch dims.
    # PERF: Use int comparison to avoid false positives from ttnn Shape objects.
    # For bs=1, CCL ops don't change the shape — skip reshape entirely.
    out_shape = [int(d) for d in result.shape]
    if out_shape != input_logical_shape:
        out_vol = 1
        in_vol = 1
        for d in out_shape:
            out_vol *= d
        for d in input_logical_shape:
            in_vol *= d
        if out_vol == in_vol:
            # Same volume — safe to reshape (just metadata change).
            result = ttnn.reshape(result, input_logical_shape, out_shape)
        else:
            # CCL padded a dimension (e.g., batch 1→32 in TILE_LAYOUT).
            # Slice back to the original logical shape.
            result = ttnn.slice(
                result,
                starts=[0] * len(input_logical_shape),
                ends=input_logical_shape,
            )

    return result


def _simple_all_gather(tensor, mesh_device, cluster_axis, dim, memory_config=None, subdevice_id=None):
    """Simple all-gather using ttnn.all_gather (no semaphore management).

    For initial bringup — can be replaced with optimized CCL wrappers later.
    """
    if mesh_device.__class__.__name__ != "MeshDevice":
        return tensor
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1]:
        return tensor
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    _sd_kw = {"subdevice_id": subdevice_id} if subdevice_id is not None else {}
    _nl = glm4_moe_ccl_num_links_for_axis(int(cluster_axis))
    _topo = glm4_moe_ccl_topology_for_collectives()
    return ttnn.all_gather(
        tensor,
        dim=dim,
        num_links=_nl,
        cluster_axis=cluster_axis,
        topology=_topo,
        memory_config=mc,
        **_sd_kw,
    )


class Glm4MoeAttention(LightweightModule):
    """GQA attention for GLM-4.7-REAP-218B on Galaxy Wormhole (TG mesh, 32 chips).

    Architecture:
    - 96 Q heads, 8 KV heads (GQA ratio 12:1), head_dim=128
    - TP=8 along columns -> 12 Q heads + 1 KV head per device
    - Fused QKV weight [5120, 1792] per device
    - QKV bias (attention_bias=True)
    - QK norm (RMSNorm per head, dim=128)
    - Partial RoPE: partial_rotary_factor=0.5 -> rotary_dim=64, pass_dim=64
    - Paged KV cache: separate K and V, dtype BF8
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        layer_weights: DecoderLayerTTWeights,
        hparams: Glm4MoeHParams,
        configuration: Any,
        paged_attention_config=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.hparams = hparams

        # Core dimensions
        self.n_heads = hparams.num_attention_heads  # 96
        self.n_kv_heads = hparams.num_key_value_heads  # 8
        self.head_dim = hparams.head_dim  # 128
        self.hidden_size = hparams.hidden_size  # 5120

        # Device topology
        self.num_devices = configuration["num_devices"]  # 32 for Galaxy TG
        self.TG = self.num_devices == 32
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices  # 8
        self.num_device_groups = self.num_devices // self.num_devices_per_group  # 4

        # Per-device head counts (TP=8)
        self.n_local_heads = self.n_heads // self.num_devices_per_group  # 12
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group  # 1

        # Batch sizing
        self.max_batch_size = configuration["max_batch_size"]
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        # Partial RoPE
        self.partial_rotary_factor = hparams.partial_rotary_factor  # 0.5
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)  # 64
        self.pass_dim = self.head_dim - self.rotary_dim  # 64

        # Scale for SDPA
        self.scale = self.head_dim**-0.5

        # Paged attention config
        self.paged_attention_config = paged_attention_config

        # SDPA decode program config: limit cores to 64 (8x8) for Galaxy Wormhole
        # Galaxy has 72 cores (8x9) but SDPA tree reduction supports max 64 cores per KV head
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        # Prefetcher-compatible SDPA config: uses sub_core_grids to restrict to worker
        # cores (cols 0-5). Created lazily by _get_prefetch_sdpa_config().
        self._prefetch_sdpa_config = None

        # Sequence length limits
        self.MAX_QKV_MM_SEQ_LEN = configuration.get("MAX_QKV_MM_SEQ_LEN", 4096)

        # CCL dtype (bfloat16 for TG linear topology)
        self.ccl_dtype = configuration.get("ccl_dtype", ttnn.bfloat16)

        # Compute kernel config — HiFi2 for attention precision through 92 layers
        import os

        _attn_fidelity_raw = os.environ.get("GLM4_MOE_ATTN_FIDELITY", "hifi2").strip().lower()
        _fidelity_map = {
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi3": ttnn.MathFidelity.HiFi3,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }
        _attn_fidelity = _fidelity_map.get(_attn_fidelity_raw, ttnn.MathFidelity.HiFi2)
        _attn_approx = os.environ.get("GLM4_MOE_ATTN_APPROX", "0").strip() != "0"
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_attn_fidelity,
            math_approx_mode=_attn_approx,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Weights from layer_weights (already fused and sharded)
        self.wqkv = layer_weights.w_qkv  # [1, 1, 5120, 1792] per device (TP-sharded)
        self.wqkv_bias = layer_weights.w_qkv_bias  # [1, 1, 1, 1792] per device (TP-sharded)
        self.wo = layer_weights.w_o  # [1, 1, 1536, 5120] per device (row-parallel)
        # Interleaved copies for prefill when prefetch uses DRAM-sharded weights
        self.wqkv_interleaved = layer_weights.w_qkv_interleaved  # None if no prefetch
        self.wo_interleaved = layer_weights.w_o_interleaved  # None if no prefetch

        # QK norm (RMSNorm, per-head dim=128, replicated)
        self.q_norm = layer_weights.q_norm
        self.k_norm = layer_weights.k_norm

        # Prefill program configs — explicit to avoid TG mesh auto-config hangs.
        # Without these, ttnn.linear auto-selects a broken all-DRAM-interleaved TG
        # program config that hangs during mesh device creation/dispatch.
        # Pattern adapted from tt_transformers/tt/model_config.py (Llama Galaxy).
        self._prefill_prog_cfg_cache: dict[str, object] = {}

        # Per-batch-bucket HEIGHT_SHARDED memory configs for decode mode.
        # "auto" L1_HEIGHT_SHARDED_MEMORY_CONFIG doesn't work with to_memory_config —
        # must use explicit shard specs + interleaved_to_sharded.
        # Configs are built lazily per batch bucket and cached in dicts.
        self._grid_size = mesh_device.compute_with_storage_grid_size()
        self._shard_cfg_cache: dict[int, dict] = {}  # batch -> {q, k, rope, trans}

        # DRAM-sharded matmul configs (when GLM4_MOE_DRAM_SHARD=1)
        self._use_dram_shard = os.environ.get("GLM4_MOE_DRAM_SHARD", "0").strip() == "1"
        if self._use_dram_shard:
            self._setup_dram_sharded_configs()

        # Fused QK RoPE caching (B1): doubled cos/sin/trans are created once per
        # decode step and reused across all 89 layers.
        self._fused_rot_cos_id = None
        self._fused_rot_cache = None  # (cos_doubled, sin_doubled)
        self._fused_trans_id = None
        self._fused_trans_cached = None

        # TG user selection matrices (for slicing batch from all-reduce output).
        # Pre-computed per batch bucket because slice_mat and user_selection_matrix
        # dimensions depend on the runtime batch size (B_phys, Bg vary per bucket).
        self._slice_mats: dict[int, object] = {}
        self._user_sel_mats: dict[int, object] = {}
        if self.TG:
            self._tg_init_batch_mats(self.max_batch_size)

    def _tg_init_batch_mats(self, batch_size: int):
        """Lazily create slice_mat and user_selection_matrix for a given batch size."""
        if batch_size in self._slice_mats:
            return
        Ng = self.num_device_groups  # 4 for Galaxy TG
        Bg = max(batch_size // Ng, 1)  # per DP group
        B_phys = ((batch_size + 31) // 32) * 32  # tile-padded

        # slice_mat: [1, num_devices, Bg, B_phys] — each device selects its DP group's batch entries
        weight = torch.zeros(1, self.num_devices, Bg, B_phys)
        for i in range(self.num_devices):
            col = i % Ng
            weight[:, i, :, col * Bg : (col + 1) * Bg] = torch.eye(Bg)
        self._slice_mats[batch_size] = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )

        # user_selection_matrix: [Ng*Bg, Ng*B_phys] — reorders batch after all-gather across DP groups
        Bg_phys = ((Bg + 31) // 32) * 32
        user_sel = torch.eye(Bg, Bg)
        user_sel = torch.nn.functional.pad(user_sel, (0, Bg_phys - Bg), "constant", 0)
        user_sel = [user_sel] * Ng
        user_sel = torch.block_diag(*user_sel)
        self._user_sel_mats[batch_size] = ttnn.from_torch(
            user_sel,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _get_prefetch_sdpa_config(self):
        """Lazily create SDPA program config restricted to worker cores (cols 0-5)."""
        if self._prefetch_sdpa_config is not None:
            return self._prefetch_sdpa_config
        # Worker grid: cols 0-5, rows 0-8 = 6x9 = 54 cores
        worker_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))])
        start_core = ttnn.CoreCoord(0, 0)
        num_sdpa_cores = 54  # all worker cores
        self._prefetch_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(6, 9),
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                start_core,
                num_sdpa_cores,
                worker_core_range_set,
                row_wise=True,
            ),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        return self._prefetch_sdpa_config

    def _get_prefill_program_config(self, seq_len: int, K: int, N: int, label: str = ""):
        """Build MatmulMultiCoreReuseMultiCastProgramConfig for prefill matmuls on TG mesh.

        Without explicit program_config, ttnn.linear auto-selects a program that hangs
        on TG mesh. This mirrors the Llama Galaxy approach from
        tt_transformers/tt/model_config.py:get_attn_qkv_program_config(PREFILL).

        Args:
            seq_len: sequence length (after any reshape for long sequences)
            K: inner dimension (weight rows)
            N: output dimension (weight columns, per device)
            label: cache key label (e.g. "qkv", "o_proj")
        """
        import math

        cache_key = f"{label}_{seq_len}_{K}_{N}"
        if cache_key in self._prefill_prog_cfg_cache:
            return self._prefill_prog_cfg_cache[cache_key]

        tile_size = 32
        grid_rows = 8
        grid_cols = 8  # WH per-chip compute grid: 8x8 (conservative, works on both T3K and Galaxy)

        M_tiles = seq_len // tile_size
        N_tiles = math.ceil(N / tile_size)

        per_core_M = max(1, math.ceil(M_tiles / grid_rows))
        per_core_N = max(1, math.ceil(N_tiles / grid_cols))

        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_cols, grid_rows),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        self._prefill_prog_cfg_cache[cache_key] = cfg
        return cfg

    def _get_minimal_matmul_config(self):
        """Return a MinimalMatmulConfig for large-sequence prefill matmuls.

        Unlike MatmulMultiCoreReuseMultiCastProgramConfig, MinimalMatmulConfig
        has no num_blocks_y <= num_cores_y constraint, so it works for arbitrary
        M dimensions (including fuse_batch with many batches).  This mirrors the
        Llama Galaxy approach from tt_transformers/tt/attention.py.

        Block sizes are kept at 4 to fit within Wormhole L1 SRAM (1499136 B)
        given GLM's large hidden_size (K=5120) for QKV.
        """
        if not hasattr(self, "_minimal_mm_cfg"):
            self._minimal_mm_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=4,
                N_block_size=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
        return self._minimal_mm_cfg

    def _get_shard_cfgs(self, batch: int, prefetch: bool = False) -> dict:
        """Return per-batch shard configs (lazily created and cached).

        When prefetch=True, uses sub_core_grids to place shards within the worker
        grid (cols 0-5) instead of the full grid. This avoids overlapping with
        prefetcher sender cores (cols 6 and 7).

        When _FUSED_QK_ROPE is enabled (non-prefetch only), Q and K are placed on
        disjoint core grids as required by rotary_embedding_llama_fused_qk. Extra
        shard configs for the doubled cos/sin/trans are also created.
        """
        cache_key = (batch, prefetch)
        cached = self._shard_cfg_cache.get(cache_key)
        if cached is not None:
            return cached
        b = max(batch, 1)
        if prefetch:
            worker_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))])
            start_core = ttnn.CoreCoord(0, 0)
            user_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                start_core,
                b,
                worker_crs,
                row_wise=True,
            )
        else:
            user_grid = ttnn.num_cores_to_corerangeset(b, self._grid_size, row_wise=True)

        _height_shard = dict(
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        if _FUSED_QK_ROPE and not prefetch:
            from models.tt_transformers.tt.model_config import num_to_corerange

            row_size = self._grid_size.x  # 8 for Galaxy WH
            k_start = ttnn.CoreCoord(b % row_size, b // row_size)
            q_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
            k_grid = ttnn.CoreRangeSet({num_to_corerange(b, start_core=k_start)})
            combined_grid = ttnn.CoreRangeSet({num_to_corerange(2 * b)})

            cfgs = {
                "q": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=q_grid,
                    **_height_shard,
                ),
                "k": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=k_grid,
                    **_height_shard,
                ),
                "fused_cos_sin": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=combined_grid,
                    **_height_shard,
                ),
                "fused_trans": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                    core_grid=combined_grid,
                    **_height_shard,
                ),
            }
        else:
            cfgs = {
                "q": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=user_grid,
                    **_height_shard,
                ),
                "k": ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=user_grid,
                    **_height_shard,
                ),
            }

        self._shard_cfg_cache[cache_key] = cfgs
        return cfgs

    def _setup_dram_sharded_configs(self):
        """Pre-compute DRAM-sharded matmul configs and weight copies for decode attention.

        Creates DRAM WIDTH_SHARDED copies of QKV and O weights for decode (full DRAM bank
        utilization), while keeping the original interleaved weights for prefill.
        """
        from models.demos.deepseek_v3.utils.config_helpers import (
            dram_sharded_weight_config,
            get_activation_sharding_core_counts_for_dram_matmul,
            get_dram_sharded_matmul_config,
        )

        grid = self._grid_size
        max_cores = grid.x * grid.y  # 72 for Galaxy WH (8x9)
        _DS_BATCH = 32  # tile-padded batch for decode

        def _ds_act_mc(width: int) -> ttnn.MemoryConfig:
            cores = max(get_activation_sharding_core_counts_for_dram_matmul(width, max_cores))
            return ttnn.create_sharded_memory_config_(
                shape=(_DS_BATCH, width // cores),
                core_grid=ttnn.num_cores_to_corerangeset(
                    cores,
                    ttnn.CoreCoord(grid.x, grid.y),
                    row_wise=True,
                ),
                strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )

        def _shard_weight(weight):
            """Create a DRAM WIDTH_SHARDED copy of a weight tensor."""
            K, N = int(weight.shape[2]), int(weight.shape[3])
            dram_grid = self.mesh_device.dram_grid_size()
            dram_mc = dram_sharded_weight_config(K, N, dram_grid)
            host_w = ttnn.from_device(weight)
            return host_w.to(self.mesh_device, dram_mc)

        # QKV matmul: [1,1,B,5120] × [1,1,5120,1792] → [1,1,B,1792]
        K_qkv = int(self.wqkv.shape[2])
        N_qkv = int(self.wqkv.shape[3])
        qkv_in = max(get_activation_sharding_core_counts_for_dram_matmul(K_qkv, max_cores))
        qkv_out = max(get_activation_sharding_core_counts_for_dram_matmul(N_qkv, max_cores))
        self._ds_qkv_act_mc = _ds_act_mc(K_qkv)
        self._ds_qkv_prog_cfg = get_dram_sharded_matmul_config(
            m=_DS_BATCH,
            k=K_qkv,
            n=N_qkv,
            input_num_shards=qkv_in,
            output_num_shards=qkv_out,
        )
        self._ds_wqkv = _shard_weight(self.wqkv)

        # O projection: [1,1,B,1536] × [1,1,1536,5120] → [1,1,B,5120]
        K_o = int(self.wo.shape[2])
        N_o = int(self.wo.shape[3])
        o_in = max(get_activation_sharding_core_counts_for_dram_matmul(K_o, max_cores))
        o_out = max(get_activation_sharding_core_counts_for_dram_matmul(N_o, max_cores))
        self._ds_o_act_mc = _ds_act_mc(K_o)
        self._ds_o_prog_cfg = get_dram_sharded_matmul_config(
            m=_DS_BATCH,
            k=K_o,
            n=N_o,
            input_num_shards=o_in,
            output_num_shards=o_out,
        )
        self._ds_wo = _shard_weight(self.wo)

        logger.info(
            "DRAM-sharded attention configs: QKV K=%d N=%d in=%d out=%d, O K=%d N=%d in=%d out=%d",
            K_qkv,
            N_qkv,
            qkv_in,
            qkv_out,
            K_o,
            N_o,
            o_in,
            o_out,
        )

    def _apply_rope_fused_decode(self, x, cos, sin, trans_mat):
        """Apply partial RoPE via fused rotary_embedding_llama kernel (decode mode).

        x: [1, batch, n_heads, head_dim=128] — interleaved layout from permuted QKV weights.
        cos: [1, batch, 1, head_dim=128] — interleaved + pass-through (cos=1, sin=0).
        sin: [1, batch, 1, head_dim=128]
        trans_mat: [1, 1, 32, 32]

        Replaces the 8-10 op manual NeoX slice/concat chain with a single fused kernel.
        The NeoX→interleaved conversion is absorbed into QKV weight columns at load time.
        """
        return ttnn.experimental.rotary_embedding_llama(
            x,
            cos,
            sin,
            trans_mat,
            is_decode_mode=True,
        )

    def _prepare_fused_qk_rot_mats(self, cos, sin, trans_mat, shard_cfgs):
        """Double cos/sin/trans for fused QK RoPE. Cached in DRAM interleaved.

        The fused kernel expects cos/sin batch_dim = q_batch + k_batch = 2*batch,
        and trans_mat sharded on >= 2*batch cores. Doubled tensors are cached in
        DRAM interleaved format (no L1 pressure) and converted to HEIGHT_SHARDED
        per call, then immediately deallocated to avoid L1 clashes with downstream ops.
        """
        cos_id = id(cos)
        if cos_id != self._fused_rot_cos_id:
            if self._fused_rot_cache is not None:
                ttnn.deallocate(self._fused_rot_cache[0], force=False)
                ttnn.deallocate(self._fused_rot_cache[1], force=False)
            cos_i = ttnn.sharded_to_interleaved(cos, ttnn.DRAM_MEMORY_CONFIG)
            sin_i = ttnn.sharded_to_interleaved(sin, ttnn.DRAM_MEMORY_CONFIG)
            cos_dram = ttnn.concat([cos_i, cos_i], dim=1)
            sin_dram = ttnn.concat([sin_i, sin_i], dim=1)
            ttnn.deallocate(cos_i, force=False)
            ttnn.deallocate(sin_i, force=False)
            self._fused_rot_cos_id = cos_id
            self._fused_rot_cache = (cos_dram, sin_dram)

        trans_id = id(trans_mat)
        if trans_id != self._fused_trans_id:
            if self._fused_trans_cached is not None:
                ttnn.deallocate(self._fused_trans_cached, force=False)
            trans_i = ttnn.sharded_to_interleaved(trans_mat, ttnn.DRAM_MEMORY_CONFIG)
            trans_dram = ttnn.concat([trans_i, trans_i], dim=1)
            ttnn.deallocate(trans_i, force=False)
            self._fused_trans_cached = trans_dram
            self._fused_trans_id = trans_id

        cos_ds = ttnn.interleaved_to_sharded(self._fused_rot_cache[0], shard_cfgs["fused_cos_sin"])
        sin_ds = ttnn.interleaved_to_sharded(self._fused_rot_cache[1], shard_cfgs["fused_cos_sin"])
        trans_ds = ttnn.interleaved_to_sharded(self._fused_trans_cached, shard_cfgs["fused_trans"])
        return (cos_ds, sin_ds, trans_ds)

    def _apply_rope_fused_qk_decode(self, q, k, cos, sin, trans_mat, shard_cfgs):
        """Apply fused Q+K RoPE in a single kernel launch (decode mode).

        Requires Q and K on disjoint HEIGHT_SHARDED core grids, and cos/sin/trans
        with doubled batch dimension. The doubled HEIGHT_SHARDED tensors are created
        from DRAM cache right before the kernel and deallocated immediately after to
        avoid L1 conflicts with downstream ops.
        """
        cos_d, sin_d, trans_d = self._prepare_fused_qk_rot_mats(cos, sin, trans_mat, shard_cfgs)
        q_out, k_out = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q,
            k,
            cos_d,
            sin_d,
            trans_d,
        )
        ttnn.deallocate(cos_d, force=False)
        ttnn.deallocate(sin_d, force=False)
        ttnn.deallocate(trans_d, force=False)
        return q_out, k_out

    def _apply_rope_fused_prefill(self, x, cos, sin, trans_mat):
        """Apply partial RoPE via fused rotary_embedding_llama kernel (prefill mode).

        x: [1, n_heads, seq_len, head_dim=128] — interleaved layout.
        cos: [1, 1, padded_len, head_dim=128] — sliced to seq_len inside kernel.
        sin: [1, 1, padded_len, head_dim=128]
        trans_mat: [1, 1, 32, 32]
        """
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, dtype=ttnn.bfloat16)
        seq_len = int(x.shape[2])
        head_dim = int(x.shape[3])
        # Slice cos/sin to match seq_len if they're padded to max_seq_len
        cos_sl = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
        sin_sl = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
        return ttnn.experimental.rotary_embedding_llama(
            x,
            cos_sl,
            sin_sl,
            trans_mat,
            is_decode_mode=False,
        )

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
        active_batch: int | None = None,
        global_cb=None,
        sub_device_id=None,
        prefetch_qkv_pc=None,
        prefetch_oproj_pc=None,
        prefetch_qkv_in_mc=None,
        prefetch_qkv_out_mc=None,
        prefetch_oproj_in_mc=None,
        prefetch_oproj_out_mc=None,
    ) -> ttnn.Tensor:
        """Decode forward: single token per sequence.

        Args:
            x: [seq_len=1, 1, batch, hidden_size=5120]
            current_pos: [batch_size] current token positions
            rot_mats: (cos, sin) rotation matrices for RoPE
            page_table: page table tensor for paged KV cache
            kv_cache: [keys, values] external KV cache tensors
            active_batch: Actual logical batch size (from model_tt, not x.shape which
                may be tile-padded by ttnn.add / sharded_to_interleaved)

        Returns:
            Output tensor [1, 1, batch, hidden_size=5120]
        """

        # 1. QKV matmul
        if self._use_dram_shard:
            x_sharded = ttnn.to_memory_config(x, self._ds_qkv_act_mc)
            xqkv = ttnn.linear(
                x_sharded,
                self._ds_wqkv,
                program_config=self._ds_qkv_prog_cfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(x_sharded, force=False)
        elif global_cb is not None:
            # Prefetcher path: gather_in0 ring matmul.
            # Requirements (from test_matmul_1d_gather_in0.py):
            #   1. Must use ttnn.matmul (NOT ttnn.linear — gather_in0 doesn't support bias)
            #   2. Input must be WIDTH_SHARDED on ring cores
            #   3. Output must have explicit shard spec
            x_ring = ttnn.to_memory_config(x, prefetch_qkv_in_mc)
            xqkv = ttnn.matmul(
                x_ring,
                self.wqkv,
                program_config=prefetch_qkv_pc,
                memory_config=prefetch_qkv_out_mc,
                compute_kernel_config=self.compute_kernel_config,
                global_cb=global_cb,
                sub_device_id=sub_device_id,
                dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
            )
            ttnn.deallocate(x_ring, force=False)
        else:
            _w = self.wqkv_interleaved or self.wqkv
            xqkv = ttnn.linear(
                x,
                _w,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )

        # Add QKV bias
        xqkv = xqkv + self.wqkv_bias

        ttnn.deallocate(x)

        # Use caller-provided active_batch (reliable) instead of x.shape[-2] which
        # can be inflated to 32 by tile-padding in ttnn.add / sharded_to_interleaved.
        if active_batch is None:
            active_batch = int(x.shape[-2])
        dp_size = self.num_device_groups  # 4 for TG
        tg_batch_sliced = self.TG and active_batch > 1 and active_batch % dp_size == 0

        # TG: slice batch to this DP group's entries via slice_mat matmul.
        # slice_mat expects xqkv dim[-2] = B_phys (tile-padded batch = 32).
        # With bs<32, logical dim[-2]<32 but TILE_LAYOUT pads to 32 physically.
        # Reshape to expose physical batch as logical so matmul inner dims match.
        if tg_batch_sliced:
            # TG multi-user path currently expects DRAM interleaved before slice-mat.
            xqkv = ttnn.to_memory_config(xqkv, ttnn.DRAM_MEMORY_CONFIG)
            # Multi-user: slice batch to this DP group entries via matmul.
            # Lazily create slice_mat/user_sel_mat for this batch size if needed.
            self._tg_init_batch_mats(active_batch)
            xqkv_shape = [int(d) for d in xqkv.shape]
            B_phys = ((active_batch + 31) // 32) * 32
            if xqkv_shape[-2] != B_phys:
                xqkv = ttnn.reshape(xqkv, (1, 1, B_phys, xqkv_shape[-1]), (1, 1, B_phys, xqkv_shape[-1]))
            _slice_kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            if sub_device_id is not None:
                _slice_kw["sub_device_id"] = sub_device_id
                # Restrict to worker sub-device cores (cols 0-5, rows 0-8 = 6x9)
                _slice_kw["core_grid"] = ttnn.CoreGrid(y=6, x=4)
            xqkv = ttnn.matmul(
                self._slice_mats[active_batch],
                xqkv,
                **_slice_kw,
            )
            logical_batch_after_slice = active_batch // dp_size
        else:
            # Single-user or unsupported batch size: skip slice_mat.
            xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            logical_batch_after_slice = active_batch

        # 3. Reshape xqkv so nlp_create_qkv_heads_decode sees the true logical batch,
        #    not the tile-padded physical batch (32).  Without this, the op outputs
        #    q/k/v with batch=32 which breaks paged_update_cache (page_table has batch=1).
        #    Mirrors tt_transformers/tt/attention.py lines 497-501.
        _fqkv_shape = xqkv.shape
        B_phys_local = ((logical_batch_after_slice + 31) // 32) * 32
        xqkv = ttnn.reshape(
            xqkv,
            (1, 1, logical_batch_after_slice, int(_fqkv_shape[3])),
            (1, 1, B_phys_local, int(_fqkv_shape[3])),
        )

        # Split into Q, K, V heads (output must be HEIGHT_SHARDED per ttnn API)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.n_local_heads,  # 12
            num_kv_heads=self.n_local_kv_heads,  # 1
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        ttnn.deallocate(xqkv)

        # 4. QK Norm (RMSNorm per head, dim=128)
        # RMSNorm requires interleaved input (HEIGHT_SHARDED not supported by layernorm op)
        # GLM4_MOE_QK_L1=1 (default): keep Q/K in L1 interleaved instead of DRAM round-trip.
        # Q=98KB, K=8KB — trivially fit L1. RMSNorm accepts L1 interleaved.
        _qk_mc = ttnn.L1_MEMORY_CONFIG if _QK_L1 else ttnn.DRAM_MEMORY_CONFIG
        q = ttnn.to_memory_config(q, _qk_mc)
        k = ttnn.to_memory_config(k, _qk_mc)
        q = self.q_norm(q, mode="decode")
        k = self.k_norm(k, mode="decode")

        # 5. Fused RoPE via rotary_embedding_llama (1 kernel instead of 8-10 ops).
        # QKV weights were column-permuted NeoX→interleaved at load time, so Q/K
        # are already in the interleaved format the fused kernel expects.
        # cos/sin include pass-through padding (cos=1, sin=0) for the non-rotary dims.
        # Decode-mode fused kernel requires ALL inputs HEIGHT_SHARDED:
        #   Q/K: convert from interleaved (post-QK-norm) to HEIGHT_SHARDED
        #   cos/sin/trans_mat: caller (model_tt) converts once before the layer loop
        # When prefetcher SubDevice is active, restrict eltwise ops to worker cores (cols 0-5)
        _worker_scg = None
        if sub_device_id is not None:
            _worker_scg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))])
        _shard_cfgs = self._get_shard_cfgs(logical_batch_after_slice, prefetch=(sub_device_id is not None))
        q = ttnn.interleaved_to_sharded(q, _shard_cfgs["q"])
        k = ttnn.interleaved_to_sharded(k, _shard_cfgs["k"])
        if _FUSED_QK_ROPE and sub_device_id is None:
            q, k = self._apply_rope_fused_qk_decode(
                q,
                k,
                rot_mats[0],
                rot_mats[1],
                rot_mats[2],
                _shard_cfgs,
            )
        else:
            q = self._apply_rope_fused_decode(q, rot_mats[0], rot_mats[1], rot_mats[2])
            k = self._apply_rope_fused_decode(k, rot_mats[0], rot_mats[1], rot_mats[2])
        # Output is already HEIGHT_SHARDED (fused kernel preserves input memory layout)

        # 6. KV cache update
        keys = kv_cache[0]
        values = kv_cache[1]

        # Multi-device correctness: pass mesh_coords to ensure KV update dispatches
        # to ALL mesh coordinates. Without it, KV boundary corruption observed on TG
        # mesh when second KV page is first touched. (Ported from glm4_moe_lite.)
        _puc_kw = dict(update_idxs_tensor=current_pos, page_table=page_table)
        if self.mesh_device.__class__.__name__ == "MeshDevice":
            try:
                _mr, _mc = int(self.mesh_device.shape[0]), int(self.mesh_device.shape[1])
                _puc_kw["mesh_coords"] = {ttnn.MeshCoordinate(r, c) for r in range(_mr) for c in range(_mc)}
            except Exception:
                pass
        _can_fuse_kv = _FUSED_KV_UPDATE and k.is_sharded() and v.is_sharded() and _FUSED_QK_ROPE
        if _can_fuse_kv:
            ttnn.experimental.paged_fused_update_cache(keys, k, values, v, **_puc_kw)
        else:
            ttnn.experimental.paged_update_cache(keys, k, **_puc_kw)
            ttnn.experimental.paged_update_cache(values, v, **_puc_kw)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # 7. SDPA (paged) — limit cores for Galaxy Wormhole tree reduction.
        # When prefetcher is active, use sub_core_grids to restrict to worker cores (cols 0-5).
        _sdpa_pc = self._get_prefetch_sdpa_config() if sub_device_id is not None else self.sdpa_decode_program_config
        _sdpa_mc = ttnn.L1_MEMORY_CONFIG if _SDPA_L1 else ttnn.DRAM_MEMORY_CONFIG
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=_sdpa_pc,
            memory_config=_sdpa_mc,
        )
        ttnn.deallocate(q)

        # 8. Concat heads (requires HEIGHT_SHARDED input)
        attn_output = ttnn.interleaved_to_sharded(attn_output, _shard_cfgs["q"])
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,  # 12
            sub_core_grids=_worker_scg,
        )
        # -> [1, 1, batch, 1536] (12 heads * 128 dim)

        # 9. TG multi-user: gather DP groups' batch entries, then select real entries.
        # After concat_heads, each device has [1,1,B_phys,1536] with only Bg real entries.
        # all_gather(dim=2, axis=1) concatenates 4 DP groups → [1,1,4*B_phys,1536].
        # user_selection_matrix [active_batch, 4*B_phys] selects the Bg real entries
        # from each group, discarding padding → [1,1,active_batch,1536].
        # Pattern from tt_transformers/tt/attention.py lines 667-687.
        if tg_batch_sliced:
            attn_shape = [int(d) for d in attn_output.shape]
            B_phys = ((logical_batch_after_slice + 31) // 32) * 32
            if attn_shape[-2] != B_phys:
                attn_output = ttnn.reshape(attn_output, (1, 1, B_phys, attn_shape[-1]), (1, 1, B_phys, attn_shape[-1]))

            # Gather across DP groups (cluster_axis=1) on batch dim
            attn_output = _simple_all_gather(
                attn_output,
                self.mesh_device,
                cluster_axis=1,
                dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                subdevice_id=sub_device_id,
            )

            attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
            # Worker grid: cols 0-5 when prefetcher active, else full 8 cols
            _usel_grid = ttnn.CoreGrid(y=4, x=6) if sub_device_id is not None else ttnn.CoreGrid(y=4, x=8)
            _usel_kw = dict(
                core_grid=_usel_grid,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            if sub_device_id is not None:
                _usel_kw["sub_device_id"] = sub_device_id
            attn_output = ttnn.matmul(
                self._user_sel_mats[active_batch],
                attn_output,
                **_usel_kw,
            )
            B_phys_out = ((active_batch + 31) // 32) * 32
            attn_output = ttnn.reshape(
                attn_output, (1, 1, active_batch, attn_shape[-1]), (1, 1, B_phys_out, attn_shape[-1])
            )

        # 10. Output projection (BF16 output for precision through 92 layers)
        if self._use_dram_shard:
            ao_sharded = ttnn.to_memory_config(attn_output, self._ds_o_act_mc)
            ttnn.deallocate(attn_output)
            dense_out = ttnn.linear(
                ao_sharded,
                self._ds_wo,
                program_config=self._ds_o_prog_cfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(ao_sharded, force=False)
        elif global_cb is not None:
            # Prefetcher path: gather_in0 ring matmul for O-proj.
            # Same pattern as QKV — reshard input, use ttnn.matmul, explicit output config.
            ao_ring = ttnn.to_memory_config(attn_output, prefetch_oproj_in_mc)
            ttnn.deallocate(attn_output)
            dense_out = ttnn.matmul(
                ao_ring,
                self.wo,
                program_config=prefetch_oproj_pc,
                memory_config=prefetch_oproj_out_mc,
                compute_kernel_config=self.compute_kernel_config,
                global_cb=global_cb,
                sub_device_id=sub_device_id,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(ao_ring, force=False)
        else:
            _w_o = self.wo_interleaved or self.wo
            _oproj_grid = ttnn.CoreGrid(y=4, x=8) if self.TG else None
            dense_out = ttnn.matmul(
                attn_output,
                _w_o,
                core_grid=_oproj_grid,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(attn_output)

        # DEBUG: force synchronize to isolate if compute pipeline or reduce hangs
        import os as _os

        if _os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            logger.info("  [DEBUG] synchronizing before all_reduce, dense_out.shape=%s", list(dense_out.shape))
            ttnn.synchronize_device(self.mesh_device)
            logger.info("  [DEBUG] synchronize complete, proceeding to all_reduce")

        # 11. All-reduce on cluster_axis=0 (TP reduction for row-parallel O projection)
        dense_out = _simple_all_reduce(
            dense_out,
            self.mesh_device,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ccl=self.tt_ccl,
            subdevice_id=sub_device_id,
        )

        return dense_out

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Prefill forward: process entire prompt sequence.

        Args:
            x: [1, 1, seq_len, hidden_size=5120]
            rot_mats: (cos, sin) rotation matrices for RoPE
            user_id: batch index for KV cache fill
            page_table: page table tensor for paged KV cache
            chunk_page_table: page table for chunked prefill
            chunk_start_idx: starting index for chunked prefill
            kv_cache: [keys, values] external KV cache tensors

        Returns:
            Output tensor [1, 1, seq_len, hidden_size=5120]
        """
        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        import os as _os
        import sys

        _dbg = _os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
        _dbg_prefill = _os.environ.get("GLM4_MOE_DEBUG_PREFILL", "0") != "0"

        def _sync(label):
            if _dbg:
                print(f"  [DEBUG PREFILL] {label} ...", flush=True, file=sys.stderr)
                ttnn.synchronize_device(self.mesh_device)
                print(f"  [DEBUG PREFILL] {label} OK", flush=True, file=sys.stderr)

        if _dbg_prefill:
            print(
                f"      [ATTN PREFILL] seq_len={seq_len}, user_id={user_id}, "
                f"chunk_start_idx={chunk_start_idx}, "
                f"page_table={'None' if page_table is None else list(page_table.shape)}, "
                f"chunk_page_table={'None' if chunk_page_table is None else list(chunk_page_table.shape)}, "
                f"kv_cache={'None' if kv_cache is None else f'[{list(kv_cache[0].shape)}, {list(kv_cache[1].shape)}]'}",
                flush=True,
                file=sys.stderr,
            )

        _sync("before QKV linear")

        # 1. QKV linear (reshape for long sequences)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x = ttnn.reshape(x, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        _wqkv = self.wqkv_interleaved if self.wqkv_interleaved is not None else self.wqkv

        _use_minimal_mm = self.TG and seq_len > 128

        print(
            f"      [ATTN PREFILL] QKV matmul: x.shape={list(x.shape)}, w.shape={list(_wqkv.shape)}, "
            f"w.memory_config={_wqkv.memory_config()}, using_interleaved={self.wqkv_interleaved is not None}, "
            f"minimal_matmul={_use_minimal_mm}, dtype={self.ccl_dtype if self.TG else 'bf16'}",
            flush=True,
            file=sys.stderr,
        )

        if _use_minimal_mm:
            xqkv = ttnn.experimental.minimal_matmul(
                x,
                _wqkv,
                compute_kernel_config=self.compute_kernel_config,
                config=self._get_minimal_matmul_config(),
            )
        else:
            _mm_seq = min(seq_len, self.MAX_QKV_MM_SEQ_LEN)
            _qkv_pc = (
                self._get_prefill_program_config(_mm_seq, int(_wqkv.shape[2]), int(_wqkv.shape[3]), label="qkv")
                if self.TG
                else None
            )
            _qkv_kwargs = {}
            if _qkv_pc is not None:
                _qkv_kwargs["program_config"] = _qkv_pc
            xqkv = ttnn.linear(
                x,
                _wqkv,
                dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                **_qkv_kwargs,
            )

        print(f"      [ATTN PREFILL] QKV matmul issued (async), syncing...", flush=True, file=sys.stderr)

        _sync("after QKV linear")

        # Add QKV bias
        xqkv = xqkv + self.wqkv_bias

        # Column-parallel QKV: each device has its own output slice, no all-reduce needed.

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # 3. Split into Q, K, V heads (prefill)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,  # 12
            num_kv_heads=self.n_local_kv_heads,  # 1
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # q: [1, 12, seq_len, 128], k: [1, 1, seq_len, 128], v: [1, 1, seq_len, 128]

        ttnn.deallocate(xqkv)
        _sync("after split heads")

        # 4. QK Norm
        q = self.q_norm(q, mode="prefill")
        k = self.k_norm(k, mode="prefill")

        _sync("after QK norm")

        # 5. Partial RoPE (prefill mode)
        if _dbg_prefill:
            print(
                f"      [ATTN PREFILL] RoPE: cos.shape={list(rot_mats[0].shape)}, "
                f"sin.shape={list(rot_mats[1].shape)}, q.shape={list(q.shape)}, "
                f"k.shape={list(k.shape)} — NOTE: cos/sin always sliced from [0:seq_len], "
                f"not from chunk_start_idx (Bug 2: wrong positions for chunks > 0)",
                flush=True,
                file=sys.stderr,
            )
        q = self._apply_rope_fused_prefill(q, rot_mats[0], rot_mats[1], rot_mats[2])
        k = self._apply_rope_fused_prefill(k, rot_mats[0], rot_mats[1], rot_mats[2])
        _sync("after RoPE")

        # 6. Fill KV cache
        keys = kv_cache[0]
        values = kv_cache[1]

        k_8b = ttnn.typecast(k, dtype=keys.dtype)
        ttnn.deallocate(k)
        v_8b = ttnn.typecast(v, dtype=values.dtype)
        ttnn.deallocate(v)

        # BUGFIX: Always fill ALL devices' KV caches in prefill.
        # _prefill_prepare_tensor_for_kv_cache selects only one DP column (8 devices),
        # but batch=1 decode uses all 32 devices without DP sharding — the 24 unfilled
        # devices produce garbage attention, corrupting the EP reduce sum.
        # K,V are identical across DP columns (same TP position data), so filling
        # all devices is correct and harmless for batch>1 (unused data is never read).
        k_fill = k_8b
        v_fill = v_8b

        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        if fill_page_table is not None:
            block_size = keys.shape[2]
            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            if _dbg_prefill:
                print(
                    f"      [ATTN PREFILL] KV cache fill: paged_fill_cache, "
                    f"fill_page_table.shape={list(fill_page_table.shape)}, "
                    f"block_size={block_size}, page_len={page_len}, "
                    f"k_fill.shape={list(k_fill.shape)}, v_fill.shape={list(v_fill.shape)}, "
                    f"keys.shape={list(keys.shape)}, user_id={user_id}",
                    flush=True,
                    file=sys.stderr,
                )
            ttnn.experimental.paged_fill_cache(keys, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            if _dbg_prefill:
                print(
                    f"      [ATTN PREFILL] KV cache fill: fill_cache (non-paged), "
                    f"user_id={user_id}, batch_idx={user_id % self.batch_size_per_device_group}",
                    flush=True,
                    file=sys.stderr,
                )
            ttnn.fill_cache(keys, k_fill, user_id % self.batch_size_per_device_group)
            ttnn.fill_cache(values, v_fill, user_id % self.batch_size_per_device_group)

        _sync("after KV cache fill")

        # 7. SDPA — match Q dtype to KV cache dtype (avoid BF8 precision loss for BF16 KV)
        q_8b = ttnn.typecast(q, dtype=keys.dtype)
        ttnn.deallocate(q)

        if chunk_start_idx is not None:
            if _dbg_prefill:
                print(
                    f"      [ATTN PREFILL] SDPA: chunked mode, chunk_start_idx={chunk_start_idx}, "
                    f"q.shape={list(q_8b.shape)}, keys.shape={list(keys.shape)}",
                    flush=True,
                    file=sys.stderr,
                )
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_8b,
                input_tensor_k=keys,
                input_tensor_v=values,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
            )
        else:
            if _dbg_prefill:
                print(
                    f"      [ATTN PREFILL] SDPA: local mode (NO cross-chunk attention), "
                    f"q.shape={list(q_8b.shape)}, k.shape={list(k_8b.shape)}, v.shape={list(v_8b.shape)}",
                    flush=True,
                    file=sys.stderr,
                )
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                q_8b,
                k_8b,
                v_8b,
                is_causal=True,
                scale=self.scale,
            )

        ttnn.deallocate(q_8b)
        ttnn.deallocate(k_8b)
        ttnn.deallocate(v_8b)

        _sync("after SDPA")

        # 8. Reshape and concat heads
        attn_output = ttnn.reshape(attn_output, [1, self.n_local_heads, -1, self.head_dim])
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # -> [1, 1, seq_len, 1536]

        _sync("after concat_heads")

        # Reshape for long sequences (must match QKV path threshold)
        _oproj_thresh = self.MAX_QKV_MM_SEQ_LEN
        if seq_len > _oproj_thresh:
            if seq_len % _oproj_thresh != 0:
                raise ValueError(f"O-proj: seq_len {seq_len} must be divisible by {_oproj_thresh}")
            attn_output = ttnn.reshape(attn_output, [1, seq_len // _oproj_thresh, _oproj_thresh, -1])

        # 9. Output projection (BF16 output for precision through 92 layers)
        _wo = self.wo_interleaved if self.wo_interleaved is not None else self.wo
        _use_minimal_o = self.TG and seq_len > 128

        print(
            f"      [ATTN PREFILL] before O-proj linear (minimal_matmul={_use_minimal_o})...",
            flush=True,
            file=sys.stderr,
        )

        if _use_minimal_o:
            output = ttnn.experimental.minimal_matmul(
                attn_output,
                _wo,
                compute_kernel_config=self.compute_kernel_config,
                config=self._get_minimal_matmul_config(),
            )
        else:
            _o_mm_seq = min(seq_len, _oproj_thresh)
            _o_pc = (
                self._get_prefill_program_config(_o_mm_seq, int(_wo.shape[2]), int(_wo.shape[3]), label="o_proj")
                if self.TG
                else None
            )
            _o_kwargs = {}
            if _o_pc is not None:
                _o_kwargs["program_config"] = _o_pc
            output = ttnn.linear(
                attn_output,
                _wo,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                **_o_kwargs,
            )
        _sync("after O-proj linear")

        if seq_len > _oproj_thresh:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output)

        _sync("after O projection")

        _sync("before all_reduce")

        # 10. All-reduce on cluster_axis=0 (TP reduction for row-parallel O projection)
        # NOTE: Prefill runs outside trace — do NOT pass ccl to avoid async CCL ops
        # conflicting with the existing captured trace's semaphore state.
        # Use host-side reduce: rs_ag also hangs on first CCL call during prefill
        # on TG mesh (no prior kernel compilation from decode warmup).
        # Host reduce is slow but prefill is one-time per prompt.
        output = _simple_all_reduce(
            output,
            self.mesh_device,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            impl="host",
        )
        print(f"      [ATTN PREFILL] host all_reduce done", flush=True, file=sys.stderr)

        return output

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        active_batch: int | None = None,
        global_cb=None,
        sub_device_id=None,
        prefetch_qkv_pc=None,
        prefetch_oproj_pc=None,
        prefetch_qkv_in_mc=None,
        prefetch_qkv_out_mc=None,
        prefetch_oproj_in_mc=None,
        prefetch_oproj_out_mc=None,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        else:
            return self.forward_decode(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                active_batch=active_batch,
                global_cb=global_cb,
                sub_device_id=sub_device_id,
                prefetch_qkv_pc=prefetch_qkv_pc,
                prefetch_oproj_pc=prefetch_oproj_pc,
                prefetch_qkv_in_mc=prefetch_qkv_in_mc,
                prefetch_qkv_out_mc=prefetch_qkv_out_mc,
                prefetch_oproj_in_mc=prefetch_oproj_in_mc,
                prefetch_oproj_out_mc=prefetch_oproj_out_mc,
            )

    def _prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        """For TG: select tensors from the correct column chips for KV cache fill."""
        tensor_copy = ttnn.typecast(key_or_value_layer, dtype=key_or_value_layer.dtype)
        tensors = ttnn.get_device_tensors(tensor_copy)
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)
        return multi_device_tensor
