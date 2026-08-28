# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.indexer import (
    NullIndexer,
    ReuseIndexer,
    TtIndexer,
    indexer_layer_is_reused,
    resolve_has_indexer,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG, MLA_SDPA_CONFIG
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache, MlaKvCacheFormat, MlaKvCacheGeometry


class ttMLA:
    MLA_WEIGHT_NAMES = [
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "wkv_b1",
        "wkv_b2",
        "o_proj",
    ]

    @staticmethod
    def weight_names(has_output_gate: bool = False) -> list[str]:
        """Cache-file stems for this MLA flavour. ``g_proj`` is appended only for gated MLA (Kimi-K3):
        adding it to ``MLA_WEIGHT_NAMES`` unconditionally would make every existing non-gated cache
        report itself incomplete via ``check_cache_complete`` and get rebuilt."""
        return ttMLA.MLA_WEIGHT_NAMES + (["g_proj"] if has_output_gate else [])

    @staticmethod
    def check_cache_complete(
        cache_path: Path, cache_name_prefix: str, has_indexer: bool = False, has_output_gate: bool = False
    ) -> bool:
        """Check that the dense MLA weight cache files exist, plus the indexer tensorbins when sparse.

        Dense by default (preserves existing callers). When ``has_indexer=True`` the indexer cache
        (``{prefix}.indexer_*``) must also be complete — a disjoint prefix space from the dense MLA
        names, so the dense loop here never matches indexer files and vice versa
        (see ``TtIndexer.check_cache_complete``). ``has_output_gate=True`` additionally requires
        ``g_proj`` (Kimi-K3)."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for name in ttMLA.weight_names(has_output_gate):
            if not pattern_exists(f"{cache_name_prefix}.{name}*.tensorbin", "MLA"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{name}")
                return False
        if has_indexer and not TtIndexer.check_cache_complete(cache_path, cache_name_prefix):
            return False
        return True

    @staticmethod
    def _convert_and_cache_weights(
        state_dict: dict | None,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        cache_path: Path | None = None,
        device: ttnn.MeshDevice | None = None,
        kv_only: bool = False,
    ) -> dict | None:
        """
        Shared logic for converting MLA weights to ttnn with caching.

        Args:
            state_dict: Weight dict, or None/empty for cache-only loading.
            mesh_device: Mesh device reference
            config: Model config with attention dimensions
            layer_idx: Layer index for cache file naming
            sp_axis: Sequence parallel axis
            tp_axis: Tensor parallel axis
            cache_path: Cache directory path
            device: None for cache-only (build cache), mesh_device for load to device

        Returns:
            Dict of ttnn.Tensor if device is not None, else None
        """
        num_heads = config.num_attention_heads
        kv_lora_rank = config.kv_lora_rank
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        q_lora_rank = config.q_lora_rank
        hidden_size = config.hidden_size
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        use_gate = bool(getattr(config, "mla_use_output_gate", False))

        def _cache_name(name):
            return str(cache_path / f"layer_{layer_idx}.mla.{name}") if cache_path else None

        # Prepare tensors — real weights or placeholders
        if state_dict and "q_a_layernorm.weight" in state_dict:
            q_a_ln = state_dict["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            kv_a_ln = state_dict["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            q_a_proj = state_dict["q_a_proj.weight"].transpose(-2, -1)
            q_b_proj = state_dict["q_b_proj.weight"].transpose(-2, -1)
            kv_a_proj = state_dict["kv_a_proj_with_mqa.weight"].transpose(-2, -1)
            kv_b = state_dict["kv_b_proj.weight"].reshape(1, num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            wkv_b1 = kv_b[..., :qk_nope_head_dim, :].transpose(-2, -1).transpose(-2, -1)
            wkv_b2 = kv_b[..., qk_nope_head_dim:, :].transpose(-2, -1)
            o_proj = state_dict["o_proj.weight"].transpose(-2, -1)
            # Kimi-K3 output gate: [hidden, num_heads * v_head_dim], same size as o_proj transposed.
            g_proj = state_dict["g_proj.weight"].transpose(-2, -1) if use_gate else None
        else:
            q_a_ln = torch.empty(1, 1, q_lora_rank // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            kv_a_ln = torch.empty(1, 1, kv_lora_rank // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            q_a_proj = torch.empty(hidden_size, q_lora_rank)
            q_b_proj = torch.empty(q_lora_rank, num_heads * qk_head_dim)
            kv_a_proj = torch.empty(hidden_size, kv_lora_rank + qk_rope_head_dim)
            wkv_b1 = torch.empty(1, num_heads, qk_nope_head_dim, kv_lora_rank)
            wkv_b2 = torch.empty(1, num_heads, kv_lora_rank, v_head_dim)
            o_proj = torch.empty(num_heads * v_head_dim, hidden_size)
            g_proj = torch.empty(hidden_size, num_heads * v_head_dim) if use_gate else None

        # Mesh mappers
        shard_dims_tp0 = [None, None]
        shard_dims_tp0[tp_axis] = 0
        mapper_tp0 = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims_tp0)
        shard_dims_tp1 = [None, None]
        shard_dims_tp1[tp_axis] = 1
        shard_dims_tp1[sp_axis] = None
        mapper_tp1 = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims_tp1)

        mem = ttnn.DRAM_MEMORY_CONFIG if device else None

        # KV-branch weights (always loaded). The kv-only forward path only
        # needs these; the rest are gated below on `kv_only`.
        result = {
            "kv_a_layernorm": ttnn.as_tensor(
                kv_a_ln,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=_cache_name("kv_a_layernorm"),
            ),
            "kv_a_proj_with_mqa": ttnn.as_tensor(
                kv_a_proj,
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper_tp0,
                cache_file_name=_cache_name("kv_a_proj_with_mqa"),
            ),
        }
        if not kv_only:
            result.update(
                {
                    "q_a_layernorm": ttnn.as_tensor(
                        q_a_ln,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                        cache_file_name=_cache_name("q_a_layernorm"),
                    ),
                    "q_a_proj": ttnn.as_tensor(
                        q_a_proj,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=mapper_tp0,
                        cache_file_name=_cache_name("q_a_proj"),
                    ),
                    "q_b_proj": ttnn.as_tensor(
                        q_b_proj,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=mapper_tp1,
                        cache_file_name=_cache_name("q_b_proj"),
                    ),
                    "wkv_b1": ttnn.as_tensor(
                        wkv_b1,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=mapper_tp1,
                        cache_file_name=_cache_name("wkv_b1"),
                    ),
                    "wkv_b2": ttnn.as_tensor(
                        wkv_b2,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=mapper_tp1,
                        cache_file_name=_cache_name("wkv_b2"),
                    ),
                    "o_proj": ttnn.as_tensor(
                        o_proj,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=mapper_tp0,
                        cache_file_name=_cache_name("o_proj"),
                    ),
                }
            )
            if use_gate:
                # Kimi-K3 output gate. mapper_tp1 N-shards the 12288 onto the same contiguous head
                # ranges q_b_proj uses, so the gate multiply after nlp_concat_heads needs no reshape.
                result["g_proj"] = ttnn.as_tensor(
                    g_proj,
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=mapper_tp1,
                    cache_file_name=_cache_name("g_proj"),
                )

        if device is None:
            for v in result.values():
                del v
            return None
        return result

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        seq_len: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        kv_only: bool = False,
        has_indexer: bool | None = None,
    ):
        """Build TTNN cache for MLA weights using device=None (no device copy). For DSA-sparse
        variants also writes the indexer tensorbins. Fails fast if sparse mode is resolved but the
        host indexer weights are missing — never silently builds a dense-only cache for a sparse layer."""
        ttMLA._convert_and_cache_weights(
            state_dict, mesh_device, config, layer_idx, sp_axis, tp_axis, cache_path, device=None, kv_only=kv_only
        )
        # GLM-5.2 shared layers are sparse but own no indexer weights (they reuse a prior full layer's
        # top-k) -> build the MLA cache only, skip the indexer tensorbins.
        resolved_has_indexer = resolve_has_indexer(config, state_dict=state_dict, explicit=has_indexer)
        if resolved_has_indexer and not indexer_layer_is_reused(config, layer_idx):
            if not TtIndexer.has_host_weights(state_dict):
                raise ValueError(
                    f"Sparse MLA cache build for layer {layer_idx} resolved has_indexer=True but the "
                    f"state dict is missing indexer weights {TtIndexer.WEIGHT_NAMES}. Provide them or "
                    f"pass has_indexer=False."
                )
            TtIndexer.build_ttnn_cache(
                TtIndexer.extract_host_weights(state_dict), cache_path, mesh_device, config, layer_idx, sp_axis, tp_axis
            )

    def __init__(
        self,
        config: PretrainedConfig,  # TODO: figure out how to use this for GLM and DSv32
        state_dict: dict[str, torch.Tensor],
        mesh_device: ttnn.MeshDevice,
        layer_idx: int = 0,
        seq_len: int = 1024,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        topology=ttnn.Topology.Linear,
        weight_cache_path: Optional[Path] = None,
        is_chunked: bool = False,
        slot_num: int = 1,
        layer_num: int = 61,
        kv_only: bool = False,
        has_indexer: bool | None = None,
        sparse_kv_cache_format: MlaKvCacheFormat = MlaKvCacheFormat.BF16_RM,
        active_seq_len: Optional[int] = None,
        first_layer_idx: Optional[int] = None,
        tp_shard_kv: bool = False,
    ):
        # DSA indexer weights (v3.2 / GLM): extract NON-mutating, so the caller's state_dict survives
        # repeated construction / cache build+load (the old pop() emptied it on the first pass). Dense
        # v3.1 has none. Sparse capability is resolved below via resolve_has_indexer (config DSA fields /
        # host weights / complete cache) — never from the mere presence of these keys — so cache-only
        # construction stays sparse instead of silently going dense.
        idx_host = TtIndexer.extract_host_weights(state_dict)
        self.config = config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.kv_only = kv_only
        self.is_balanced = is_balanced
        self.weight_cache_path = weight_cache_path
        self.is_chunked = is_chunked
        self.max_seq_len = seq_len
        # In chunked mode ``seq_len`` is the KV-cache capacity, while ``active_seq_len`` is the fixed
        # physical activation slab.  Logical ISL may be shorter, but callers zero-pad it in this slab;
        # keeping that shape fixed is what permits init-time high-BW-gather scratch allocation and
        # program-cache reuse across rotated/partial chunks.
        if is_chunked:
            assert active_seq_len is not None, "chunked ttMLA requires the fixed physical active_seq_len"
        self.active_seq_len = active_seq_len if active_seq_len is not None else seq_len
        assert (
            self.active_seq_len <= self.max_seq_len
        ), f"active_seq_len ({self.active_seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        # KV dedup: KVPE (and indexer key) caches sharded across SP*TP. Writes pass tp_axis, reads add a
        # TP-inner all-gather leg before the SP gather. Sparse (DSA) path only; the dense forward asserts.
        self.tp_shard_kv = tp_shard_kv
        self.slot_num = slot_num
        self.layer_num = layer_num
        self.sparse_kv_cache_format = MlaKvCacheFormat(sparse_kv_cache_format or MlaKvCacheFormat.BF16_RM)
        self.kv_cache_geometry = MlaKvCacheGeometry.from_config(config)

        # DSA indexer (v3.2 / GLM): resolve sparse mode EXPLICITLY — config DSA fields, then live host
        # weights, then a complete indexer cache — never from bool(idx_host), which silently went dense
        # for cache-only construction. Resolved here (before buffer alloc + rope/attention binding) so all
        # three can key off it. Inert for dense v3.1.
        self._has_indexer = resolve_has_indexer(
            config,
            state_dict=state_dict,
            explicit=has_indexer,
            weight_cache_path=self.weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.mla",
        )

        # The RoPE op is fixed by the configured mode. It is bound AFTER self._has_indexer is resolved
        # (below), because sparse always runs the block-cyclic path (single-shot is one full-seq chunk at
        # offset 0) and so needs the indexed op even when not chunked. Dense keeps: chunked -> indexed,
        # single-shot -> rotary_embedding_llama.

        self.sp_axis = sp_axis
        self.tp_axis = tp_axis

        # Store per-matmul and SDPA config dicts keyed by local seq_len for runtime lookup
        self.mm_configs = {
            name: MLA_MATMUL_CONFIG.get(name, {})
            for name in [
                "q_a_proj",
                "q_b_proj",
                "wkv_b1",
                "kv_a_proj_with_mqa",
                "wkv_b2",
                "o_proj",
                "g_proj",  # Kimi-K3; listed unconditionally, _resolve_mm_cfg indexes with a bare []
            ]
        }
        self.sdpa_configs = MLA_SDPA_CONFIG

        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Kimi-K3 flags; both absent (-> False) on every other variant, so those paths stay identical.
        self._use_nope = bool(getattr(config, "mla_use_nope", False))
        self._use_gate = bool(getattr(config, "mla_use_output_gate", False))

        # YaRN mscale, keyed on the PRESENCE of "factor", not on `rope_scaling is not None`:
        # transformers >= 5 synthesizes a factor-less rope_scaling dict for configs that omit it (K3's
        # case), so an is-not-None guard would pass and then KeyError. Getting this wrong is a silent
        # 2x SDPA-scale error, not a crash.
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_factor = rope_scaling.get("factor")

        self.scale = self.qk_head_dim**-0.5
        if rope_factor is not None and rope_factor > 1.0:
            mscale = rope_scaling["mscale"]
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        assert not (self._use_nope and self.scale != self.qk_head_dim**-0.5), (
            f"mla_use_nope=True but rope_scaling carries a YaRN factor ({rope_factor}) that scaled "
            f"softmax to {self.scale}; a NoPE model has no positional scaling to compensate for"
        )

        self.default_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.hifi4_fp32_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.ring_sdpa_compute_grid = (
            mesh_device.compute_with_storage_grid_size().x - 1,
            mesh_device.compute_with_storage_grid_size().y,
        )

        # Create CCL object for semaphore management
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.tp_factor = mesh_device.shape[self.tp_axis]
        self.sp_factor = mesh_device.shape[self.sp_axis]
        assert (
            self.active_seq_len % self.sp_factor == 0
        ), f"active_seq_len ({self.active_seq_len}) must divide SP factor ({self.sp_factor})"
        self.active_seq_len_local = self.active_seq_len // self.sp_factor

        self.ccl_num_links = 2 if is_blackhole() else 1  # Blackhole trains 2 fabric routing planes, others 1

        # The TP high-bandwidth all-gathers operate on the fixed prefill chunk, never on the full
        # growing KV-cache. Allocate their worst-case outputs once at construction and share them
        # across serial MLA layers through TT_CCL; forward only reuses these stable addresses.
        self._q_a_latent_gather_output = None
        self._kv_stem_gather_output = None
        self._output_gate_gather_output = None
        if self.tp_factor > 1:
            if not self.kv_only:
                self._q_a_latent_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                    name="q_a_latent",
                    shape=[1, 1, self.active_seq_len_local, self.q_lora_rank],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
            self._kv_stem_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                name="kv_stem",
                shape=[1, self.tp_factor, self.active_seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            if self._use_gate and not self.kv_only:
                self._output_gate_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                    name="output_gate",
                    shape=[1, 1, self.active_seq_len_local, self.hidden_size],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )

        # Per-axis CCL topology, named symmetrically by axis. The q/kv/wo collectives run on the TP
        # axis (cluster_axis=tp_axis) and use tp_ccl_topology; the ring-attention SDPA (ring_mla /
        # ring_joint_sdpa) runs on the SP axis (cluster_axis=sp_axis) and MUST use sp_ccl_topology.
        # Conflating them deadlocks the SDPA when the two axes differ: e.g. under FABRIC_2D_TORUS_X the
        # TP axis is Ring but the SP axis has no physical wrap, so a TP-Ring topology on the SP-axis
        # SDPA waits forever on a missing wrap link. A scalar applies to both axes (preserves 1D-ring /
        # non-torus behavior).
        if isinstance(topology, tuple):
            # The tuple is (dim0, dim1); unpacking as (sp, tp) is only correct when sp_axis=0/tp_axis=1.
            # Guard it so a future sp_axis/tp_axis swap fails loudly here instead of silently cross-
            # wiring Ring onto the wrong axis (a runtime deadlock). Mirrors the sparse-path assert below.
            assert self.sp_axis == 0 and self.tp_axis == 1, "per-axis topology tuple assumes sp_axis=0, tp_axis=1"
            self.sp_ccl_topology, self.tp_ccl_topology = topology  # (sp_axis_0, tp_axis_1)
        else:
            self.sp_ccl_topology = self.tp_ccl_topology = topology

        # Ring-attention persistent buffers. Chunked prefill (ring_mla) and the standard ring
        # joint SDPA use disjoint buffer sets, so allocate only the one the configured mode needs --
        # holding both would waste DRAM. Both sets are owned once per model by TT_CCL and shared by
        # every layer's MLA (uniform across layers, scratch / no per-layer state) instead of
        # re-allocated per layer.
        #
        # kv_only (last layer) never reaches SDPA, so it needs no ring/gather buffers. Sparse (DSA) uses
        # sparse_sdpa + the transient _gather_kvpe_prefix gather — neither the ring_mla chunked scratch nor
        # the ring-joint-SDPA buffers — so it allocates none of these regardless of is_chunked.
        if kv_only or self._has_indexer:
            pass
        elif self.is_chunked:
            # Single combined gathered-KV scratch buffer for ring_mla: K and V both come from the
            # latent kvpe cache, so one (1, 1, seq_len, kvpe_dim) buffer replaces the separate
            # per-K/per-V ring-SDPA buffers (and the dummy joint tensors) used in the other mode.
            # ring_mla's single-slot gather (kv_cache_batch_idx) writes only the active cache slot
            # into gathered slot 0, so the scratch is batch-1 regardless of slot_num * layer_num.
            self._chunked_kv_buf = self.tt_ccl.get_mla_chunked_kv_buffer(
                cache_batch=1,
                seq_len=seq_len,
                kvpe_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            )
        else:
            # All-gather K/V outputs + dummy joint_q/kv/v placeholders are uniform across layers
            # (config + seq_len + mesh), so they're owned once per model by TT_CCL and shared by every
            # layer's MLA instead of re-allocated per layer. forward() reads them off self exactly as
            # before. See TT_CCL.get_mla_ring_attention_buffers.
            ring_buffers = self.tt_ccl.get_mla_ring_attention_buffers(
                seq_len=seq_len,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                qk_head_dim=self.qk_head_dim,
                v_head_dim=self.v_head_dim,
                num_heads=self.num_heads,
                tp_axis=self.tp_axis,
            )
            self.persistent_k_output_buffer = ring_buffers["persistent_k_output_buffer"]
            self.persistent_v_output_buffer = ring_buffers["persistent_v_output_buffer"]
            self.joint_q = ring_buffers["joint_q"]
            self.joint_kv = ring_buffers["joint_kv"]
            self.joint_v = ring_buffers["joint_v"]

        # Load weights to TT device. In kv_only mode the returned dict only
        # contains kv_a_layernorm / kv_a_proj_with_mqa; the Q-side / V / wo
        # weights are skipped entirely (saves DRAM + cache reads).
        weights = self._convert_and_cache_weights(
            state_dict,
            mesh_device,
            config,
            layer_idx,
            sp_axis,
            tp_axis,
            self.weight_cache_path,
            device=mesh_device,
            kv_only=kv_only,
        )
        self.kv_a_layernorm_weight = weights["kv_a_layernorm"]
        self.kv_a_proj_with_mqa_weight = weights["kv_a_proj_with_mqa"]
        if not kv_only:
            self.q_a_layernorm_weight = weights["q_a_layernorm"]
            self.q_a_proj_weight = weights["q_a_proj"]
            self.q_b_proj_weight = weights["q_b_proj"]
            self.wkv_b1_weight = weights["wkv_b1"]
            self.wkv_b2_weight = weights["wkv_b2"]
            self.o_proj_weight = weights["o_proj"]
            if self._use_gate:
                self.g_proj_weight = weights["g_proj"]
        logger.info(f"Loaded {len(weights)} weights in MLA layer {layer_idx} (kv_only={kv_only})")

        # DSA indexer (v3.2 / GLM): self._has_indexer was resolved above (before the buffer alloc). The
        # TtIndexer owns the indexer stems / RoPE tables / device key-cache and reuses this MLA's q_a stem
        # + collectives. Inert for dense v3.1.
        # DSA *family* (config carries the indexer fields), independent of whether the indexer is active
        # this layer. V3.1's dense config lacks them; V3.2's config has them even when a benchmark forces
        # the attention dense (has_indexer=False). Dense-path tuning gates that must tell V3.1 from a
        # dense-run V3.2 key on this, not _has_indexer (see _get_sdpa_program_config).
        self._is_dsa_family = TtIndexer.matches_config(config)
        self._sparse_kv_gather_buffer = None
        if self._has_indexer and not self.kv_only and self.sp_factor > 1:
            self._sparse_kv_gather_buffer = self.tt_ccl.get_mla_sparse_kv_gather_buffer(
                seq_len=seq_len,
                row_width=self.sparse_kv_cache_format.storage_width(self.kv_cache_geometry),
                dtype=self.sparse_kv_cache_format.storage_dtype,
                layout=self.sparse_kv_cache_format.storage_layout,
            )
        # GLM-5.2 indexer reuse: a "shared" layer is sparse but owns no indexer weights — it reuses the
        # most recent "full" layer's top-k indices, injected at forward, and binds a weight-less
        # ReuseIndexer (never computes). Absent indexer_types (v3.1 / v3.2 / GLM-5.1) every layer is
        # "full" -> current behavior, unchanged.
        self._indexer_reuse = indexer_layer_is_reused(config, layer_idx)
        if self._has_indexer:
            # The indexer assumes natural-order SP sharding (contiguous per-chip query blocks: its
            # device RoPE and the indexer_score per-device causal offset both index positions as
            # start_pos + sp_rank*S_local). The balanced chunk reorder breaks that, so guard it.
            assert not self.is_balanced, "DSA indexer requires is_balanced=False (natural-order SP sharding)"
            if self._indexer_reuse:
                self._indexer = ReuseIndexer()  # shared layer: reused indices injected at forward
            else:
                # TtIndexer warns (does not raise) if given neither host weights nor a complete cache —
                # mirroring dense MLA's lenient placeholder load, but loudly. The layer still stays sparse
                # (binds TtIndexer), so it never silently falls back to dense.
                self._indexer = TtIndexer(
                    idx_host if idx_host else None,  # None → TtIndexer loads cache-only placeholders
                    config=config,
                    mesh_device=self.mesh_device,
                    sp_axis=self.sp_axis,
                    tp_axis=self.tp_axis,
                    default_compute_kernel_config=self.default_compute_kernel_config,
                    hifi4_fp32_compute_kernel_config=self.hifi4_fp32_compute_kernel_config,
                    weight_cache_path=self.weight_cache_path,
                    layer_idx=self.layer_idx,
                    tt_ccl=self.tt_ccl,
                    ccl_num_links=self.ccl_num_links,
                    sp_ccl_topology=self.sp_ccl_topology,
                    tp_ccl_topology=self.tp_ccl_topology,
                    seq_len=seq_len,
                    active_seq_len=self.active_seq_len,
                    slot_num=slot_num,
                    layer_num=self.layer_num,
                    first_layer_idx=first_layer_idx,
                    tp_shard_kv=self.tp_shard_kv,
                )
        else:
            self._indexer = NullIndexer()  # dense v3.1: forward calls .forward() -> None (dense path)
            self._indexer_reuse = False

        # Bind the RoPE op now that self._has_indexer is known: sparse always uses the indexed
        # (block-cyclic) op — single-shot is folded onto the block-cyclic path as one full-seq chunk at
        # offset 0 — so its key cache persists layer-stacked (migratable to decode). Dense: chunked ->
        # indexed, single-shot -> rotary_embedding_llama.
        # NoPE (Kimi-K3) binds a pass-through: the op is dropped but the nope/rope slices stay (they
        # are dimension-driven), so the cached latent is still 576 wide and rope_tensors goes unused.
        if self._use_nope:
            assert not self._has_indexer, (
                "mla_use_nope with a DSA indexer is not supported: TtIndexer applies its own rope to "
                "the indexer queries/keys, so the rope tensors cannot be dropped wholesale"
            )
            self._apply_rope = self._apply_rope_none
        else:
            self._apply_rope = (
                self._apply_rope_padded if (self.is_chunked or self._has_indexer) else self._apply_rope_one_shot
            )

        # Bind the attention core once, by config. Sparse ALWAYS uses the block-cyclic
        # _sparse_chunked_attn (single-shot = one full-seq chunk); dense splits by chunking. forward()
        # then calls self._attention(...) with no mode ladder: the decision is made here, not per call.
        if self._has_indexer:
            self._attention = self._sparse_chunked_attn
        else:
            self._attention = self._dense_chunked_attn if self.is_chunked else self._dense_single_attn

    @staticmethod
    def kv_cache_to_host(kvpe_cache: MlaKvCache, mesh_device: ttnn.MeshDevice, sp_axis: int = 0):
        """Read and decode the logical KVPE cache in natural SP order."""
        host = ttnn.to_torch(
            kvpe_cache.storage,
            mesh_composer=ttnn.create_mesh_composer(
                mesh_device,
                config=ttnn.MeshComposerConfig(
                    dims=(2, -1),
                    mesh_shape_override=ttnn.MeshShape(
                        mesh_device.shape[sp_axis],  # concat SP shards
                        1,  # collapse TP replicas
                    ),
                ),
            ),
        )
        return kvpe_cache.unpack_host(host)

    def get_weight_shapes(self) -> dict[str, tuple]:
        shapes = {
            "kv_a_proj_with_mqa.weight": tuple(self.kv_a_proj_with_mqa_weight.shape),
            "kv_a_layernorm.weight": tuple(self.kv_a_layernorm_weight.shape),
        }
        if not self.kv_only:
            shapes.update(
                {
                    "q_a_proj.weight": tuple(self.q_a_proj_weight.shape),
                    "q_a_layernorm.weight": tuple(self.q_a_layernorm_weight.shape),
                    "q_b_proj.weight": tuple(self.q_b_proj_weight.shape),
                    "wkv_b1_weight": tuple(self.wkv_b1_weight.shape),
                    "wkv_b2_weight": tuple(self.wkv_b2_weight.shape),
                    "o_proj.weight": tuple(self.o_proj_weight.shape),
                }
            )
        return shapes

    # Default output dtypes per weight, used when no tuned config exists for the seq_len_local
    MM_DEFAULT_DTYPES = {
        "q_a_proj": ttnn.bfloat16,
        "q_b_proj": ttnn.bfloat16,
        "wkv_b1": ttnn.bfloat16,
        "kv_a_proj_with_mqa": ttnn.bfloat16,
        "wkv_b2": ttnn.bfloat8_b,
        "o_proj": ttnn.bfloat16,
        "g_proj": ttnn.bfloat16,  # Kimi-K3; this dict is a KeyError not a fallback (see _get_mm_kwargs)
    }

    # Matmul dimensions for batched matmuls (wkv_b1 / wkv_b2) keyed by weight name.
    # Each entry: (K_attr, N_attr) where values are attribute names on self.
    _BATCHED_MM_DIMS = {
        "wkv_b1": ("qk_nope_head_dim", "kv_lora_rank"),
        "wkv_b2": ("kv_lora_rank", "v_head_dim"),
    }

    def _cfg_matches(self, cfg: dict) -> bool:
        """Do this tuned config's declared gating tags match this live ttMLA?

        Tags are declared in mla_config.py; only the match is resolved here, because it depends on this
        instance (``chunked_only`` especially -- the static config can't know it). Shared by the matmul
        and SDPA resolvers so a new tag can't be honoured by one and ignored by the other.
        """
        # Some tuned configs are head-count specific (the chunked-prefill 640 set was tuned for Kimi's
        # 64 heads; several program_configs overflow the grid at DeepSeek's 128). A config may declare
        # the num_heads it was tuned for; when it doesn't match this model, fall back so a different
        # variant at the same seq_len_local doesn't pick up a dimensionally-invalid program_config.
        if cfg.get("num_heads") not in (None, self.num_heads):
            return False
        # Some of those configs are additionally q_lora_rank-specific: the 640 set's program_configs are
        # dimensionally valid at Kimi's q_lora_rank (1536) but overflow the grid at GLM-5.1's (2048), even
        # though both have 64 heads. When a config declares a q_lora_rank that doesn't match this model,
        # fall back so a same-heads/same-seq variant doesn't pick up an invalid program_config.
        if cfg.get("q_lora_rank") not in (None, self.q_lora_rank):
            return False
        # The chunked-prefill 640 set is only dimensionally valid in chunked mode (e.g. wkv_b1/wkv_b2
        # are true batched per-head matmuls over the per-head SDPA output; the single-shot path applies
        # them to a batch=1 latent). Fall back to defaults when this ttMLA was not built for chunked.
        if cfg.get("chunked_only") and not self.is_chunked:
            return False
        # Dense-path head ceiling: above dense_head_cap_non_dsa, non-DSA models fall back rather than
        # use this tiling. Empirically derived (V3.1 at 128 heads); DSA-family models are exempt.
        cap = cfg.get("dense_head_cap_non_dsa")
        if cap is not None and self.num_heads > cap and not self._is_dsa_family:
            return False
        return True

    def _select_cfg(self, entry) -> dict | None:
        """Pick the first tuned config whose tags match, from a single dict or a list of candidates.

        A slot holds several candidates when variants share a seq_len (Kimi-K2.6 at 64 heads and K3 at
        96 both want ``640``), because the tags only reject -- they cannot choose. List order is
        priority order; most specific first.
        """
        if entry is None:
            return None
        candidates = entry if isinstance(entry, (list, tuple)) else (entry,)
        return next((cfg for cfg in candidates if self._cfg_matches(cfg)), None)

    def _resolve_mm_cfg(self, weight_name: str, seq_len_local: int) -> dict | None:
        """Resolve the tuned matmul config for this weight/seq_len, applying the gating tags.
        Returns None when no tuned config applies (caller falls back to defaults)."""
        if not is_blackhole():
            return None
        return self._select_cfg(self.mm_configs[weight_name].get(seq_len_local))

    def _get_act_mem_config(self, weight_name: str, seq_len_local: int) -> ttnn.MemoryConfig:
        """Memory config for the activation (in0) feeding this weight's matmul, as tuned in the mm
        config (act_mem_config). Defaults to DRAM when no tuned config applies."""
        cfg = self._resolve_mm_cfg(weight_name, seq_len_local)
        return cfg["act_mem_config"] if cfg is not None else ttnn.DRAM_MEMORY_CONFIG

    def _get_mm_kwargs(self, weight_name: str, seq_len_local: int) -> dict:
        """Get matmul kwargs from config, falling back to defaults."""
        cfg = self._resolve_mm_cfg(weight_name, seq_len_local)
        if cfg is None:
            if weight_name in self._BATCHED_MM_DIMS:
                return self._make_batched_mm_kwargs(weight_name, seq_len_local)
            return {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": self.MM_DEFAULT_DTYPES[weight_name]}
        return {
            "memory_config": cfg["out_mem_config"],
            "program_config": cfg["program_config"],
            "dtype": cfg["out_dtype"],
        }

    def _make_batched_mm_kwargs(self, weight_name: str, seq_len_local: int) -> dict:
        """Build MatmulMultiCoreReuseMultiCast1DProgramConfig for batched matmuls (wkv_b1/wkv_b2).

        These matmuls require fuse_batch=False and mcast_in0=False to support
        batch broadcasting (in0 batch=1, in1 batch=num_heads).
        """
        k_attr, n_attr = self._BATCHED_MM_DIMS[weight_name]
        K_tiles = getattr(self, k_attr) // ttnn.TILE_SIZE
        N_tiles = getattr(self, n_attr) // ttnn.TILE_SIZE
        M_tiles = seq_len_local // ttnn.TILE_SIZE

        num_cores = self.ring_sdpa_compute_grid[0] * self.ring_sdpa_compute_grid[1]
        per_core_M = max(1, -(-M_tiles // num_cores))  # ceil division
        while M_tiles % per_core_M != 0:
            per_core_M += 1

        # out_subblock: h * w <= 8, h divides per_core_M, w divides N_tiles
        out_subblock_w = min(N_tiles, 8)
        while N_tiles % out_subblock_w != 0:
            out_subblock_w -= 1
        out_subblock_h = min(per_core_M, 8 // out_subblock_w)
        while per_core_M % out_subblock_h != 0:
            out_subblock_h -= 1

        # in0_block_w: factor of K_tiles, capped for L1 pressure
        in0_block_w = min(4, K_tiles)
        while K_tiles % in0_block_w != 0:
            in0_block_w -= 1

        return {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "dtype": self.MM_DEFAULT_DTYPES[weight_name],
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_M,
                per_core_N=N_tiles,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=False,
            ),
        }

    def _get_sdpa_program_config(self, seq_len_local: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA program config, falling back to default chunk sizes."""
        # Same gating tags as the matmul configs, via the shared _cfg_matches. dense_head_cap_non_dsa
        # is consumed only here (sparse V3.2/GLM go through sparse_sdpa), and keys on the DSA family
        # because V3.1 and V3.2 are dimensionally identical. It is an EMPIRICAL guard -- V3.1 OOMs L1 at
        # k=640 above it -- NOT "L1 scales with head count": the program factory sizes every CB from
        # Sq_chunk_t / Sk_chunk_t / DHt with no num_heads term. Do not extend it to a new model on that
        # reasoning; measure it (test_ring_joint_sdpa.py's k_chunk_sizes sweeps).
        cfg = self._select_cfg(self.sdpa_configs.get(seq_len_local))
        q_chunk_size = cfg["q_chunk_size"] if cfg else 32
        k_chunk_size = cfg["k_chunk_size"] if cfg else 32
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

    def _apply_rope_padded(
        self, t: ttnn.Tensor, rope_tensors: dict, kv_actual_isl: int, metadata: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        """Chunked rotated RoPE via the indexed op. rope_tensors carry the whole-cache,
        block-cyclic-sharded cos/sin (built once via RotarySetup.get_rope_tensors_indexed); the op
        derives this chunk's per-chip shard offset on-device from kv_actual_global -- the same
        update_idxt math the KV-cache writer uses, keeping rotation and cache write consistent.

        Per-element-tensor (trace-safe) path: `metadata` is a 3-tuple of 1-element uint32 tensors
        (slot_id, actual_start, actual_end); rope reads kv_actual_global = actual_start = metadata[1].
        """
        if metadata is not None:
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                t,
                rope_tensors["cos_matrix"],
                rope_tensors["sin_matrix"],
                rope_tensors["trans_matrix"],
                metadata[1],  # actual_start = kv_actual_global (1-element tensor)
                cluster_axis=self.sp_axis,
            )
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            t,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )

    def _apply_rope_none(
        self,
        t: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int] = None,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """NoPE (Kimi-K3): identity -- position comes from the KDA layers, so the 64 'rope' columns are
        a non-positional key channel. Signature must track _apply_rope_padded / _apply_rope_one_shot:
        _q_stem / _kv_stem call whichever is bound with the full arg list, so a new kwarg there has to
        be accepted and ignored here or a NoPE forward raises TypeError."""
        return t

    def _apply_rope_one_shot(
        self,
        t: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int] = None,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Single-shot RoPE: natural-order rope_tensors + rotary_embedding_llama. (metadata unused --
        single-shot has no chunked rotation; accepted so callers can pass it uniformly.)"""
        return ttnn.experimental.rotary_embedding_llama(
            t,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

    def _chunked_attn(
        self,
        *,
        tt_q: ttnn.Tensor,
        tt_kvpe: ttnn.Tensor,
        kvpe_cache: MlaKvCache,
        kv_actual_isl: int,
        cache_batch_idx: int,
        cache_layer_idx: int,
        cache_user_id: int,
        seq_len_local: int,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Chunked-prefill attention via update_padded_kv_cache + ring_mla.

        Unified path for both rotated (kv_actual_isl mid-slab) and chunk-aligned prefill: the
        chunk-aligned case is the degenerate kv_actual_isl = n * chunk_size_global, where
        update_padded_kv_cache reduces to a uniform per-chip write and the indexed rope/SDPA reduce
        to natural order. K and V both come from the single latent kvpe cache -- ring_mla reads V as
        the first kv_lora_rank columns of KV and materializes it in-op, so wkv_b2 is applied to the
        compact (kv_lora_rank-wide) attention output afterwards. Returns attn_out in v_head_dim space.
        """
        assert not self.is_balanced, "chunked prefill currently requires is_balanced=False"

        tile_size = ttnn.TILE_SIZE
        chunk_size_global = seq_len_local * self.sp_factor
        assert chunk_size_global % (tile_size * self.sp_factor) == 0, (
            f"chunk_size_global ({chunk_size_global}) must be a multiple of "
            f"TILE_SIZE * sp_factor ({tile_size * self.sp_factor})"
        )
        # Metadata path: kv_actual_isl is read on-device from metadata[1] and may be omitted host-side.
        assert (
            metadata is not None or kv_actual_isl % tile_size == 0
        ), f"kv_actual_isl ({kv_actual_isl}) must be tile-aligned"

        # Write this chunk into the cache. update_padded_kv_cache derives each chip's local write
        # offset on-device from kv_actual_global (chunk-aligned kv_actual -> uniform per-chip write).
        # The dense ring_mla cache is still TP-replicated; only _sparse_chunked_attn is TP-dedup wired.
        assert not self.tp_shard_kv, "tp_shard_kv is only supported on the sparse (DSA) path, not dense ring_mla"
        # Metadata (trace-safe) path reads slot_idx/kv_actual_global on-device from the metadata tensor.
        self._update_kv_cache(
            kvpe_cache,
            tt_kvpe,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            metadata=metadata,
        )

        # K and V are the single latent kvpe cache (V = first kv_lora_rank columns, materialized
        # in-op). logical_n = prior valid length + this chunk; cache_batch_idx selects this
        # user/layer's slot; kv_actual_isl drives the on-device rotation/causality offset.
        #
        # Trace-safe metadata path: ring_mla reads its per-chunk scalars on-device -- the all-gather +
        # SDPA readers take the cache slot from metadata[0] and derive logical_nt / masks from
        # kv_actual_isl = metadata[1]. Every kernel derives logical_n on-device on this path, so the host
        # logical_n is a placeholder = global cache capacity. metadata[0] holds only the user slot, so pass
        # the per-layer factor (kv_cache_num_layers/kv_cache_layer_idx) so the readers recompute the full
        # (user, layer) slot on-device -- otherwise every layer would read layer 0's KV cache.
        if metadata is not None:
            meta_slot_kwargs = {
                "slot_id": metadata[0],
                "kv_actual_isl_tensor": metadata[1],
                "kv_cache_num_layers": self.layer_num,
                "kv_cache_layer_idx": cache_layer_idx,
            }
            ring_logical_n = kvpe_cache.storage.shape[2] * self.sp_factor  # global cache capacity
        else:
            meta_slot_kwargs = {"kv_cache_batch_idx": cache_batch_idx, "kv_actual_isl": kv_actual_isl}
            ring_logical_n = kv_actual_isl + chunk_size_global
        attn_out, _ = ttnn.transformer.ring_mla(
            tt_q,
            kvpe_cache.storage,
            persistent_output_buffer_kv=self._chunked_kv_buf,
            head_dim_v=self.kv_lora_rank,
            logical_n=ring_logical_n,
            program_config=self._get_sdpa_program_config(seq_len_local),
            scale=self.scale,
            compute_kernel_config=self.default_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=self.ccl_num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh_device,
            topology=self.sp_ccl_topology,
            ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_balanced=self.is_balanced,
            **meta_slot_kwargs,
        )

        # ring_mla output is in kv_lora_rank (latent V) space; expand to v_head_dim per head. Unlike the
        # single-shot path this in0 is the per-head SDPA output (batch=local_heads), so the tuned 640
        # config is a true batched MatmulMultiCoreReuse. When no tuned config matches (non-Kimi variant
        # or non-blackhole) _get_mm_kwargs falls back to the 1D batched default.
        # NOTE: Input is ideally L1 but DRAM comes from SDPA
        attn_out = ttnn.linear(
            attn_out,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )
        return attn_out

    def _q_a_latent(
        self, hidden_states: ttnn.Tensor, seq_len_local: int, norm_memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """q_a projection + TP all-reduce + q_a_layernorm → the q_a latent (qr). Computed once per
        layer and shared: _q_stem consumes it for q_b_proj, and (when present) TtIndexer.forward reads
        it for the indexer queries — so the sparse path no longer recomputes the q_a stem."""
        # NOTE: input is ideally L1 for chunked, but hidden states memory config is set outside the module
        qr = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_a_proj", seq_len_local),
        )

        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            qr = ttnn.experimental.reduce_scatter_minimal_async(
                qr,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            assert seq_len_local == self.active_seq_len_local, (
                f"q_a latent gather was preallocated for {self.active_seq_len_local} local tokens, "
                f"got {seq_len_local}"
            )
            qr = ttnn.experimental.high_bw_all_gather(
                qr,
                dim=3,
                output_tensor=self._q_a_latent_gather_output,
                cluster_axis=self.tp_axis,
                num_links=self.ccl_num_links,
            )

        return ttnn.rms_norm(
            qr,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=norm_memory_config,
            compute_kernel_config=self.default_compute_kernel_config,
        )

    def _q_stem(
        self,
        qr: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int],
        seq_len_local: int,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Absorbed-Q stem from the q_a latent: q_b_proj → heads → split → wkv_b1(nope) → RoPE(rope)
        → concat. Consumes qr (the indexer, if any, has already read it by this point)."""
        num_heads_local = self.num_heads // self.tp_factor
        tt_q = ttnn.linear(
            qr,
            self.q_b_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_b_proj", seq_len_local),
        )
        ttnn.deallocate(qr)

        # convert to
        # [batch (1), num_heads_local, seq_len_local, qk_head_dim]
        tt_q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            tt_q,
            num_heads=num_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # TODO: split rope and nope, workaround remove with ttnn.narrow or fusion
        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads_local, seq_len_local, self.qk_nope_head_dim])
        tt_q_rope = ttnn.slice(
            tt_q, [0, 0, 0, self.qk_nope_head_dim], [1, num_heads_local, seq_len_local, self.qk_head_dim]
        )
        ttnn.deallocate(tt_q)

        tt_q_nope = ttnn.linear(
            tt_q_nope,
            self.wkv_b1_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b1", seq_len_local),
        )

        tt_q_rope = self._apply_rope(tt_q_rope, rope_tensors, kv_actual_isl, metadata=metadata)

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)
        ttnn.deallocate(tt_q_nope)
        ttnn.deallocate(tt_q_rope)
        return tt_q

    def _kv_stem(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int],
        seq_len_local: int,
        return_kv_intermediates: bool,
        kvpe_cache: MlaKvCache,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> tuple[ttnn.Tensor, Optional[ttnn.Tensor], dict | None]:
        """Shared KV stem.

        Returns tt_kvpe in the persistent cache representation. The returned value is both written to the
        cache and consumed by attention without a decode/re-encode round trip.
        """
        # NOTE: input is ideally L1 for chunked, but hidden states memory config is set outside the module
        kv_mm_kwargs = self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local)
        # high_bw_all_gather directly streams its source from DRAM. Kimi's 640-token
        # matmul tune otherwise returns L1, while the fixed-slab TP path is active.
        # Request DRAM from the producer instead of adding a forward-time conversion
        # or falling back to the legacy all-gather.
        if self.tp_factor > 1:
            kv_mm_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        tt_kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **kv_mm_kwargs,
        )

        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            assert seq_len_local == self.active_seq_len_local, (
                f"KV stem gather was preallocated for {self.active_seq_len_local} local tokens, " f"got {seq_len_local}"
            )
            tt_kv = ttnn.experimental.high_bw_all_gather(
                tt_kv,
                dim=1,
                output_tensor=self._kv_stem_gather_output,
                cluster_axis=self.tp_axis,
                num_links=self.ccl_num_links,
            )
            tt_kv = ttnn.experimental.fast_reduce_nc(
                tt_kv, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )

        # Raw compressed KV (pre-norm/pre-rope, [.., 576]) for debug/PCC against golden traces.
        kv_intermediates = {"tt_kv": ttnn.clone(tt_kv)} if return_kv_intermediates else None

        # TODO: split rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len_local, self.kv_lora_rank])
        tt_kv_rope = ttnn.slice(
            tt_kv, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim]
        )
        ttnn.deallocate(tt_kv)

        tt_kv_nope = ttnn.rms_norm(
            tt_kv_nope,
            weight=self.kv_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl, metadata=metadata)

        if return_kv_intermediates:
            # post-RMSNorm latent ([.., 512]) and post-RoPE k_pe ([.., 64]); clone before concat.
            kv_intermediates["tt_kv_nope"] = ttnn.clone(tt_kv_nope)
            kv_intermediates["tt_kv_rope"] = ttnn.clone(tt_kv_rope)

        tt_kvpe = kvpe_cache.pack(tt_kv_nope, tt_kv_rope, intermediates=kv_intermediates)
        ttnn.deallocate(tt_kv_rope)
        if self._has_indexer:
            # Sparse attention reads latent V from the cache; only dense attention needs this standalone tensor.
            ttnn.deallocate(tt_kv_nope)
            tt_kv_nope = None

        return tt_kvpe, tt_kv_nope, kv_intermediates

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        return ttnn.linear(
            t,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )

    def _write_kvpe(self, kvpe_cache: MlaKvCache, tt_kvpe: ttnn.Tensor, cache_layer_idx: int) -> None:
        """DENSE single-shot cache fill: write this layer's whole kvpe slot (bf8/TILE, already in the
        cache's dtype/layout via MlaKvCache.pack). Chunked modes (dense + sparse) and sparse single-shot
        write through update_padded_kv_cache instead; only dense single-shot still uses this TILE-only
        fill_cache_for_user_ primitive.

        TODO: unify dense single-shot onto update_padded_kv_cache too (one write op model-wide, drop this
        helper). Blocked on the single-chip case: test_prefill_block_loop[mesh-1x1] runs dense single-shot
        on a (1,1) mesh, where fill_cache_for_user_ is mesh-agnostic but update_padded_kv_cache's SP
        cluster_axis / block-cyclic / tile-aligned-kv_actual_global path is not yet validated. Confirm
        update_padded handles 1x1 (and sp=1), then switch _dense_single_attn onto it too (the sparse path
        already folded its single-shot onto the block-cyclic update_padded write)."""
        ttnn.kv_cache.fill_cache_for_user_(kvpe_cache.storage, tt_kvpe, cache_layer_idx)

    def _update_kv_cache(
        self,
        cache: MlaKvCache,
        values: ttnn.Tensor,
        *,
        cache_user_id: int,
        cache_layer_idx: int,
        kv_actual_isl: int,
        metadata: Optional[ttnn.Tensor] = None,
        tp_axis: Optional[int] = None,
    ) -> None:
        # TP-sharded writes need the host kv_actual_global (the reader derives its 1/tp window from it),
        # which the metadata path cannot supply. The op re-checks; assert here for a readable failure.
        assert not (metadata is not None and tp_axis is not None), (
            "tp_shard_kv is not supported on the metadata (traced) write path -- run with PREFILL_USE_TRACE=0, "
            "or with PREFILL_KV_ONLY_LAST_LAYER=0 so the kv-only layer does not take the metadata path."
        )
        # Metadata (trace-safe) path: slot_idx (metadata[0]) + kv_actual_global (metadata[1]) read
        # on-device, each its own 1-element tensor. Scalar path passes host slot/kv_actual_global.
        if metadata is not None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache.storage,
                values,
                metadata[0],  # slot_idx tensor
                metadata[1],  # kv_actual_global tensor
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                cluster_axis=self.sp_axis,
                tp_axis=tp_axis,
            )
        else:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache.storage,
                values,
                slot_idx=cache_user_id,
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                kv_actual_global=kv_actual_isl,
                cluster_axis=self.sp_axis,
                tp_axis=tp_axis,
            )

    def _output_gate(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """Kimi-K3 gated MLA: sigmoid(g_proj(hidden_states)), head-sharded to match concat_heads.

        hidden_states is TP-fractured and g_proj is full-rank, so one collective is unavoidable.
        All-gathering the activation and N-sharding the weight beats K-sharding + reduce-scattering the
        12288-wide partial: less traffic, no wide intermediate, and g is complete per device so sigmoid
        can fuse into the matmul (measured ~292 us less collective at FABRIC_2D).
        """
        if self.tp_factor > 1:
            assert self._output_gate_gather_output is not None
            assert seq_len_local == self.active_seq_len_local, (
                f"output-gate gather was preallocated for {self.active_seq_len_local} local tokens, "
                f"got {seq_len_local}"
            )
            h = ttnn.experimental.high_bw_all_gather(
                hidden_states,
                dim=3,
                output_tensor=self._output_gate_gather_output,
                cluster_axis=self.tp_axis,
                num_links=self.ccl_num_links,
            )
        else:
            h = hidden_states
        g = ttnn.linear(
            h,
            self.g_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("g_proj", seq_len_local),
        )
        # The TP gather aliases a construction-time persistent output buffer shared across serial MLA
        # instances/layers. Keep it allocated; deallocating the returned wrapper invalidates later users.
        # Fused only when a tuned config supplied fused_activation; otherwise a standalone sigmoid.
        # Keyed off the resolved config so the two paths can't silently drift into double-sigmoid.
        if not self._gate_sigmoid_fused(seq_len_local):
            g = ttnn.sigmoid(g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return g

    def _gate_sigmoid_fused(self, seq_len_local: int) -> bool:
        """True when the tuned g_proj config already applies sigmoid via fused_activation. getattr, not
        attribute access: only the multicast program-config classes expose ``fused_activation``."""
        cfg = self._resolve_mm_cfg("g_proj", seq_len_local)
        if cfg is None:
            return False
        return getattr(cfg.get("program_config"), "fused_activation", None) is not None

    def _o_proj_epilogue(
        self, attn_out: ttnn.Tensor, seq_len_local: int, hidden_states: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        """Shared nlp_concat_heads -> (K3 gate) -> o_proj -> TP reduce-scatter epilogue.

        The gate multiply sits after nlp_concat_heads so g needs no head split, and cannot move before
        wkv_b2: it acts in v_head_dim space, and g*(attn @ W_b2) != (g*attn) @ W_b2.
        """
        v_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self._use_gate:
            assert hidden_states is not None, "gated MLA needs hidden_states to compute g_proj"
            g = self._output_gate(hidden_states, seq_len_local)
            v_out = ttnn.multiply(v_out, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
        v_out = ttnn.linear(
            v_out,
            self.o_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("o_proj", seq_len_local),
        )
        if self.tp_factor > 1:
            return ttnn.experimental.reduce_scatter_minimal_async(
                v_out,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
        return v_out

    # Expects activation in form of:
    # [1, batch_size == 1, seq_len // sp_factor, hidden_size // tp_factor]
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: MlaKvCache,
        cache_layer_idx: int = 0,
        actual_start: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
        return_indexer_indices: bool = False,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, Optional[dict]]":
        # Trace-safe metadata path: a 3-tuple of 1-element uint32 DRAM tensors (slot_id, actual_start,
        # actual_end) passed in from outside. When provided, the chunked-prefill ops (update_padded_kv_cache,
        # rotary_embedding_indexed, ring_mla) read their per-chunk scalars on-device from it instead of from
        # host actual_start/cache_user_id. Threaded through verbatim -- ttMLA never reads/reconstructs it.
        #
        # Chunked-prefill mode is fixed at construction: self.is_chunked drives buffer allocation in
        # __init__ and the rope variant, and forward honors that flag -- it does not infer the mode from
        # the arguments. actual_start is the chunk parameter, supplied iff chunked (EXCEPT the metadata
        # path, where the per-chunk scalars are read on-device and host actual_start may be None).
        assert metadata is not None or (actual_start is not None) == self.is_chunked, (
            f"actual_start ({'set' if actual_start is not None else 'None'}) does not match construction "
            f"(self.is_chunked={self.is_chunked}); pass actual_start iff built with is_chunked=True"
        )
        if kvpe_cache.geometry != self.kv_cache_geometry:
            raise ValueError(f"MLA configured for KV geometry {self.kv_cache_geometry}, got {kvpe_cache.geometry}")

        if self.kv_only:
            return self._forward_kv_only(
                hidden_states,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx,
                kv_actual_isl=actual_start or 0,
                cache_user_id=cache_user_id,
                index_kv_cache=index_kv_cache,
                metadata=metadata,
            )

        seq_len_local = hidden_states.shape[2]
        kv_actual_isl = actual_start

        # Sparse always runs the block-cyclic path (indexed rope + kvpe cache read-back), which treats
        # single-shot as one full-seq chunk at offset 0. Coerce the None single-shot offset to 0 so the
        # indexed rope op, the cache write, and the indexer all get a concrete kv_actual_global.
        if self._has_indexer and kv_actual_isl is None:
            kv_actual_isl = 0

        if self._has_indexer:
            assert (
                kvpe_cache.format == self.sparse_kv_cache_format
            ), f"MLA configured for {self.sparse_kv_cache_format}, got {kvpe_cache.format} cache"

        signpost(header="MLA_START")

        # q-norm output uses the tuned activation memory_config in every mode (dense and sparse);
        # the next op (q_b_proj) is the same matmul regardless of attention path.
        q_norm_mem_config = self._get_act_mem_config("q_b_proj", seq_len_local)
        # Compute the q_a latent once and share it: the DSA indexer reads it for its queries, then
        # _q_stem consumes it for q_b_proj — so the sparse path does not recompute the q_a stem.
        qr = self._q_a_latent(hidden_states, seq_len_local, q_norm_mem_config)

        # DSA dispatch (v3.2 / GLM): the op graph is fixed by CONFIG — self._indexer and self._attention
        # are bound once at construction (sparsity × chunking), never by the runtime sequence length.
        # Host-scalar cache slot/start/length arguments are still rebound on each eager dispatch; a captured
        # device trace therefore remains specific to the values used during capture. (At seq_len <=
        # index_topk the indexer top-k simply selects all available causal keys, so sparse is numerically
        # equal to dense there.) The indexer's forward also writes its K-cache (a no-op on the dense
        # null-indexer), so no separate warm-up write is needed.
        # GLM-5.2 reuse: a shared layer receives a prior full layer's top-k indices and skips its own
        # indexer (its ReuseIndexer.forward would raise). Absent injection -> compute as usual.
        indices = (
            indexer_indices
            if indexer_indices is not None
            else self._indexer.forward(
                hidden_states,
                qr,
                seq_len_local,
                start_pos=kv_actual_isl or 0,
                rope_tensors=rope_tensors,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                index_kv_cache=index_kv_cache,
            )
        )

        tt_q = self._q_stem(qr, rope_tensors, kv_actual_isl, seq_len_local, metadata=metadata)
        tt_kvpe, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            kvpe_cache,
            metadata=metadata,
        )

        attn_out = self._attention(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            tt_kv_nope=tt_kv_nope,
            indices=indices,
            kvpe_cache=kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            kv_actual_isl=kv_actual_isl,
            metadata=metadata,
        )

        out = self._o_proj_epilogue(attn_out, seq_len_local, hidden_states=hidden_states)
        signpost(header="MLA_END")
        # ``indices`` survives _sparse_mla (it deallocs only re-sharded copies), so it is safe to return
        # for a "full" layer to hand to downstream "shared" layers (GLM-5.2 reuse).
        if return_kv_intermediates and return_indexer_indices:
            return out, kv_intermediates, indices
        if return_kv_intermediates:
            return out, kv_intermediates
        if return_indexer_indices:
            return out, indices
        return out

    # Attention core variants, one bound to self._attention at construction (sparsity × chunking).
    # All four share the forward() call signature and ignore the kwargs they don't need (**_), so the
    # bound name can be invoked uniformly. Bodies are the former mode-ladder branches, unchanged.

    def _dense_single_attn(self, *, tt_q, tt_kvpe, tt_kv_nope, kvpe_cache, cache_layer_idx, seq_len_local, **_):
        # Single-shot prefill: materialize V before causal ring SDPA.
        self._write_kvpe(kvpe_cache, tt_kvpe, cache_layer_idx)
        tt_v_embedding = self._apply_wkv_b2(tt_kv_nope, seq_len_local)
        attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_kvpe,
            tt_v_embedding,
            self.joint_q,
            self.joint_kv,
            self.joint_v,
            persistent_output_buffer_k=self.persistent_k_output_buffer,
            persistent_output_buffer_v=self.persistent_v_output_buffer,
            joint_strategy="rear",
            logical_n=seq_len_local * self.sp_factor,
            program_config=self._get_sdpa_program_config(seq_len_local),
            compute_kernel_config=self.default_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=self.ccl_num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh_device,
            topology=self.sp_ccl_topology,
            ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=self.scale,
            is_balanced=self.is_balanced,
        )
        return attn_out

    def _cache_batch_idx(self, cache_user_id: int, cache_layer_idx: int) -> int:
        """Flat KVPE-cache slot for (user, layer). The cache batch dim is user-major: each user reserves
        self.layer_num contiguous slots, so the flat slot is cache_user_id * layer_num + cache_layer_idx.
        Shared by the dense (ring_mla) and sparse (sparse_sdpa) chunked paths."""
        assert cache_user_id < self.slot_num, f"cache_user_id {cache_user_id} >= slot_num {self.slot_num}"
        return cache_user_id * self.layer_num + cache_layer_idx

    def _dense_chunked_attn(
        self,
        *,
        tt_q,
        tt_kvpe,
        kvpe_cache,
        kv_actual_isl,
        cache_layer_idx,
        cache_user_id,
        seq_len_local,
        metadata=None,
        **_,
    ):
        cache_batch_idx = self._cache_batch_idx(cache_user_id, cache_layer_idx)
        return self._chunked_attn(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            kvpe_cache=kvpe_cache,
            kv_actual_isl=kv_actual_isl,
            cache_batch_idx=cache_batch_idx,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            metadata=metadata,
        )

    def _sparse_chunked_attn(
        self,
        *,
        tt_q,
        tt_kvpe,
        indices,
        kvpe_cache,
        kv_actual_isl,
        cache_layer_idx,
        cache_user_id,
        seq_len_local,
        **_,
    ):
        assert indices is not None, "sparse MLA forward requires indexer top-k indices"

        cache_batch_idx = self._cache_batch_idx(cache_user_id, cache_layer_idx)

        # Chunked: the prefix lives in the BLOCK-CYCLIC cache. The high-bandwidth gather selects this
        # (user, layer) slot in-device and gathers only its populated prefix into the shared batch-1
        # worst-case buffer. sparse_sdpa receives the cache still block-cyclic and remaps natural top-k
        # indices to physical pages in-kernel (block_cyclic_chunk_local = per-shard chunk = seq_len_local),
        # so no host reorder or per-call output allocation is needed.
        self._update_kv_cache(
            kvpe_cache,
            tt_kvpe,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            tp_axis=self.tp_shard_kv_axis,  # KV dedup: write only this chip's 1/tp window
        )
        # After the write above, KV is populated up to [0, kv_actual_isl + chunk_size_global); the gather
        # only needs that populated prefix (top-k indices never address the unwritten suffix).
        populated_global = kv_actual_isl + seq_len_local * self.sp_factor
        kvpe_dev = self._gather_kvpe_prefix(
            kvpe_cache,
            cache_batch_idx,
            populated_global,
            block_cyclic_chunk_local=seq_len_local,
        )
        ttnn.deallocate(tt_kvpe)

        # Sparse attention runs over latent V; project to v_head_dim afterwards. The prefix is already
        # sliced to this slot (batch-1), so no cache_batch_idx.
        attn_out = self._sparse_mla(
            tt_q, kvpe_dev, indices, block_cyclic_chunk_local=seq_len_local, cache_batch_idx=None
        )
        # high_bw_all_gather returns a fresh Python wrapper for its supplied output. Do not use
        # wrapper identity here: SP>1 always writes the model-owned persistent scratch. At SP=1,
        # slicing a multi-slot cache creates transient storage, while slicing a single-slot cache
        # is a no-op that aliases the caller-owned persistent cache.
        # KV dedup is the exception: its two-stage all_gather_async allocates its own output at every
        # SP factor, so that result is ALWAYS transient and leaks per layer per chunk if not released.
        if (self.tp_shard_kv and self.tp_factor > 1) or (self.sp_factor == 1 and kvpe_cache.storage.shape[0] > 1):
            ttnn.deallocate(kvpe_dev.storage)
        ttnn.deallocate(tt_q)
        return self._apply_wkv_b2(attn_out, seq_len_local)

    def _forward_kv_only(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: MlaKvCache,
        cache_layer_idx: int,
        kv_actual_isl: int,
        cache_user_id: int,
        index_kv_cache: Optional[ttnn.Tensor],
        metadata: Optional[ttnn.Tensor] = None,
    ) -> None:
        """Last-layer fast path: fill the migratable KVPE cache, then stop before query/attention/output.

        A sparse full-indexer layer also writes its index-key chunk to the caller-owned ``index_kv_cache``;
        a shared/reuse-indexer layer deliberately skips that write because it owns no indexer state. The
        enclosing block skips FFN/MoE/norm/LM head as well, so this path produces no first-token output.
        """
        signpost(header="MLA_START")
        seq_len_local = hidden_states.shape[2]

        # Sparse decode needs the index key cache for every full-indexer layer even though this fast path
        # skips query construction and scoring. Shared-indexer layers reuse a prior layer's selection and
        # intentionally own no indexer weights/cache write.
        if self._has_indexer and not self._indexer_reuse:
            assert index_kv_cache is not None, "sparse kv_only requires the caller-owned index key cache"
            self._indexer.write_k(
                hidden_states,
                seq_len_local,
                kv_actual_isl,
                rope_tensors=rope_tensors,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                index_kbuf=index_kv_cache,
            )

        # Reuse the regular KV stem so kv_only and full attention cannot drift in normalization,
        # RoPE, quantization, or cache packing behavior.
        tt_kvpe, tt_kv_nope, _ = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates=False,
            kvpe_cache=kvpe_cache,
            metadata=metadata,
        )
        if tt_kv_nope is not None:
            ttnn.deallocate(tt_kv_nope)

        # Write the chunk via the SAME chunked path as _chunked_attn (not a single-shot fill):
        # update_padded_kv_cache writes at the per-chip offset derived from kv_actual_global.
        self._update_kv_cache(
            kvpe_cache,
            tt_kvpe,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            metadata=metadata,
            tp_axis=self.tp_shard_kv_axis,  # KV dedup: write only this chip's 1/tp window
        )
        ttnn.deallocate(tt_kvpe)

        signpost(header="MLA_END")
        return None

    # ----------------------------------------------------------------------------------------
    # DSA indexer + sparse attention (v3.2 / GLM). Inert unless _has_indexer (dense v3.1 path
    # never reaches these). The full forward above shares the dense/sparse Q/KV stem and epilogue;
    # only sparse-specific gather/attention helpers live below.
    # ----------------------------------------------------------------------------------------

    @property
    def tp_shard_kv_axis(self) -> Optional[int]:
        """The tp_axis to hand the cache-write ops, or None when the cache is TP-replicated. Every
        update_padded_kv_cache call site takes it from here so the axis and the enabling flag cannot drift
        apart (passing tp_axis with tp_shard_kv off would write a 1/tp window into a replicated cache)."""
        return self.tp_axis if self.tp_shard_kv else None

    @property
    def _needs_head_to_seq_reshard(self) -> bool:
        """True when the per-chip MLA head shard is too thin for sparse_sdpa (needs H % 32 == 0 and
        H >= 32). When the TP head shard is too thin (e.g. GLM's 64 heads at tp=4 → 16), _sparse_mla
        transposes the TP sharding axis heads → sequence for the duration of the attention."""
        heads_local = self.num_heads // self.tp_factor
        return self.tp_factor > 1 and (heads_local < 32 or heads_local % 32 != 0)

    def _sparse_mla(
        self,
        q: ttnn.Tensor,
        kvpe: MlaKvCache,
        indices: ttnn.Tensor,
        block_cyclic_chunk_local: Optional[int] = None,
        cache_batch_idx: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Absorbed MQA over the top-k selected latents (FlashMLA sparse contract: no causal mask —
        ``indices`` already encode it via the 0xFFFFFFFF sentinel). Invoked SPMD on the SP×TP mesh:
        each chip runs the single-chip ``ttnn.transformer.sparse_sdpa`` over its own q shard, so q's
        distribution is preserved — q SP-sharded on seq (dim2), TP-sharded on heads (dim1), out the same.

        q is absorbed ``[1, H/tp, S/sp, geometry.logical_width]`` TILE bf16. ``kvpe`` carries one
        replicated ROW_MAJOR physical cache row per token; its width depends on the explicit cache format.
        Indices are ``[1, 1, S_global, k]`` uint32, re-sharded onto SP (dim2) to match q when needed.

        block_cyclic_chunk_local: when set, ``kvpe`` is the KVPE cache in its native BLOCK-CYCLIC SP layout
        (not natural order) and ``indices`` are natural positions; sparse_sdpa remaps each index to its
        physical page in-kernel (invP) over the SP mesh axis, so the host reorder is eliminated. It is the
        per-shard chunk length (chunk_size_global / sp). None → natural-order kvpe (single-shot path).

        cache_batch_idx: when set, ``kvpe`` is the whole multi-user physical cache [B, 1, T, row_width] (B =
        num_users*num_layers user-major slots) and this selects the slot to attend — the op offsets its
        gather page ids by cache_batch_idx * T in-kernel, so no host slot-slice. None → ``kvpe`` is a
        single [1, 1, T, row_width] slot (single-shot path)."""
        assert self.sp_axis == 0 and self.tp_axis == 1, "sparse_mla assumes sp_axis=0 (outer), tp_axis=1"
        sp = self.sp_factor
        seq_len_local = q.shape[2]  # per-chip query rows == S / sp

        # sparse_sdpa requires per-chip heads H % 32 == 0 and H >= 32. When the TP head shard is too
        # thin (e.g. GLM's 64 heads at tp=4 → 16), transpose the TP sharding axis from heads to sequence
        # with one 2D-fabric all-to-all. Each chip sends only its destination sequence quarter and receives
        # all head quarters for that sequence quarter. We invert the redistribution after sparse_sdpa to
        # restore the head-sharded layout expected by the epilogue. No replicated intermediate or wasted
        # network traffic; tp=1 and already-fat shards are untouched.
        # The SP indexer emits S/sp indices; we split them over TP below to match the resharded q rows.
        transpose_head_to_seq = self._needs_head_to_seq_reshard

        q_seq_sharded = q
        if transpose_head_to_seq:
            q_seq_sharded = ttnn.experimental.all_to_all_async_generic(
                q,
                in_dim=1,
                out_dim=2,
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=self.tp_axis,
            )  # [1,H,S/(sp·tp),576] — FABRIC_2D path selected at runtime; topology resolves to Linear

        q_rm = ttnn.to_layout(q_seq_sharded, ttnn.ROW_MAJOR_LAYOUT)  # the op is ROW_MAJOR-only; q comes in TILE
        if q_seq_sharded is not q:
            ttnn.deallocate(q_seq_sharded)

        # indices must match q_rm's seq sharding. Incoming is replicated full-glob [1,1,S_global,k] or
        # SP-sharded [1,1,S/sp,k]; under reshard the row count must drop to S/(sp·tp), so split over TP.
        idx = indices
        if sp > 1 and indices.shape[2] == seq_len_local * sp:
            # Replicated full-glob indices → reshard rows onto the SP axis (inverse of all_gather).
            idx = ttnn.mesh_partition(indices, dim=2, cluster_axis=self.sp_axis)
        if transpose_head_to_seq and idx.shape[2] != q_rm.shape[2]:
            idx_seq_sharded = ttnn.mesh_partition(
                idx, dim=2, cluster_axis=self.tp_axis
            )  # split seq across TP to match q
            if idx is not indices:
                ttnn.deallocate(idx)
            idx = idx_seq_sharded
        # k_chunk_size must be a multiple of 32 that divides TOPK (prod TOPK=2048 → 128).
        k_chunk = next((c for c in (128, 64, 32) if idx.shape[-1] % c == 0), 32)
        out = ttnn.transformer.sparse_sdpa(
            q_rm,
            kvpe.storage,
            idx,
            v_dim=self.kv_lora_rank,
            kv_format=kvpe.format.sparse_sdpa_format,
            scale=self.scale,
            k_chunk_size=k_chunk,
            block_cyclic_sp_axis=self.sp_axis if block_cyclic_chunk_local is not None else None,
            block_cyclic_chunk_local=block_cyclic_chunk_local,
            # KV dedup: _gather_kvpe_prefix returns an sp*tp-striped buffer, so decode chunk_local/tp.
            block_cyclic_cache_tp_sharded=self.tp_shard_kv and block_cyclic_chunk_local is not None,
            cache_batch_idx=cache_batch_idx,
        )
        ttnn.deallocate(q_rm)
        if idx is not indices:
            ttnn.deallocate(idx)
        ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # back to TILE for the downstream wkv_b2 linear
        ttnn.deallocate(out)

        if transpose_head_to_seq:
            # Invert the redistribution so the result matches the head-sharded
            # [1, H/tp, S/sp, v_dim] consumed by the epilogue.
            head_sharded = ttnn.experimental.all_to_all_async_generic(
                ret,
                in_dim=2,
                out_dim=1,
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=self.tp_axis,
            )
            ttnn.deallocate(ret)
            ret = head_sharded
        return ret

    def _gather_kvpe_prefix(
        self,
        kvpe_cache: MlaKvCache,
        cache_batch_idx,
        populated_global: int,
        *,
        block_cyclic_chunk_local: int,
    ) -> MlaKvCache:
        """On-device read-back of the chunked KVPE prefix for sparse attention. The cache is
        ND-sharded / block-cyclic across SP, in the op's format (BF16 or packed scaled FP8, ROW_MAJOR).
        sparse_sdpa consumes it replicated and remaps the
        natural-position indices to physical block-cyclic pages in-kernel (invP), so the buffer is
        LEFT in block-cyclic order — no host reorder.

        SLOT SELECT IN THE GATHER: the persistent cache's per-chip shape is [B, 1, seq_len_local,
        row_width], B = num_users*num_layers (user-major slots), seq_len_local = seq_len_cache / sp. The
        all-gather offsets its input pages to the active (user, layer) cache slot and packs only that slot's
        valid prefix into the model-owned [1, 1, max_seq_len, row_width] output scratch. It never transports
        the other B-1 slots, avoiding the ~5 GB full-cache gather at 78 layers. The gathered KV is batch-1,
        so sparse_sdpa needs no cache_batch_idx.

        Pipeline (all on device): a selected-slot prefix SP all-gather directly from the persistent
        ND-sharded cache into the model-owned worst-case scratch (a transient selected-slot slice at sp==1).
        The cache is already in the op format, so there is no read-back dtype/layout or memory-layout
        conversion. The prefix is rounded to a whole block-cyclic slab; sparse SDPA only dereferences current
        top-k indices, so its unwritten suffix is never consumed."""

        if self.tp_shard_kv and self.tp_factor > 1:
            # GLM-5.2 KV dedup: high_bw_all_gather rides ONE cluster axis and lands in the single
            # preallocated worst-case scratch, so it cannot rebuild an sp*tp-striped slab -- that needs a
            # TP-inner gather first, into an intermediate the scratch has no room for. Take the pre-fusion
            # two-stage route; teaching the high-BW gather sp*tp stripes is follow-up work.
            return self._gather_kvpe_prefix_tp_sharded(
                kvpe_cache,
                cache_batch_idx,
                populated_global,
                block_cyclic_chunk_local=block_cyclic_chunk_local,
            )

        storage = kvpe_cache.storage
        slot_lo = cache_batch_idx if storage.shape[0] > 1 else 0
        if self.sp_factor == 1:
            # The native high-bandwidth gather requires multiple devices. Preserve the single-device
            # behavior, where sparse_sdpa still needs a batch-1 cache. For a multi-slot cache this
            # slice creates owned transient storage that the caller releases; for a single-slot cache
            # it is a no-op alias of the persistent cache and must not be released by the caller.
            gathered = ttnn.slice(
                storage,
                [slot_lo, 0, 0, 0],
                [slot_lo + 1, 1, storage.shape[2], storage.shape[3]],
            )
        else:
            # Block-cyclic storage is meaningful only in complete SP slabs. The new AG writes each
            # rank's active local prefix into its fixed worst-case slot, retaining the allocation and
            # the natural-to-physical stride expected by sparse SDPA for the next chunk/layer.
            slab_global = block_cyclic_chunk_local * self.sp_factor
            gathered_dim_size = min(
                storage.shape[2] * self.sp_factor,
                ((populated_global + slab_global - 1) // slab_global) * slab_global,
            )
            assert gathered_dim_size > 0
            assert self._sparse_kv_gather_buffer is not None
            gathered = ttnn.experimental.high_bw_all_gather(
                storage,
                dim=2,
                output_tensor=self._sparse_kv_gather_buffer,
                num_links=self.ccl_num_links,
                cluster_axis=self.sp_axis,
                input_batch_index=slot_lo,
                gathered_dim_size=gathered_dim_size,
            )

        return MlaKvCache(
            format=kvpe_cache.format,
            storage=gathered,
            geometry=kvpe_cache.geometry,
        )

    def _gather_kvpe_prefix_tp_sharded(
        self,
        kvpe_cache: MlaKvCache,
        cache_batch_idx,
        populated_global: int,
        *,
        block_cyclic_chunk_local: int,
    ) -> MlaKvCache:
        """_gather_kvpe_prefix for an SP*TP-DEDUPED cache: the pre-fusion two-stage gather.

        Each device holds only 1/tp of its SP row's slab, so the slab must be rebuilt TP-inner BEFORE the
        SP-outer gather; that order is what yields the linear chip-major buffer sparse_sdpa decodes with
        sp*tp stripes (block_cyclic_cache_tp_sharded=True), for any slab count. high_bw_all_gather cannot do it:
        it rides one cluster axis and writes the single model-owned worst-case scratch, which has no room
        for the TP-stage intermediate -- hence the plain, self-allocating all_gather_async this path was
        validated on. Both stages allocate, so the result is ALWAYS transient and the caller releases it
        (unlike the SP-only path above, whose sp>1 result IS the persistent scratch).

        Pipeline (all on device): ND->interleaved, slot + populated-width slice (no-op for a single-slot,
        full-width cache), TP-inner all-gather, SP-outer all-gather (no-op at sp==1). The cache is already
        in the op format, so there is NO read-back dtype/layout conversion."""
        storage = ttnn.to_memory_config(kvpe_cache.storage, ttnn.DRAM_MEMORY_CONFIG)
        slot_lo = cache_batch_idx if storage.shape[0] > 1 else 0
        seq_hi = storage.shape[2]
        if populated_global is not None and block_cyclic_chunk_local is not None:
            chunk_size_global = block_cyclic_chunk_local * self.sp_factor
            num_slabs = -(-populated_global // chunk_size_global)  # ceil-div
            cl_dev = block_cyclic_chunk_local // self.tp_factor  # per-DEVICE slab width
            seq_hi = min(num_slabs * cl_dev, seq_hi)

        if storage.shape[0] > 1 or seq_hi != storage.shape[2]:
            selected = ttnn.slice(
                storage,
                [slot_lo, 0, 0, 0],
                [slot_lo + 1, 1, seq_hi, storage.shape[3]],
            )
            ttnn.deallocate(storage)
            storage = selected
        tp_full = self._kvpe_all_gather(storage, dim=2, cluster_axis=self.tp_axis)  # [1,1,seq_hi*tp,row_width]
        ttnn.deallocate(storage)
        gathered = self._kvpe_all_gather(tp_full, dim=2, cluster_axis=self.sp_axis)  # [1,1,chunk_global,row_width]
        if self.sp_factor > 1:
            ttnn.deallocate(tp_full)

        return MlaKvCache(
            format=kvpe_cache.format,
            storage=gathered,
            geometry=kvpe_cache.geometry,
        )

    def _kvpe_all_gather(self, t, dim, cluster_axis):
        """All-gather across a mesh cluster axis -> replicated on that axis. factor==1: no-op.
        cluster_axis picks SP (sequence) or TP; the guard reads the matching mesh factor.

        Survives #52606 (which moved the SP-only gathers to high_bw_all_gather and dropped this helper):
        that op writes ONE preallocated worst-case scratch, so it cannot hold the TP-stage intermediate the
        dedup gather needs. Sole caller is _gather_kvpe_prefix_tp_sharded, validated on this plain gather."""
        factor = self.sp_factor if cluster_axis == self.sp_axis else self.tp_factor
        if factor == 1:
            return t
        # Per-axis topology: match the ring/line topology to the axis this gather rides.
        topology = self.sp_ccl_topology if cluster_axis == self.sp_axis else self.tp_ccl_topology
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=cluster_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=cluster_axis,
        )
