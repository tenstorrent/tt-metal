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
    def check_cache_complete(cache_path: Path, cache_name_prefix: str, has_indexer: bool = False) -> bool:
        """Check that the dense MLA weight cache files exist, plus the indexer tensorbins when sparse.

        Dense by default (preserves existing callers). When ``has_indexer=True`` the indexer cache
        (``{prefix}.indexer_*``) must also be complete — a disjoint prefix space from the dense MLA
        names, so the dense loop here never matches indexer files and vice versa
        (see ``TtIndexer.check_cache_complete``)."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for name in ttMLA.MLA_WEIGHT_NAMES:
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
        else:
            q_a_ln = torch.empty(1, 1, q_lora_rank // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            kv_a_ln = torch.empty(1, 1, kv_lora_rank // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            q_a_proj = torch.empty(hidden_size, q_lora_rank)
            q_b_proj = torch.empty(q_lora_rank, num_heads * qk_head_dim)
            kv_a_proj = torch.empty(hidden_size, kv_lora_rank + qk_rope_head_dim)
            wkv_b1 = torch.empty(1, num_heads, qk_nope_head_dim, kv_lora_rank)
            wkv_b2 = torch.empty(1, num_heads, kv_lora_rank, v_head_dim)
            o_proj = torch.empty(num_heads * v_head_dim, hidden_size)

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
        self.slot_num = slot_num
        self.layer_num = layer_num

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
        rope_factor = config.rope_scaling["factor"]
        mscale = config.rope_scaling["mscale"]

        self.scale = self.qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale

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

        self.ccl_num_links = 2 if is_blackhole() else 1  # Blackhole trains 2 fabric routing planes, others 1
        self.ccl_topology = topology

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
        logger.info(f"Loaded {len(weights)} weights in MLA layer {layer_idx} (kv_only={kv_only})")

        # DSA indexer (v3.2 / GLM): self._has_indexer was resolved above (before the buffer alloc). The
        # TtIndexer owns the indexer stems / RoPE tables / device key-cache and reuses this MLA's q_a stem
        # + collectives. Inert for dense v3.1.
        # DSA *family* (config carries the indexer fields), independent of whether the indexer is active
        # this layer. V3.1's dense config lacks them; V3.2's config has them even when a benchmark forces
        # the attention dense (has_indexer=False). Dense-path tuning gates that must tell V3.1 from a
        # dense-run V3.2 key on this, not _has_indexer (see _get_sdpa_program_config).
        self._is_dsa_family = TtIndexer.matches_config(config)
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
            # kv_only (last-layer KV-only fast path) skips Q/SDPA AND the indexer K-cache write, so a
            # sparse decode would read an unpopulated indexer cache. Not implemented — fail at construction.
            assert not self.kv_only, "DSA sparse path does not support kv_only (skips the indexer K-cache write)"
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
                    ccl_topology=self.ccl_topology,
                    seq_len=seq_len,
                    slot_num=slot_num,
                    layer_num=self.layer_num,
                )
        else:
            self._indexer = NullIndexer()  # dense v3.1: forward calls .forward() -> None (dense path)
            self._indexer_reuse = False

        # Bind the RoPE op now that self._has_indexer is known: sparse always uses the indexed
        # (block-cyclic) op — single-shot is folded onto the block-cyclic path as one full-seq chunk at
        # offset 0 — so its key cache persists layer-stacked (migratable to decode). Dense: chunked ->
        # indexed, single-shot -> rotary_embedding_llama.
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
    def kv_cache_to_host(kvpe_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int = 0):
        """Read KVPE cache from device to host tensor [1, 1, seq_total, kv_lora_rank + qk_rope_head_dim]."""
        return ttnn.to_torch(
            kvpe_cache,
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
        ).to(torch.bfloat16)

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
    }

    # Matmul dimensions for batched matmuls (wkv_b1 / wkv_b2) keyed by weight name.
    # Each entry: (K_attr, N_attr) where values are attribute names on self.
    _BATCHED_MM_DIMS = {
        "wkv_b1": ("qk_nope_head_dim", "kv_lora_rank"),
        "wkv_b2": ("kv_lora_rank", "v_head_dim"),
    }

    def _resolve_mm_cfg(self, weight_name: str, seq_len_local: int) -> dict | None:
        """Resolve the tuned matmul config for this weight/seq_len, applying head-count and
        chunked-mode gating. Returns None when no tuned config applies (caller falls back to defaults).

        The gating *tags* (num_heads / q_lora_rank / chunked_only) are declared in the config
        (mla_config.py); only the *match* is resolved here at runtime, because it depends on this live
        ttMLA. chunked_only in particular is a per-instance property (single-shot vs chunked runner) that
        the static, shared config can't know — so keeping all three checks together at this single
        consume-time point is more cohesive than splitting head/q_lora filtering into the config and
        leaving chunked here.
        """
        cfg = self.mm_configs[weight_name].get(seq_len_local) if is_blackhole() else None
        # Some tuned configs are head-count specific (the chunked-prefill 640 set was tuned for Kimi's
        # 64 heads; several program_configs overflow the grid at DeepSeek's 128). A config may declare
        # the num_heads it was tuned for; when it doesn't match this model, fall back so a different
        # variant at the same seq_len_local doesn't pick up a dimensionally-invalid program_config.
        if cfg is not None and cfg.get("num_heads") not in (None, self.num_heads):
            cfg = None
        # Some of those configs are additionally q_lora_rank-specific: the 640 set's program_configs are
        # dimensionally valid at Kimi's q_lora_rank (1536) but overflow the grid at GLM-5.1's (2048), even
        # though both have 64 heads. When a config declares a q_lora_rank that doesn't match this model,
        # fall back so a same-heads/same-seq variant doesn't pick up an invalid program_config.
        if cfg is not None and cfg.get("q_lora_rank") not in (None, self.q_lora_rank):
            cfg = None
        # The chunked-prefill 640 set is only dimensionally valid in chunked mode (e.g. wkv_b1/wkv_b2
        # are true batched per-head matmuls over the per-head SDPA output; the single-shot path applies
        # them to a batch=1 latent). Fall back to defaults when this ttMLA was not built for chunked.
        if cfg is not None and cfg.get("chunked_only") and not self.is_chunked:
            cfg = None
        return cfg

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
        cfg = self.sdpa_configs.get(seq_len_local)
        # Like the matmul configs, an SDPA config may be head-count specific (the chunked 640 entry
        # was tuned for Kimi's 64 heads). Fall back to defaults when it doesn't match this model.
        if cfg is not None and cfg.get("num_heads") not in (None, self.num_heads):
            cfg = None
        # The 640 tiling's shape is head-agnostic, but its dense-path L1 footprint (full-context K over
        # every head) only fits large head counts for the DSA family. This config is consumed ONLY on
        # the dense path (ring_mla / ring_joint SDPA); sparse V3.2/GLM go through sparse_sdpa and never
        # reach here. The dense consumers are pure-dense V3.1 (128 heads) and Kimi (64), plus a
        # dense-run V3.2 benchmark (128 heads, DSA family). V3.1 and V3.2 are dimensionally identical,
        # so num_heads can't separate them — key on the DSA family. Above dense_head_cap_non_dsa,
        # non-DSA models (V3.1) OOM L1 at k=640, so fall back to the k=32 default; DSA-family V3.2 is
        # exempt (validated dense) and Kimi stays under the cap.
        cap = cfg.get("dense_head_cap_non_dsa") if cfg is not None else None
        if cap is not None and self.num_heads > cap and not self._is_dsa_family:
            cfg = None
        # The 640 chunk tiling drives ring joint attention and is only valid in chunked mode.
        if cfg is not None and cfg.get("chunked_only") and not self.is_chunked:
            cfg = None
        q_chunk_size = cfg["q_chunk_size"] if cfg else 32
        k_chunk_size = cfg["k_chunk_size"] if cfg else 32
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

    def _apply_rope_padded(self, t: ttnn.Tensor, rope_tensors: dict, kv_actual_isl: int) -> ttnn.Tensor:
        """Chunked rotated RoPE via the indexed op. rope_tensors carry the whole-cache,
        block-cyclic-sharded cos/sin (built once via RotarySetup.get_rope_tensors_indexed); the op
        derives this chunk's per-chip shard offset on-device from kv_actual_global -- the same
        update_idxt math the KV-cache writer uses, keeping rotation and cache write consistent.
        """
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            t,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )

    def _apply_rope_one_shot(
        self, t: ttnn.Tensor, rope_tensors: dict, kv_actual_isl: Optional[int] = None
    ) -> ttnn.Tensor:
        """Single-shot RoPE: natural-order rope_tensors + rotary_embedding_llama."""
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
        kvpe_cache: ttnn.Tensor,
        kv_actual_isl: int,
        cache_batch_idx: int,
        cache_layer_idx: int,
        cache_user_id: int,
        seq_len_local: int,
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
        assert kv_actual_isl % tile_size == 0, f"kv_actual_isl ({kv_actual_isl}) must be tile-aligned"

        # Write this chunk into the cache. update_padded_kv_cache derives each chip's local write
        # offset on-device from kv_actual_global (chunk-aligned kv_actual -> uniform per-chip write).
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kvpe_cache,
            tt_kvpe,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self.layer_num,
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )

        # K and V are the single latent kvpe cache (V = first kv_lora_rank columns, materialized
        # in-op). logical_n = prior valid length + this chunk; cache_batch_idx selects this
        # user/layer's slot; kv_actual_isl drives the on-device rotation/causality offset.
        attn_out, _ = ttnn.transformer.ring_mla(
            tt_q,
            kvpe_cache,
            persistent_output_buffer_kv=self._chunked_kv_buf,
            head_dim_v=self.kv_lora_rank,
            logical_n=kv_actual_isl + chunk_size_global,
            program_config=self._get_sdpa_program_config(seq_len_local),
            scale=self.scale,
            compute_kernel_config=self.default_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=self.ccl_num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh_device,
            topology=self.ccl_topology,
            ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_balanced=self.is_balanced,
            kv_cache_batch_idx=cache_batch_idx,
            kv_actual_isl=kv_actual_isl,
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
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
            qr = ttnn.experimental.all_gather_async(
                qr,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
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

        tt_q_rope = self._apply_rope(tt_q_rope, rope_tensors, kv_actual_isl)

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
        kvpe_cache: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, dict | None]:
        """Shared KV stem.

        Returns tt_kvpe: the kvpe converted to the persistent cache's dtype/layout via
        _to_cache_format (bf8/TILE for dense, bf16 or fp8/ROW_MAJOR for sparse). This single tensor
        is BOTH written to the cache and consumed as the attention K input — dense ring SDPA takes it
        as-is, and sparse_sdpa reads the uncompressed sparse cache format natively, so no separate
        full-precision copy is needed. Also returns tt_kv_nope (dense V projection) + intermediates.
        """
        # NOTE: input is ideally L1 for chunked, but hidden states memory config is set outside the module
        tt_kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local),
        )

        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            tt_kv = ttnn.experimental.all_gather_async(
                tt_kv,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
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

        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl)

        if return_kv_intermediates:
            # post-RMSNorm latent ([.., 512]) and post-RoPE k_pe ([.., 64]); clone before concat.
            kv_intermediates["tt_kv_nope"] = ttnn.clone(tt_kv_nope)
            kv_intermediates["tt_kv_rope"] = ttnn.clone(tt_kv_rope)

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kvpe_raw = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        # Match the persistent cache format: bf8/TILE (dense) OR bf16 or fp8/ROW_MAJOR (sparse). For the
        # uncompressed sparse cache this is a layout-only change (no dtype typecast), so sparse_sdpa reads
        # it natively with no bf8->bf16 round-trip.
        tt_kvpe = self._to_cache_format(tt_kvpe_raw, kvpe_cache)
        ttnn.deallocate(tt_kvpe_raw)

        if return_kv_intermediates:
            # post-transform concat ([.., 576]) in the cache dtype -- what actually gets written.
            kv_intermediates["tt_kvpe"] = ttnn.clone(tt_kvpe)

        return tt_kvpe, tt_kv_nope, kv_intermediates

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        return ttnn.linear(
            t,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )

    def _to_cache_format(self, t: ttnn.Tensor, cache: ttnn.Tensor) -> ttnn.Tensor:
        """Convert the freshly computed kvpe `t` (bf16, TILE) to the persistent cache's dtype+layout
        for a write. The write ops require input dtype AND layout to match the cache:
          - bf8/TILE cache (dense): typecast bf16->bfloat8_b, stays TILE.
          - bf16 or fp8_e4m3 / ROW_MAJOR cache (sparse): retile TILE->ROW_MAJOR, and typecast only for
            fp8 (bf16 needs none) — so sparse_sdpa reads the prefix natively, no bf8->bf16 round-trip.
        LAYOUT is converted BEFORE dtype: fp8_e4m3 is ROW_MAJOR-only, so it must be produced by
        typecasting an already-ROW_MAJOR tensor (never a TILE fp8 intermediate, which is unsupported) —
        matching how init_kvpe_cache builds the fp8 cache. bf8 (block-float, TILE-only) is the reverse
        case, but its layout already matches (both TILE) so the layout step is a no-op there.
        Always returns a fresh tensor (clones if `t` already matches the cache format)."""
        out = t
        if out.layout != cache.layout:
            out = ttnn.to_layout(out, cache.layout)
        if out.dtype != cache.dtype:
            relaid = ttnn.typecast(out, cache.dtype)
            if out is not t:
                ttnn.deallocate(out)
            out = relaid
        return ttnn.clone(t) if out is t else out

    def _write_kvpe(self, kvpe_cache: ttnn.Tensor, tt_kvpe: ttnn.Tensor, cache_layer_idx: int) -> None:
        """DENSE single-shot cache fill: write this layer's whole kvpe slot (bf8/TILE, already in the
        cache's dtype/layout via _to_cache_format). Chunked modes (dense + sparse) and sparse single-shot
        write through update_padded_kv_cache instead; only dense single-shot still uses this TILE-only
        fill_cache_for_user_ primitive.

        TODO: unify dense single-shot onto update_padded_kv_cache too (one write op model-wide, drop this
        helper). Blocked on the single-chip case: test_prefill_block_loop[mesh-1x1] runs dense single-shot
        on a (1,1) mesh, where fill_cache_for_user_ is mesh-agnostic but update_padded_kv_cache's SP
        cluster_axis / block-cyclic / tile-aligned-kv_actual_global path is not yet validated. Confirm
        update_padded handles 1x1 (and sp=1), then switch _dense_single_attn onto it too (the sparse path
        already folded its single-shot onto the block-cyclic update_padded write)."""
        ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, tt_kvpe, cache_layer_idx)

    def _o_proj_epilogue(self, attn_out: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """Shared nlp_concat_heads -> o_proj -> TP reduce-scatter epilogue."""
        v_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
        return v_out

    # Expects activation in form of:
    # [1, batch_size == 1, seq_len // sp_factor, hidden_size // tp_factor]
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: ttnn.Tensor,
        cache_layer_idx: int = 0,
        actual_start: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
        return_indexer_indices: bool = False,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, Optional[dict]]":
        if self.kv_only:
            return self._forward_kv_only(
                hidden_states,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx,
                kv_actual_isl=actual_start,
                cache_user_id=cache_user_id,
            )

        # Chunked-prefill mode is fixed at construction: self.is_chunked drives buffer allocation in
        # __init__ and the rope variant, and forward honors that flag -- it does not infer the mode from
        # the arguments. actual_start is the chunk parameter, supplied iff chunked.
        assert (actual_start is not None) == self.is_chunked, (
            f"actual_start ({'set' if actual_start is not None else 'None'}) does not match construction "
            f"(self.is_chunked={self.is_chunked}); pass actual_start iff built with is_chunked=True"
        )

        seq_len_local = hidden_states.shape[2]
        kv_actual_isl = actual_start
        is_chunked = self.is_chunked

        # Sparse always runs the block-cyclic path (indexed rope + kvpe cache read-back), which treats
        # single-shot as one full-seq chunk at offset 0. Coerce the None single-shot offset to 0 so the
        # indexed rope op, the cache write, and the indexer all get a concrete kv_actual_global.
        if self._has_indexer and kv_actual_isl is None:
            kv_actual_isl = 0

        # Sparse attention (sparse_sdpa) reads the KVPE cache natively and only accepts bf16 / fp8_e4m3,
        # ROW_MAJOR. Require the cache to already be in that op-wanted format so the whole path is a single
        # kvpe tensor with NO compression round-trip (no bf8 typecast on write, no upcast on read-back).
        if self._has_indexer:
            assert kvpe_cache.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3), (
                f"sparse MLA requires an uncompressed KVPE cache (bf16 or fp8_e4m3) that sparse_sdpa reads "
                f"natively, got {kvpe_cache.dtype}"
            )
            assert (
                kvpe_cache.layout == ttnn.ROW_MAJOR_LAYOUT
            ), f"sparse MLA requires a ROW_MAJOR KVPE cache, got {kvpe_cache.layout}"

        signpost(header="MLA_START")

        # q-norm output uses the tuned activation memory_config in every mode (dense and sparse);
        # the next op (q_b_proj) is the same matmul regardless of attention path.
        q_norm_mem_config = self._get_act_mem_config("q_b_proj", seq_len_local)
        # Compute the q_a latent once and share it: the DSA indexer reads it for its queries, then
        # _q_stem consumes it for q_b_proj — so the sparse path does not recompute the q_a stem.
        qr = self._q_a_latent(hidden_states, seq_len_local, q_norm_mem_config)

        # DSA dispatch (v3.2 / GLM): the op graph is fixed by CONFIG — self._indexer and self._attention
        # are bound once at construction (sparsity × chunking), never by the runtime sequence length — so
        # a single recorded op trace replays identically for any input. (At seq_len <= index_topk the
        # indexer top-k simply selects all available causal keys, so sparse is numerically equal to dense
        # there.) The indexer's forward also writes its K-cache (a no-op on the dense null-indexer), so no
        # separate warm-up write is needed.
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

        tt_q = self._q_stem(qr, rope_tensors, kv_actual_isl, seq_len_local)
        tt_kvpe, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            kvpe_cache,
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
        )

        out = self._o_proj_epilogue(attn_out, seq_len_local)
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
            topology=self.ccl_topology,
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

        # Chunked: the prefix lives in the BLOCK-CYCLIC cache. Slice this (user, layer) slot out and
        # gather ONLY that slot on device, then hand it to sparse_sdpa still block-cyclic — the op remaps
        # the natural top-k indices to physical pages in-kernel (block_cyclic_chunk_local = per-shard
        # chunk = seq_len_local), so no host reorder. Slicing the slot BEFORE the gather keeps the
        # all-gather to a single [1,1,T,576] slot (gathering the whole B=num_users*num_layers cache OOMs
        # at 78 layers). The gathered kv is batch-1, so cache_batch_idx is unset (None) below.
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kvpe_cache,
            tt_kvpe,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self.layer_num,
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )
        kvpe_dev = self._gather_kvpe_prefix(kvpe_cache, cache_batch_idx)
        ttnn.deallocate(tt_kvpe)

        # Sparse attention runs over latent V; project to v_head_dim afterwards. The prefix is already
        # sliced to this slot (batch-1), so no cache_batch_idx.
        attn_out = self._sparse_mla(
            tt_q, kvpe_dev, indices, block_cyclic_chunk_local=seq_len_local, cache_batch_idx=None
        )
        ttnn.deallocate(kvpe_dev)
        ttnn.deallocate(tt_q)
        return self._apply_wkv_b2(attn_out, seq_len_local)

    def _forward_kv_only(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: ttnn.Tensor,
        cache_layer_idx: int,
        kv_actual_isl: int,
        cache_user_id: int,
    ) -> None:
        """Last-layer fast path: fill the KV cache (which migration consumes), then stop. Skips
        Q / SDPA / output projection entirely; the block also skips FFN/MoE/norm/LM head, so no
        first-token output is produced.
        """
        signpost(header="MLA_START")
        seq_len_local = hidden_states.shape[2]

        tt_kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local),
        )

        if self.tp_factor > 1:
            tt_kv = ttnn.experimental.all_gather_async(
                tt_kv,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
            tt_kv = ttnn.experimental.fast_reduce_nc(
                tt_kv, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )

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

        # Same rope as the full chunked path (indexed/padded when chunked, single-shot otherwise) so
        # the KV written to the cache carries the correct per-chunk positional offset.
        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl)

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        tt_kvpe = ttnn.typecast(tt_kvpe, dtype=ttnn.bfloat8_b)

        # Write the chunk via the SAME chunked path as _chunked_attn (not a single-shot fill):
        # update_padded_kv_cache writes at the per-chip offset derived from kv_actual_global.
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kvpe_cache,
            tt_kvpe,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self.layer_num,
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )

        signpost(header="MLA_END")
        return None

    # ----------------------------------------------------------------------------------------
    # DSA indexer + sparse attention (v3.2 / GLM). Inert unless _has_indexer (dense v3.1 path
    # never reaches these). The full forward above shares the dense/sparse Q/KV stem and epilogue;
    # only sparse-specific gather/attention helpers live below.
    # ----------------------------------------------------------------------------------------

    def _all_gather(self, t, dim, cluster_axis):
        """All-gather across a mesh cluster axis → replicated on that axis. factor==1: no-op.
        cluster_axis picks SP (sequence) or TP; the guard reads the matching mesh factor."""
        factor = self.sp_factor if cluster_axis == self.sp_axis else self.tp_factor
        if factor == 1:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=cluster_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=cluster_axis,
        )

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
        kvpe: ttnn.Tensor,
        indices: ttnn.Tensor,
        block_cyclic_chunk_local: Optional[int] = None,
        cache_batch_idx: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Absorbed MQA over the top-k selected latents (FlashMLA sparse contract: no causal mask —
        ``indices`` already encode it via the 0xFFFFFFFF sentinel). Invoked SPMD on the SP×TP mesh:
        each chip runs the single-chip ``ttnn.transformer.sparse_sdpa`` over its own q shard, so q's
        distribution is preserved — q SP-sharded on seq (dim2), TP-sharded on heads (dim1), out the same.

        q: [1, H/tp, S/sp, 576] absorbed (TILE bf16); kvpe: [1, 1, T, 576] full latent prefix, ROW_MAJOR
        replicated (K = full 576, V = leading kv_lora_rank). indices: [1, 1, S_global, k] uint32 replicated,
        re-sharded onto SP (dim2) to match q when sp > 1 (or already SP-sharded → pass through).

        block_cyclic_chunk_local: when set, ``kvpe`` is the KVPE cache in its native BLOCK-CYCLIC SP layout
        (not natural order) and ``indices`` are natural positions; sparse_sdpa remaps each index to its
        physical page in-kernel (invP) over the SP mesh axis, so the host reorder is eliminated. It is the
        per-shard chunk length (chunk_size_global / sp). None → natural-order kvpe (single-shot path).

        cache_batch_idx: when set, ``kvpe`` is the whole multi-user cache [B, 1, T, 576] (B =
        num_users*num_layers user-major slots) and this selects the slot to attend — the op offsets its
        gather page ids by cache_batch_idx * T in-kernel, so no host slot-slice. None → ``kvpe`` is a
        single [1, 1, T, 576] slot (single-shot path)."""
        assert self.sp_axis == 0 and self.tp_axis == 1, "sparse_mla assumes sp_axis=0 (outer), tp_axis=1"
        sp = self.sp_factor
        seq_len_local = q.shape[2]  # per-chip query rows == S / sp

        # sparse_sdpa requires per-chip heads H % 32 == 0 and H >= 32. When the TP head shard is too
        # thin (e.g. GLM's 64 heads at tp=4 → 16), transpose the TP sharding axis from heads to sequence
        # for the duration of the attention: all-gather q's heads over TP (each chip regains all H heads),
        # then re-shard the sequence over TP so every chip attends a DISTINCT seq slice in parallel at full
        # H. After the op we invert it (gather seq, re-shard heads) to restore the head-sharded layout the
        # wkv_b2 / o_proj epilogue expects. No padded/wasted heads; tp=1 and already-fat shards are untouched.
        # The SP indexer emits S/sp indices; we split them over TP below to match the resharded q rows.
        transpose_head_to_seq = self._needs_head_to_seq_reshard

        q_seq_sharded = q
        if transpose_head_to_seq:
            q_all_heads = self._all_gather(q, dim=1, cluster_axis=self.tp_axis)  # [1, H, S/sp, 576] repl on TP
            q_seq_sharded = ttnn.mesh_partition(q_all_heads, dim=2, cluster_axis=self.tp_axis)  # [1,H,S/(sp·tp),576]
            ttnn.deallocate(q_all_heads)

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
            kvpe,
            idx,
            v_dim=self.kv_lora_rank,
            scale=self.scale,
            k_chunk_size=k_chunk,
            block_cyclic_sp_axis=self.sp_axis if block_cyclic_chunk_local is not None else None,
            block_cyclic_chunk_local=block_cyclic_chunk_local,
            cache_batch_idx=cache_batch_idx,
        )
        ttnn.deallocate(q_rm)
        if idx is not indices:
            ttnn.deallocate(idx)
        ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # back to TILE for the downstream wkv_b2 linear
        ttnn.deallocate(out)

        if transpose_head_to_seq:
            # Invert the transpose: gather the per-TP seq slices back to S/sp, then re-shard heads onto TP
            # so the result matches the head-sharded [1, H/tp, S/sp, v_dim] the epilogue consumes.
            out_all_heads = self._all_gather(ret, dim=2, cluster_axis=self.tp_axis)  # [1, H, S/sp, v_dim] repl on TP
            ttnn.deallocate(ret)
            ret = ttnn.mesh_partition(out_all_heads, dim=1, cluster_axis=self.tp_axis)  # [1, H/tp, S/sp, v_dim]
            ttnn.deallocate(out_all_heads)
        return ret

    def _gather_kvpe_prefix(self, kvpe_cache, cache_batch_idx):
        """On-device read-back of the chunked KVPE prefix for sparse attention. The cache is
        ND-sharded / block-cyclic across SP, in the op's format (bf16 or fp8_e4m3, ROW_MAJOR — the
        sparse cache is uncompressed). sparse_sdpa consumes it replicated and remaps the
        natural-position indices to physical block-cyclic pages in-kernel (invP), so the buffer is
        LEFT in block-cyclic order — no host reorder.

        SLOT SELECT BEFORE THE GATHER: the persistent cache is [B, 1, seq_len_cache, 576], B =
        num_users*num_layers (user-major slots). Slice the active (user, layer) slot out of dim 0 FIRST,
        then all-gather only that single [1, 1, seq_len_cache, 576] slot over SP — NOT the whole B-slot
        cache. At 78 layers the full-cache gather is ~5 GB (×2 in-flight → OOM against the near-full
        78-layer weights); the single-slot gather is B× smaller. The gathered kv is then batch-1, so
        sparse_sdpa needs NO cache_batch_idx (the op requires B==1 when cache_batch_idx is unset). This
        mirrors the dense ring_mla single-slot gather (kv_cache_batch_idx → batch-1 scratch).

        Pipeline (all on device): ND→interleaved, slot slice (no-op for a single-slot cache), SP
        all-gather to full-T (no-op at sp==1). The cache is already in the op format, so there is NO
        read-back dtype/layout conversion. The unwritten suffix is never addressed since indices stay
        < populated."""
        cache_i = ttnn.to_memory_config(kvpe_cache, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        if cache_i.shape[0] > 1:  # user-major slot select BEFORE the gather (single-slot cache → skip)
            sel = ttnn.slice(
                cache_i,
                [cache_batch_idx, 0, 0, 0],
                [cache_batch_idx + 1, 1, cache_i.shape[2], cache_i.shape[3]],
            )
            ttnn.deallocate(cache_i)
            cache_i = sel
        full = self._all_gather(
            cache_i, dim=2, cluster_axis=self.sp_axis
        )  # → [1,1,seq_len_cache,576] repl, block-cyclic
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        return full
