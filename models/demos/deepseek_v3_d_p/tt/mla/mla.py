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
from models.demos.deepseek_v3_d_p.tt.mla.indexer import NullIndexer, TtIndexer, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG, MLA_SDPA_CONFIG
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_to_natural
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
        resolved_has_indexer = resolve_has_indexer(config, state_dict=state_dict, explicit=has_indexer)
        if resolved_has_indexer:
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
        # Trace controller for the per-layer migration-ack boundary (runner trace path). Set via
        # set_trace_controller(); only acts when the controller carries an ack callback. None by default
        # (single-shot / test paths), in which case the ack site falls back to the direct call.
        self._trace_controller = None
        self.kv_only = kv_only
        self.is_balanced = is_balanced
        self.weight_cache_path = weight_cache_path
        self.is_chunked = is_chunked
        self.slot_num = slot_num
        self.layer_num = layer_num

        # The RoPE op is fixed by the configured mode: chunked prefill uses the indexed op,
        # single-shot uses rotary_embedding_llama. Bind once here so forward doesn't re-decide.
        self._apply_rope = self._apply_rope_padded if is_chunked else self._apply_rope_one_shot

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
        # kv_only (last layer) never reaches SDPA, so it needs no ring/gather buffers at all.
        if kv_only:
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

        # DSA indexer (v3.2 / GLM): resolve sparse mode EXPLICITLY — config DSA fields, then live host
        # weights, then a complete indexer cache — never from bool(idx_host), which silently went dense
        # for cache-only construction. The TtIndexer owns the indexer stems / RoPE tables / device
        # key-cache and reuses this MLA's q_a stem + collectives. Inert for dense v3.1.
        self._has_indexer = resolve_has_indexer(
            config,
            state_dict=state_dict,
            explicit=has_indexer,
            weight_cache_path=self.weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.mla",
        )
        if self._has_indexer:
            # The indexer assumes natural-order SP sharding (contiguous per-chip query blocks: its
            # device RoPE and the indexer_score per-device causal offset both index positions as
            # start_pos + sp_rank*S_local). The balanced chunk reorder breaks that, so guard it.
            assert not self.is_balanced, "DSA indexer requires is_balanced=False (natural-order SP sharding)"
            # kv_only (last-layer KV-only fast path) skips Q/SDPA AND the indexer K-cache write, so a
            # sparse decode would read an unpopulated indexer cache. Not implemented — fail at construction.
            assert not self.kv_only, "DSA sparse path does not support kv_only (skips the indexer K-cache write)"
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
            )
        else:
            self._indexer = NullIndexer()  # dense v3.1: forward calls .forward() -> None (dense path)

        # Bind the attention core once, by config — sparsity (self._has_indexer) × chunking
        # (self.is_chunked) — exactly as self._apply_rope is bound above. forward() then calls
        # self._attention(...) with no mode ladder: the decision is made here, not per call.
        if self._has_indexer:
            self._attention = self._sparse_chunked_attn if self.is_chunked else self._sparse_single_attn
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
        """
        cfg = self.mm_configs[weight_name].get(seq_len_local) if is_blackhole() else None
        # Some tuned configs are head-count specific (the chunked-prefill 640 set was tuned for Kimi's
        # 64 heads; several program_configs overflow the grid at DeepSeek's 128). A config may declare
        # the num_heads it was tuned for; when it doesn't match this model, fall back so a different
        # variant at the same seq_len_local doesn't pick up a dimensionally-invalid program_config.
        if cfg is not None and cfg.get("num_heads") not in (None, self.num_heads):
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

    def set_trace_controller(self, controller):
        """Register the SubDeviceTraceController so the per-layer migration ack can be chopped out of the
        captured trace (runner trace path). The ack only fires through the controller when the controller
        carries an ack callback (has_layer_ack()); otherwise the ack site falls back to the direct call."""
        self._trace_controller = controller

    def _apply_rope_padded(
        self, t: ttnn.Tensor, rope_tensors: dict, kv_actual_isl: int, metadata: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        """Chunked rotated RoPE via the indexed op. rope_tensors carry the whole-cache,
        block-cyclic-sharded cos/sin (built once via RotarySetup.get_rope_tensors_indexed); the op
        derives this chunk's per-chip shard offset on-device from kv_actual_global -- the same
        update_idxt math the KV-cache writer uses, keeping rotation and cache write consistent.

        Per-element-tensor path: `metadata` is a 3-tuple of 1-element uint32 tensors
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

    def _apply_rope_one_shot(
        self,
        t: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int] = None,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Single-shot RoPE: natural-order rope_tensors + rotary_embedding_llama. (metadata unused --
        single-shot has no chunked rotation; accepted so forward can pass it uniformly.)"""
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
        # Per-element-tensor path: slot_idx (metadata[0]) + kv_actual_global (metadata[1]) read on-device,
        # each its own 1-element tensor.
        if metadata is not None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache,
                tt_kvpe,
                metadata[0],  # slot_idx tensor
                metadata[1],  # kv_actual_global tensor
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                cluster_axis=self.sp_axis,
            )
        else:
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
        #
        # Trace-safe metadata path: when a metadata tensor is supplied, ring_mla reads its per-chunk
        # scalars on-device: the all-gather + SDPA readers take the cache slot from metadata[0], and the
        # SDPA reader derives logical_nt / q-mapping / ring masks from kv_actual_isl = metadata[1]. Every
        # kernel derives logical_n on-device on this path, so the host logical_n is a placeholder = global
        # cache capacity (safe upper bound). metadata[0] holds only the user slot, so pass the per-layer
        # factor (kv_cache_num_layers/kv_cache_layer_idx) so the readers recompute the full (user,layer)
        # slot on-device -- otherwise every layer would read layer 0's KV cache.
        if metadata is not None:
            meta_slot_kwargs = {
                "slot_id": metadata[0],
                "kv_actual_isl_tensor": metadata[1],
                "kv_cache_num_layers": self.layer_num,
                "kv_cache_layer_idx": cache_layer_idx,
            }
            ring_logical_n = kvpe_cache.shape[2] * self.sp_factor  # global cache capacity
        else:
            meta_slot_kwargs = {"kv_cache_batch_idx": cache_batch_idx, "kv_actual_isl": kv_actual_isl}
            ring_logical_n = kv_actual_isl + chunk_size_global
        attn_out, _ = ttnn.transformer.ring_mla(
            tt_q,
            kvpe_cache,
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
            topology=self.ccl_topology,
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
        keep_kvpe_bf16: bool,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor, ttnn.Tensor, dict | None]:
        """Shared KV stem.

        Returns bf16 kvpe only for sparse single-shot, where sparse attention consumes the
        full latent prefix in bf16. All modes receive the bf8 kvpe written to the cache.
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

        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl, metadata=metadata)

        if return_kv_intermediates:
            # post-RMSNorm latent ([.., 512]) and post-RoPE k_pe ([.., 64]); clone before concat.
            kv_intermediates["tt_kv_nope"] = ttnn.clone(tt_kv_nope)
            kv_intermediates["tt_kv_rope"] = ttnn.clone(tt_kv_rope)

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        tt_kvpe_b8 = ttnn.typecast(tt_kvpe, dtype=ttnn.bfloat8_b)

        if return_kv_intermediates:
            # post-transform concat ([.., 576], bf8) -- what actually gets written to the cache.
            kv_intermediates["tt_kvpe"] = ttnn.clone(tt_kvpe_b8)

        if not keep_kvpe_bf16:
            ttnn.deallocate(tt_kvpe)
            tt_kvpe = None

        return tt_kvpe, tt_kvpe_b8, tt_kv_nope, kv_intermediates

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        return ttnn.linear(
            t,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )

    def _write_kvpe(self, kvpe_cache: ttnn.Tensor, tt_kvpe_b8: ttnn.Tensor, cache_layer_idx: int) -> None:
        """Single-shot cache fill: write this layer's whole kvpe slot (bf8). Chunked modes write
        through _chunked_attn / update_padded_kv_cache instead."""
        ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, tt_kvpe_b8, cache_layer_idx)

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
        metadata: Optional[ttnn.Tensor] = None,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, Optional[dict]]":
        # Trace-safe metadata path: a 3-tuple of 1-element uint32 DRAM tensors (slot_id, actual_start,
        # actual_end) passed in from outside. When provided, the chunked-prefill ops (update_padded_kv_cache,
        # rotary_embedding_indexed, ring_mla) read their per-chunk scalars on-device from it instead of from
        # host actual_start/cache_user_id. Threaded through verbatim -- ttMLA never reads/reconstructs it.
        if self.kv_only:
            return self._forward_kv_only(
                hidden_states,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx,
                kv_actual_isl=actual_start,
                cache_user_id=cache_user_id,
                metadata=metadata,
            )

        # Chunked-prefill mode is fixed at construction: self.is_chunked drives buffer allocation in
        # __init__ and the rope variant, and forward honors that flag -- it does not infer the mode from
        # the arguments. actual_start is the chunk parameter, supplied iff chunked. EXCEPT on the metadata
        # path, where the per-chunk scalars are read on-device from the metadata tensor (no host actual_start).
        assert metadata is not None or (actual_start is not None) == self.is_chunked, (
            f"actual_start ({'set' if actual_start is not None else 'None'}) does not match construction "
            f"(self.is_chunked={self.is_chunked}); pass actual_start iff built with is_chunked=True"
        )

        seq_len_local = hidden_states.shape[2]
        kv_actual_isl = actual_start
        is_chunked = self.is_chunked

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
        indices = self._indexer.forward(hidden_states, qr, seq_len_local, start_pos=kv_actual_isl or 0)

        tt_q = self._q_stem(qr, rope_tensors, kv_actual_isl, seq_len_local, metadata=metadata)
        keep_kvpe_bf16 = self._has_indexer and not is_chunked
        tt_kvpe, tt_kvpe_b8, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            keep_kvpe_bf16,
            metadata=metadata,
        )

        attn_out = self._attention(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            tt_kvpe_b8=tt_kvpe_b8,
            tt_kv_nope=tt_kv_nope,
            indices=indices,
            kvpe_cache=kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            kv_actual_isl=kv_actual_isl,
            metadata=metadata,
        )

        out = self._o_proj_epilogue(attn_out, seq_len_local)
        signpost(header="MLA_END")
        if return_kv_intermediates:
            return out, kv_intermediates
        return out

    # Attention core variants, one bound to self._attention at construction (sparsity × chunking).
    # All four share the forward() call signature and ignore the kwargs they don't need (**_), so the
    # bound name can be invoked uniformly. Bodies are the former mode-ladder branches, unchanged.

    def _dense_single_attn(self, *, tt_q, tt_kvpe_b8, tt_kv_nope, kvpe_cache, cache_layer_idx, seq_len_local, **_):
        # Single-shot prefill: materialize V before causal ring SDPA.
        self._write_kvpe(kvpe_cache, tt_kvpe_b8, cache_layer_idx)
        tt_v_embedding = self._apply_wkv_b2(tt_kv_nope, seq_len_local)
        attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_kvpe_b8,
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

    def _dense_chunked_attn(
        self,
        *,
        tt_q,
        tt_kvpe_b8,
        kvpe_cache,
        kv_actual_isl,
        cache_layer_idx,
        cache_user_id,
        seq_len_local,
        metadata=None,
        **_,
    ):
        # Cache batch dim is user-major: each user reserves self.layer_num contiguous slots, so the
        # flat slot is cache_user_id * layer_num + cache_layer_idx.
        assert cache_user_id < self.slot_num, f"cache_user_id {cache_user_id} >= slot_num {self.slot_num}"
        cache_batch_idx = cache_user_id * self.layer_num + cache_layer_idx
        return self._chunked_attn(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe_b8,
            kvpe_cache=kvpe_cache,
            kv_actual_isl=kv_actual_isl,
            cache_batch_idx=cache_batch_idx,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            metadata=metadata,
        )

    def _sparse_single_attn(
        self, *, tt_q, tt_kvpe, tt_kvpe_b8, indices, kvpe_cache, cache_layer_idx, seq_len_local, **_
    ):
        assert indices is not None, "sparse MLA forward requires indexer top-k indices"
        # Single-shot: the live kvpe IS the whole sequence (natural order, contiguous SP shard).
        assert tt_kvpe is not None
        self._write_kvpe(kvpe_cache, tt_kvpe_b8, cache_layer_idx)
        gathered = self._sp_all_gather(tt_kvpe, dim=2)  # [1,1,T,576] bf16 TILE, replicated, natural
        kvpe_dev = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
        if self.sp_factor > 1:
            ttnn.deallocate(gathered)
        ttnn.deallocate(tt_kvpe_b8)
        ttnn.deallocate(tt_kvpe)

        # Sparse attention runs over latent V; project to v_head_dim afterwards.
        attn_out = self._sparse_mla(tt_q, kvpe_dev, indices)
        ttnn.deallocate(kvpe_dev)
        ttnn.deallocate(tt_q)
        return self._apply_wkv_b2(attn_out, seq_len_local)

    def _sparse_chunked_attn(
        self,
        *,
        tt_q,
        tt_kvpe_b8,
        indices,
        kvpe_cache,
        kv_actual_isl,
        cache_layer_idx,
        cache_user_id,
        seq_len_local,
        metadata=None,
        **_,
    ):
        assert indices is not None, "sparse MLA forward requires indexer top-k indices"
        assert metadata is None, (
            "sparse (DSA indexer) chunked prefill does not yet support the trace-safe per-element metadata "
            "path (the gather/sparse_mla derive positions from host kv_actual_isl); use the dense path."
        )
        start_pos = kv_actual_isl or 0
        end_pos = start_pos + seq_len_local * self.sp_factor

        # Chunked: the prefix lives in the BLOCK-CYCLIC cache; gather + un-rotate on device.
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kvpe_cache,
            tt_kvpe_b8,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self.layer_num,
            kv_actual_global=kv_actual_isl,
            cluster_axis=self.sp_axis,
        )
        kvpe_dev = self._gather_kvpe_prefix(kvpe_cache, seq_len_local, end_pos, cache_user_id, cache_layer_idx)
        ttnn.deallocate(tt_kvpe_b8)

        # Sparse attention runs over latent V; project to v_head_dim afterwards.
        attn_out = self._sparse_mla(tt_q, kvpe_dev, indices)
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
        metadata: Optional[ttnn.Tensor] = None,
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
        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl, metadata=metadata)

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        tt_kvpe = ttnn.typecast(tt_kvpe, dtype=ttnn.bfloat8_b)

        # Write the chunk via the SAME chunked path as _chunked_attn (not a single-shot fill):
        # update_padded_kv_cache writes at the per-chip offset derived from kv_actual_global.
        # Per-element-tensor path: slot_idx (metadata[0]) + kv_actual_global (metadata[1]) tensors.
        if metadata is not None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache,
                tt_kvpe,
                metadata[0],  # slot_idx tensor
                metadata[1],  # kv_actual_global tensor
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                cluster_axis=self.sp_axis,
            )
        else:
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

    def _sp_all_gather(self, t, dim):
        """All-gather across the SP axis (sequence) → full-S replicated on SP. sp=1: no-op."""
        if self.sp_factor == 1:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.sp_axis,
        )

    def _sparse_mla(self, q: ttnn.Tensor, kvpe: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
        """Absorbed MQA over the top-k selected latents (FlashMLA sparse contract: no causal mask —
        ``indices`` already encode it via the 0xFFFFFFFF sentinel). Invoked SPMD on the SP×TP mesh:
        each chip runs the single-chip ``ttnn.transformer.sparse_sdpa`` over its own q shard, so q's
        distribution is preserved — q SP-sharded on seq (dim2), TP-sharded on heads (dim1), out the same.

        q: [1, H/tp, S/sp, 576] absorbed (TILE bf16); kvpe: [1, 1, T, 576] full latent prefix, ROW_MAJOR
        replicated (K = full 576, V = leading kv_lora_rank). indices: [1, 1, S_global, k] uint32 replicated,
        re-sharded onto SP (dim2) to match q when sp > 1 (or already SP-sharded → pass through)."""
        assert self.sp_axis == 0 and self.tp_axis == 1, "sparse_mla assumes sp_axis=0 (outer), tp_axis=1"
        sp = list(q.device().shape)[self.sp_axis]
        q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)  # the op is ROW_MAJOR-only; q comes in TILE
        idx = indices
        if sp > 1 and indices.shape[2] == q.shape[2] * sp:
            # Replicated full-glob indices → reshard rows onto the SP axis (inverse of all_gather), matching
            # q's per-chip seq shard. Already-SP-sharded indices ([1,1,S/sp,k]) match q and pass through.
            idx = ttnn.mesh_partition(indices, dim=2, cluster_axis=self.sp_axis)
        # k_chunk_size must be a multiple of 32 that divides TOPK (prod TOPK=2048 → 128).
        k_chunk = next((c for c in (128, 64, 32) if idx.shape[-1] % c == 0), 32)
        out = ttnn.transformer.sparse_sdpa(
            q_rm, kvpe, idx, v_dim=self.kv_lora_rank, scale=self.scale, k_chunk_size=k_chunk
        )
        ttnn.deallocate(q_rm)
        if idx is not indices:
            ttnn.deallocate(idx)
        ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # back to TILE for the downstream wkv_b2 linear
        ttnn.deallocate(out)
        return ret

    def _gather_kvpe_prefix(self, kvpe_cache, seq_len_local, end_pos, cache_user_id, cache_layer_idx):
        """On-device read-back of the chunked KVPE prefix for sparse attention. The cache is
        bf8 / TILE / ND-sharded / block-cyclic across SP; the sparse op wants the full prefix
        bf16 / ROW_MAJOR / replicated / natural order. Pipeline (all on device — replaces the
        former host ConcatMesh2dToTensor + blockcyclic_positions read): ND→interleaved, SP
        all-gather to full-T (no-op at sp==1), select this user/layer slot, bf8→bf16, TILE→RM,
        un-rotate block-cyclic→natural, trim to end_pos. Returns [1, 1, end_pos, 576] bf16 RM."""
        cache_i = ttnn.to_memory_config(kvpe_cache, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        full = self._sp_all_gather(cache_i, dim=2)  # → [B, 1, seq_len_cache, 576] replicated, block-cyclic
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        if full.shape[0] > 1:  # user-major slot select (no-op for the single-slot cache)
            slot = cache_user_id * self.layer_num + cache_layer_idx
            sel = ttnn.slice(full, [slot, 0, 0, 0], [slot + 1, 1, full.shape[2], full.shape[3]])
            ttnn.deallocate(full)
            full = sel
        full16 = ttnn.typecast(full, ttnn.bfloat16)
        ttnn.deallocate(full)
        full_rm = ttnn.to_layout(full16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(full16)
        out = blockcyclic_to_natural(full_rm, self.sp_factor, seq_len_local, end_pos)
        ttnn.deallocate(full_rm)
        return out
