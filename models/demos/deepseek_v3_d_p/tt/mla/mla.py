# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.indexer import (
    NullIndexer,
    ReuseIndexer,
    TtIndexer,
    indexer_layer_is_reused,
    resolve_has_indexer,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG, MLA_SDPA_CONFIG, small_rows_entry
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import sparse_kv_push_supported
from models.demos.deepseek_v3_d_p.tt.mla.utils import llama4_scale_host
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, resolve_per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCache,
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    declared_seq_shard_factor,
    is_tp_deduped_kv_cache,
)

# Axis 0 is N/S (mesh rows), axis 1 is E/W (mesh cols) -- the same convention high_bw_all_gather uses.

# glue track: workers per link for the q_a (and merged q_a+kv_a) TP reduce-scatter; "0" = op default.
_GLUE_STEM_RS_WORKERS = int(os.environ.get("GLUE_STEM_RS_WORKERS", "8")) or None
# glue track: o_proj RS output in L1 (read once by the block's residual add).
_GLUE_EPI_L1 = os.environ.get("GLUE_EPI_L1", "1") == "1"
# glue track: L1 placement in the in-place-RoPE Q chain (head split outputs; the [16, S, 576] Q buffer).
_GLUE_Q_HEADS_L1 = os.environ.get("GLUE_Q_HEADS_L1", "1") == "1"
_GLUE_Q_OUT_L1 = os.environ.get("GLUE_Q_OUT_L1", "0") == "1"
# mm track: GLM o_proj on the full 12x10 grid (see ttMLA._oproj_grid12; MLA_OPROJ_GRID12=0 disables). In-model L10 on
# v13: 125.8 -> 117.2 us per layer, per-chip idle unchanged, KVPE bit-identical. mla_config.py keeps MLA on 11x10 for
# di/dt / throttling, so the keep decision is the L78 interleaved ABAB.
_MLA_OPROJ_GRID12 = os.environ.get("MLA_OPROJ_GRID12", "1") == "1"
# mm track: o_proj weight laid out DRAM WIDTH_SHARDED ([K, N/banks] per bank, device tile copy, bit-identical tiles) so
# the 2D mcast in1 senders read it bank by bank (MLA_OPROJ_WS=0 disables). L10 on 12x10: 117.3 -> 114.2 us per layer,
# KVPE bit-identical (R0 12x10: 118.1 -> 115.1 us, 11x10: 128.0 -> 126.8 us).
_MLA_OPROJ_WS = os.environ.get("MLA_OPROJ_WS", "1") == "1"
# glue track: seqq q_a norm on its 160 rows width-sharded over this many cores (0 = interleaved rms_norm, 5 cores).
_GLUE_SEQQ_NORM_CORES = int(os.environ.get("GLUE_SEQQ_NORM_CORES", "32"))
# glue track: workers per link for the seqq q_a reduce-scatter over rows; 0 = op default.
_GLUE_SEQQ_RS_WORKERS = int(os.environ.get("GLUE_SEQQ_RS_WORKERS", "0")) or None
# glue track (stem): on the non-overlapped sparse path (shared layers) build the sequence-sharded Q AFTER the KV
# cache write + prefix push/gather, so the [1, H, S/(sp*tp), 576] Q buffer can live in L1 for its short life
# (wkv_b1 -> in-place RoPE -> untilize to DRAM) without clashing with later static CBs.
_GLUE_Q_DEFER_L1 = os.environ.get("GLUE_Q_DEFER_L1", "1") == "1"
# ... and on full (indexer) layers too: Q built after the top-k || gather region (GLUE_Q_DEFER_FULL, opt-in).
_GLUE_Q_DEFER_FULL = os.environ.get("GLUE_Q_DEFER_FULL", "1") == "1"
# glue track (stem): kv stem intermediates (reduce split -> norm -> rope -> fp8 fields) in L1, freed right after
# the cache write (GLUE_KV_L1, opt-in); nlp_concat_heads output (o_proj in0) in L1 (GLUE_CONCAT_L1, opt-in).
_GLUE_KV_L1 = os.environ.get("GLUE_KV_L1", "1") == "1"
_GLUE_CONCAT_L1 = os.environ.get("GLUE_CONCAT_L1", "0") == "1"
_GLUE_WKVB2_OUT_L1 = os.environ.get("GLUE_WKVB2_OUT_L1", "1") == "1"
# Q-stem || push overlap (overlap agent, 2026-10-25): on shared (reuse) layers the deferred Q stem's q_b matmul needs
# nothing from the sparse KV prefix push that precedes it, so both run at once on the sparse-MLA overlap manager:
# the push on its gather sub-device (rows 0-1: the same 24 cores, right below the Ethernet row, it uses today) and
# q_b on the top-k sub-device (rows 2-9) with its grid anchored at (0, 2) (allowed_worker_cores, 11x8 = 88 cores,
# per_core_N 5 -> 6). q_b is compute only, so no two programs of the region share a fabric router
# (notes/pipeline/deadlock-matrix.md). The manager clear is the join; the rest of the Q stem (head split, wkv_b1,
# in-place RoPE: full-grid ops) follows it. On by default (GLM_QPUSH_OVERLAP=0 disables): L10 traced on v14
# 360.56/360.57 -> 359.22/359.13 ms (-0.38%, ~-21 us per shared layer), KVPE bit-identical (runs/20261025T210154Z-pre-wqp1.log).
_QPUSH_OVERLAP = os.environ.get("GLM_QPUSH_OVERLAP", "1") == "1"
# Fallback (GLM_QPUSH_QB_DRAM=1): the region's q_b output in DRAM instead of L1 (if an L1 one ever clashes with the
# push's static CBs, which the enqueue check reports as a TT_FATAL).
_QPUSH_QB_DRAM = os.environ.get("GLM_QPUSH_QB_DRAM", "0") == "1"
# Full layers (GLM_QPUSH_FULL=1, needs GLM_QPUSH_OVERLAP=1): after the top-k || gather region a full layer plans its
# successor's push (sparse_push_plan 120 cores -> the plan tables' full-mesh all-gather -> sparse_push_hot 120 cores);
# the deferred q_b runs next to that all-gather, on the same manager and sub-devices as on shared layers. Opt-in: on v14
# L10 it measured 359.46 ms vs 359.18 for shared layers alone (a ~67 us gather does not pay for the region's cost).
_QPUSH_FULL = os.environ.get("GLM_QPUSH_FULL", "0") == "1"


class _DeferredQ:
    """A Q stem whose ops are issued only when the attention consumes it (glue track, GLUE_Q_DEFER_L1).

    `qr` (the q_a latent) is kept so that a caller can issue the q_b matmul early (Q-stem || push overlap) and hand
    its output to the rest of the stem through `q_b_out`."""

    __slots__ = ("_build", "qr", "q_b_out")

    def __init__(self, build, qr=None):
        self._build = build
        self.qr = qr
        self.q_b_out = None

    def __call__(self):
        return self._build(self.q_b_out)


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
        # GLM-5.3 shared layers are sparse but own no indexer weights (they reuse a prior full layer's
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
        llama4_scale_cache: Optional[dict] = None,
        sparse_mla_overlap_profile: Optional[str] = None,
        tp_shard_kv: Optional[bool] = None,
        sparse_kv_push: Optional[bool] = None,
        seq_sharded_q: Optional[bool] = None,
    ):
        # DSA indexer weights (v3.2 / GLM): extract NON-mutating, so the caller's state_dict survives
        # repeated construction / cache build+load (the old pop() emptied it on the first pass). Dense
        # v3.1 has none. Sparse capability is resolved below via resolve_has_indexer (config DSA fields /
        # host weights / complete cache) — never from the mere presence of these keys — so cache-only
        # construction stays sparse instead of silently going dense.
        idx_host = TtIndexer.extract_host_weights(state_dict)
        self.config = config
        self.mesh_device = mesh_device
        self._is_blackhole = mesh_device.arch() == ttnn.Arch.BLACKHOLE
        # Optional segmented-trace controller. Sparse MLA uses it to split capture around the
        # top-k/KV-gather sub-device-manager load and clear, which cannot occur inside one trace.
        self._trace_controller = None
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
        # self.tp_shard_kv (KV dedup) is DERIVED further down, once _has_indexer and tp_factor are known --
        # it is not a constructor argument. See the note at its assignment.
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
                "qkv_a_proj",  # merged [q_a | kv_a | pad] stem (_merge_qkv_a)
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

        # Not every YaRN model folds the mscale amplitude into the softmax scale. Mistral Small 4
        # carries a full YaRN block (factor=128) but its own implementation uses the bare
        # qk_head_dim**-0.5 and reports attention_scaling = 1.0, so no mscale is applied anywhere.
        # Applying DeepSeek's correction anyway multiplies the attention logits by mscale^2 (2.2058
        # at factor=128): no crash, just a wrong softmax temperature. Absent (-> False) on every
        # other variant, so those paths are byte-identical.
        self.scale = self.qk_head_dim**-0.5
        if rope_factor is not None and rope_factor > 1.0 and not getattr(config, "mla_disable_yarn_mscale", False):
            mscale = rope_scaling["mscale"]
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        assert not (self._use_nope and self.scale != self.qk_head_dim**-0.5), (
            f"mla_use_nope=True but rope_scaling carries a YaRN factor ({rope_factor}) that scaled "
            f"softmax to {self.scale}; a NoPE model has no positional scaling to compensate for"
        )

        # Mistral's query temperature. Only Mistral's rope_scaling carries it, so absent (-> None)
        # leaves every other variant's op graph byte-identical.
        self._llama4_beta = rope_scaling.get("llama_4_scaling_beta")
        self._llama4_orig_max = rope_scaling.get("original_max_position_embeddings")
        # Shared across layers when the caller threads one dict down (TtPrefillTransformer does);
        # a bare ttMLA keeps its own. Contents are layer-invariant -- see _llama4_scale.
        self._llama4_cache: dict = llama4_scale_cache if llama4_scale_cache is not None else {}

        self.default_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Same HiFi2 fidelity with an fp32 dest, for tuned matmuls that set fp32_dest_acc. A wide
        # in0_block_w accumulates that much more K in dest before the packer spills; in bf16 dest that
        # costs precision (Mistral q_a_proj at in0_block_w=8: 1.4% rel. error vs 0.86% default), in fp32
        # it is both faster and more accurate than the default (0.44%).
        self.hifi2_fp32_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
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
        sdpa_fidelity = getattr(config, "mla_chunked_sdpa_matmul_fidelity", None)
        self.sdpa_matmul_fidelity = getattr(ttnn.MathFidelity, sdpa_fidelity) if sdpa_fidelity and is_chunked else None
        self.clamp_attention_to_valid_end = bool(getattr(config, "mla_chunked_clamp_attention_to_valid_end", False))

        # Create CCL object for semaphore management
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.tp_factor = mesh_device.shape[self.tp_axis]
        self.sp_factor = mesh_device.shape[self.sp_axis]
        assert (
            self.active_seq_len % self.sp_factor == 0
        ), f"active_seq_len ({self.active_seq_len}) must divide SP factor ({self.sp_factor})"
        self.active_seq_len_local = self.active_seq_len // self.sp_factor

        self.ccl_num_links = 2 if self._is_blackhole else 1  # Blackhole trains 2 fabric routing planes, others 1
        # Sparse layers fed another layer's top-k indices (GLM shared layers) receive only the KV rows those indices
        # reference, pushed by their owners, instead of gathering the whole prefix (see sparse_kv_push.py). The push
        # routes over all four torus links of every chip, so other fabrics keep the dense gather; False forces the
        # dense gather (a reference for tests).
        if sparse_kv_push is None:
            sparse_kv_push = self._has_indexer and sparse_kv_push_supported(mesh_device)
        assert not sparse_kv_push or sparse_kv_push_supported(mesh_device), (
            f"the sparse KV push needs FABRIC_2D_TORUS_XY on a mesh of at least 3x3, got {ttnn.get_fabric_config()} "
            f"on {tuple(mesh_device.shape)}"
        )
        self._sparse_kv_push = sparse_kv_push

        # The TP high-bandwidth all-gathers operate on the fixed prefill chunk, never on the full
        # growing KV-cache. Allocate their worst-case outputs once at construction and share them
        # across serial MLA layers through TT_CCL; forward only reuses these stable addresses.
        self._q_a_latent_gather_output = None
        self._kv_stem_gather_output = None
        self._output_gate_gather_output = None
        # q_a and kv_a as one matmul + one all-reduce over [q_a_proj | kv_a_proj_with_mqa | pad].
        # GLUE_MERGE_QKV_A=1 (glue track) also opts DSA/indexer models (GLM-5.3) in: the indexer only
        # reads the normed q_a latent (qr), which the merged path still produces before the indexer runs.
        _glue_merge = os.environ.get("GLUE_MERGE_QKV_A", "0") == "1"
        self._merge_qkv_a = (
            (bool(getattr(config, "mla_merge_qkv_a", False)) or _glue_merge)
            and self.tp_factor > 1
            and not self.kv_only
            and (_glue_merge or not self._has_indexer)
            and not self._use_gate
        )
        if self._merge_qkv_a:
            qkv_a_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            self._qkv_a_pad = -qkv_a_width % (self.tp_factor * ttnn.TILE_SIZE)
            self._qkv_a_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                name="qkv_a",
                shape=[1, 1, self.active_seq_len_local, qkv_a_width + self._qkv_a_pad],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        if self.tp_factor > 1:
            if not self.kv_only and not self._merge_qkv_a:
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
        # SDPA waits forever on a missing wrap link. See tt_ccl.resolve_per_axis_topology, which the V4
        # attention blocks and compressors share with this.
        self.sp_ccl_topology, self.tp_ccl_topology = resolve_per_axis_topology(topology, self.sp_axis, self.tp_axis)

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
        if self._merge_qkv_a:
            self.qkv_a_proj_weight = self._concat_qkv_a_weight()
        # glue track: q_a norm as a TP-distributed fused RMSNorm on the reduce-scatter output, BEFORE the
        # all-gather (the replicated ttnn.rms_norm on [S, q_lora_rank] runs on one core per tile-row).
        self._q_distnorm = (
            os.environ.get("GLUE_Q_DISTNORM", "1") == "1"
            and self.tp_factor > 1
            and not kv_only
            and not self._merge_qkv_a
            and self._llama4_beta is None
        )
        if self._q_distnorm:
            self._q_a_norm_weight_tp = self._tp_shard_norm_weight(self.q_a_layernorm_weight, self.q_lora_rank)

        # DSA indexer (v3.2 / GLM): self._has_indexer was resolved above (before the buffer alloc). The
        # TtIndexer owns the indexer stems / RoPE tables / device key-cache and reuses this MLA's q_a stem
        # + collectives. Inert for dense v3.1.
        # DSA *family* (config carries the indexer fields), independent of whether the indexer is active
        # this layer. V3.1's dense config lacks them; V3.2's config has them even when a benchmark forces
        # the attention dense (has_indexer=False). Dense-path tuning gates that must tell V3.1 from a
        # dense-run V3.2 key on this, not _has_indexer (see _get_sdpa_program_config).
        self._is_dsa_family = TtIndexer.matches_config(config)
        self._sparse_kv_gather_buffer = None

        self.set_tp_shard_kv(tp_shard_kv)
        if self._has_indexer:
            assert self.tp_factor > 1, (
                f"the sparse (DSA) path requires tp_factor > 1 (got {self.tp_factor}): its KV and indexer-key "
                "caches are deduped across SP*TP and the full-mesh gather has no TP leg to reassemble at tp=1"
            )
            # 2D fabric, checked HERE as well as in the gather. The deduped KVPE gather is one snake across
            # both mesh axes, so only a 2D fabric can route it -- a narrowing introduced when the SP-only
            # (single-axis, 1D-routable) gather was removed. The gather asserts this too, but that fires on
            # the first chunk, which on a deep model is a long weight load later; fail at construction.
            _fabric = ttnn.get_fabric_config()
            assert _fabric in (
                ttnn.FabricConfig.FABRIC_2D,
                ttnn.FabricConfig.FABRIC_2D_TORUS_X,
                ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            ), (
                f"the sparse (DSA) path requires a 2D fabric config, got {_fabric}: its only KVPE gather is "
                "the full-mesh snake, which spans both mesh axes. Open the mesh with FABRIC_2D (or a 2D "
                "torus variant)."
            )
        # KV dedup shards the cache over BOTH axes, so its gather needs the scratch even at sp == 1.
        if self._has_indexer and not self.kv_only:
            kvpe_row_width = self.sparse_kv_cache_format.storage_width(self.kv_cache_geometry)
            self._sparse_kv_gather_buffer = self.tt_ccl.get_mla_sparse_kv_gather_buffer(
                seq_len=seq_len,
                row_width=kvpe_row_width,
                dtype=self.sparse_kv_cache_format.storage_dtype,
                layout=self.sparse_kv_cache_format.storage_layout,
            )
        # GLM-5.3 indexer reuse: a "shared" layer is sparse but owns no indexer weights — it reuses the
        # most recent "full" layer's top-k indices, injected at forward, and binds a weight-less
        # ReuseIndexer (never computes). Absent indexer_types (v3.1 / v3.2) every layer is
        # "full" -> current behavior, unchanged.
        self._indexer_reuse = indexer_layer_is_reused(config, layer_idx)
        requested_overlap_profile = sparse_mla_overlap_profile
        if requested_overlap_profile is None:
            requested_overlap_profile = os.getenv("TT_SPARSE_MLA_OVERLAP_PROFILE")
        if requested_overlap_profile is None:
            requested_overlap_profile = getattr(config, "sparse_mla_overlap_profile", None)
        if requested_overlap_profile is not None and requested_overlap_profile.lower() in ("", "0", "off", "none"):
            requested_overlap_profile = None

        self._sparse_mla_overlap = None
        # With the sparse KV push, a layer whose successor reuses the same top-k pushes that successor's prefix rows
        # during its own sparse_sdpa (attention and push on disjoint core grids of one program); the successor then
        # pushes only its remainder. The core grids and semaphores are the overlap profile's.
        self._push_prefetch = None
        if (
            self._sparse_kv_push
            and requested_overlap_profile is not None
            and self._has_indexer
            and not self.kv_only
            # layer_num counts this stage's cache slots, so compare the successor's LOCAL index: a pipeline
            # stage starting past layer 0 would otherwise never prefetch.
            and layer_idx - (first_layer_idx or 0) + 1 < self.layer_num
            and indexer_layer_is_reused(config, layer_idx + 1)
        ):
            self._push_prefetch = self.tt_ccl.get_sparse_mla_overlap_resources(requested_overlap_profile)
        # sparse_sdpa is bound by DRAM bandwidth (indexed KV row reads), so cores beyond what its query rows need only
        # add contention: on the 12x10 Galaxy grid a 5k chunk (160 rows per chip) runs 8% faster on the overlap
        # profile's 96-core top-k grid than on all 120 cores. Used whenever it holds as few rows per core as the full
        # grid (see _sparse_sdpa_grid).
        self._sparse_sdpa_cores = None
        if requested_overlap_profile is not None and self._has_indexer and not self.kv_only and self.sp_factor > 1:
            self._sparse_sdpa_cores = self.tt_ccl.get_sparse_mla_overlap_resources(
                requested_overlap_profile
            ).topk_core_grid
        if requested_overlap_profile is not None:
            if not self._has_indexer or self.kv_only or self._indexer_reuse:
                # Only a non-kv-only full indexer layer can launch the two independent branches. Shared
                # layers receive already-finalized indices; dense and kv-only layers never select locally.
                pass
            else:
                if self.sp_factor == 1:
                    raise ValueError("sparse MLA overlap requires SP > 1; SP=1 uses an unconstrained slice path")
                self._sparse_mla_overlap = self.tt_ccl.get_sparse_mla_overlap_resources(requested_overlap_profile)
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
                    output_tp_sequence_sharded=self._needs_head_to_seq_reshard,
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
        self._use_indexed_rope = not self._use_nope and (self.is_chunked or self._has_indexer)
        if self._use_nope:
            assert not self._has_indexer, (
                "mla_use_nope with a DSA indexer is not supported: TtIndexer applies its own rope to "
                "the indexer queries/keys, so the rope tensors cannot be dropped wholesale"
            )
            self._apply_rope = self._apply_rope_none
        else:
            self._apply_rope = self._apply_rope_padded if self._use_indexed_rope else self._apply_rope_one_shot

        # glue track: absorbed-Q stem without the 512-channel concat copy. wkv_b1 runs with its N padded by
        # qk_rope_head_dim zero columns, straight into the [.., kv_lora_rank + rope] Q buffer, and the indexed
        # RoPE writes the rotated q_pe into those last channels in place (output_tensor=). Needs the
        # rotary_embedding_indexed output_tensor mode (C++); GLUE_Q_ROPE_INPLACE=0 restores concat_prefix.
        self._q_rope_inplace = (
            os.environ.get("GLUE_Q_ROPE_INPLACE", "1") == "1"
            and not kv_only
            and self._use_indexed_rope
            and self._is_blackhole
            and self._resolve_mm_cfg("wkv_b1", self.active_seq_len_local) is not None
        )
        if self._q_rope_inplace:
            self.wkv_b1_weight_padded = self._pad_wkv_b1_weight()
            self._wkv_b1_padded_pc = self._padded_wkv_b1_program_config(
                self._resolve_mm_cfg("wkv_b1", self.active_seq_len_local)["program_config"]
            )

        # Bind the attention core once, by config. Sparse ALWAYS uses the block-cyclic
        # _sparse_chunked_attn (single-shot = one full-seq chunk); dense splits by chunking. forward()
        # then calls self._attention(...) with no mode ladder: the decision is made here, not per call.
        if self._has_indexer:
            self._attention = self._sparse_chunked_attn
        else:
            self._attention = self._dense_chunked_attn if self.is_chunked else self._dense_single_attn

        # Sequence-sharded absorbed Q (see _q_stem_seq). Resolved last (it needs the rope binding), after the
        # weights load, because it swaps the head-sharded q_b / wkv_b1 / wkv_b2 for TP-replicated copies.
        self._seq_q = self._resolve_seq_sharded_q(seq_sharded_q)
        self._seq_q_out = os.getenv("MLA_SEQQ_OUT", "heads")
        assert self._seq_q_out in ("heads", "cols"), f"MLA_SEQQ_OUT must be heads or cols, got {self._seq_q_out}"
        # Debug/reference: MLA_SEQQ_QA=allreduce keeps the head-sharded path's q_a stem (all-reduce + norm of all
        # S/sp rows) and partitions the rows afterwards, isolating the reduce-scatter summation order.
        self._seq_q_qa = os.getenv("MLA_SEQQ_QA", "rows")
        # MLA_SEQQ_L1 (default on; =0 disables): the tilized sparse_sdpa output (~87 KB/core, wkv_b2's in0) lives in L1,
        # only for the wkv_b2 matmul (R0, bit-exact: tilize + wkv_b2 115 -> 84 us; R3 ABAB on best-v9 -0.50%). The
        # rope output must NOT go to L1: it is built before the kv stem and lives through the KVPE gather / top-k
        # overlap region, and on best its 98 KB/core clashed with the gather/push programs' static CBs (grid rows
        # 0-1), eager and traced (seqq-l1-b6-4).
        self._seq_q_l1 = self._seq_q and os.getenv("MLA_SEQQ_L1", "1") not in ("", "0")
        assert self._seq_q_qa in ("rows", "allreduce"), f"MLA_SEQQ_QA must be rows or allreduce, got {self._seq_q_qa}"
        if self._seq_q:
            self.q_b_proj_weight = self._replicate_over_tp(self.q_b_proj_weight, dim=-1)
            self.wkv_b1_weight = self._replicate_over_tp(self.wkv_b1_weight, dim=1)
            self.wkv_b2_weight = self._replicate_over_tp(self.wkv_b2_weight, dim=1)
        # mm track (default on; MLA_SEQQ_QB_WS=0 disables): lays the TP-replicated seqq q_b weight ([2048, 16384] bfp8, 35.6 MB/chip, the
        # weight-stream-bound 1D mcast_in0 matmul) out DRAM WIDTH_SHARDED ([K, N/banks] per bank; a device tile copy,
        # no requantization) so the matmul reads it bank by bank, one request per worker K-row slice, and runs it with
        # in0_block_w=MLA_SEQQ_QB_IB (default 2). R0: 113.0 -> 95.2 us (PCC 0.99996 vs 0.99993; the smaller K block
        # changes the packer-L1 accumulation order, so not bit-exact).
        self._seq_qb_ws_pc = None
        if self._seq_q and self._is_blackhole and os.getenv("MLA_SEQQ_QB_WS", "1") not in ("", "0"):
            qb_rows = self.active_seq_len_local // self.tp_factor
            qb_cfg = self._resolve_mm_cfg("q_b_proj", qb_rows)
            pc = qb_cfg["program_config"] if qb_cfg is not None else None
            if isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig) and pc.mcast_in0:
                ib = int(os.getenv("MLA_SEQQ_QB_IB", "2"))
                self.q_b_proj_weight = self._dram_width_shard(self.q_b_proj_weight)
                self._seq_qb_ws_rows = qb_rows
                self._seq_qb_ws_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
                    in0_block_w=ib,
                    out_subblock_h=pc.out_subblock_h,
                    out_subblock_w=pc.out_subblock_w,
                    per_core_M=pc.per_core_M,
                    per_core_N=pc.per_core_N,
                    fuse_batch=pc.fuse_batch,
                    fused_activation=pc.fused_activation,
                    mcast_in0=True,
                )
        if _MLA_OPROJ_WS and self._is_blackhole and not kv_only:
            o_cfg = self._resolve_mm_cfg("o_proj", self.active_seq_len_local)
            o_pc = o_cfg["program_config"] if o_cfg is not None else None
            if isinstance(o_pc, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
                self.o_proj_weight = self._dram_width_shard(self.o_proj_weight)
        # glue track: in-place RoPE for the sequence-sharded Q stem too (see _q_stem_seq). The head-path padded
        # weight built above is unused once Q is sequence-sharded; rebuild it from the TP-replicated wkv_b1.
        self._q_rope_inplace_seq = False
        self._q_defer_l1 = False
        if self._seq_q:
            if self._q_rope_inplace:
                ttnn.deallocate(self.wkv_b1_weight_padded)
                self.wkv_b1_weight_padded = None
                self._q_rope_inplace = False
            rows = self.active_seq_len_local // self.tp_factor
            cfg = self._resolve_mm_cfg("wkv_b1", rows) if self._is_blackhole else None
            if os.environ.get("GLUE_Q_ROPE_INPLACE", "1") == "1" and cfg is not None:
                self.wkv_b1_weight_padded = self._pad_wkv_b1_weight()
                self._wkv_b1_padded_pc_seq = self._padded_wkv_b1_program_config(cfg["program_config"])
                self._q_rope_inplace_seq = True
                self._q_defer_l1 = _GLUE_Q_DEFER_L1 and self._has_indexer and not self.kv_only
        # Q-stem || push overlap (GLM_QPUSH_OVERLAP): (overlap resources, offset q_b program config) on shared layers.
        self._qpush = None
        if (
            _QPUSH_OVERLAP
            and self._q_defer_l1
            and (self._indexer_reuse or (_QPUSH_FULL and _GLUE_Q_DEFER_FULL and self._push_prefetch is not None))
            and self._sparse_kv_push
            and requested_overlap_profile is not None
            and self.sp_factor > 1
        ):
            resources = self.tt_ccl.get_sparse_mla_overlap_resources(requested_overlap_profile)
            pc = self._qpush_q_b_program_config(resources)
            if pc is not None:
                self._qpush = (resources, pc)
        if os.getenv("MLA_MEMLOG") and layer_idx in (0, layer_num - 2, layer_num - 1):
            self._log_dram(f"after MLA layer {layer_idx} weights (seq_q={self._seq_q})")

    def release_sparse_mla_overlap_manager(self) -> None:
        """Forward shared sparse-MLA overlap teardown to the model-wide TT_CCL owner."""
        self.tt_ccl.release_sparse_mla_overlap_manager()

    def set_trace_controller(self, controller) -> None:
        """Route overlap-manager boundaries through a segmented trace controller when attached."""
        self._trace_controller = controller

    @staticmethod
    def kv_cache_to_host(kvpe_cache: MlaKvCache, mesh_device: ttnn.MeshDevice, sp_axis: int = 0):
        """Read and decode the logical KVPE cache in natural sequence order.

        Which reassembly applies is read off the cache's OWN declared distribution rather than passed in,
        because that is what the writer and the gathers key on too: init_kvpe_cache stamps a rank-2 dim-2
        sharding when the rows are striped over every mesh coordinate (KV dedup via tp_axis, or full_mesh
        -- the same striping by two names) and a rank-1 Replicate otherwise. Under the rank-2 topology the
        second axis holds DISTINCT rows, so collapsing it as a replica would return seq_len/tp of the cache.
        """
        storage = kvpe_cache.storage
        if len(list(storage.tensor_topology().distribution_shape())) == 2:
            n0, n1 = mesh_device.shape[0], mesh_device.shape[1]
            sharded = ttnn.to_torch(
                storage,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
            )  # [slots, n1, n0 * rows_per_chip, width]
            rows = sharded.shape[2] // n0
            # Row-major chip order (axis 0 outer), matching the coordinate order init_kvpe_cache declares.
            host = torch.cat(
                [sharded[:, a1 : a1 + 1, a0 * rows : (a0 + 1) * rows] for a0 in range(n0) for a1 in range(n1)],
                dim=2,
            )
        else:
            host = ttnn.to_torch(
                storage,
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
        "qkv_a_proj": ttnn.bfloat16,
    }

    # Matmul dimensions for batched matmuls (wkv_b1 / wkv_b2) keyed by weight name.
    # Each entry: (K_attr, N_attr) where values are attribute names on self.
    _BATCHED_MM_DIMS = {
        "wkv_b1": ("qk_nope_head_dim", "kv_lora_rank"),
        "wkv_b2": ("kv_lora_rank", "v_head_dim"),
    }

    def _cfg_matches(self, cfg: dict, kt: int | None = None) -> bool:
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
        # dimensionally valid at Kimi's q_lora_rank (1536) but overflow the grid at GLM-5.3's (2048), even
        # though both have 64 heads. When a config declares a q_lora_rank that doesn't match this model,
        # fall back so a same-heads/same-seq variant doesn't pick up an invalid program_config.
        if cfg.get("q_lora_rank") not in (None, self.q_lora_rank):
            return False
        # Per-device K/N follow the TP split, so a tiling sized for TP=4 overflows the grid at TP=1
        # (Mistral's PP=4 stages run [8,1] at the same 640-row seq_len_local as its 8x4 single rank).
        if cfg.get("tp_factor") not in (None, self.tp_factor):
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
        # K. The table is keyed on (weight_name, seq_len_local), so one slot is shared by variants
        # with different K. A tiling whose in0_block_w does not divide this model's per-device Kt
        # cannot run here at all -- the matmul dies with "Kt (32) must be divisible by in0_block_w
        # (14)" -- so reject it like any other tag and let a later candidate, e.g. one tuned for this
        # variant, be chosen. kt is None on the SDPA path, which has no weight to size against.
        if kt is not None:
            in0_block_w = getattr(cfg.get("program_config"), "in0_block_w", None)
            if in0_block_w is not None and kt % in0_block_w != 0:
                return False
        return True

    def _select_cfg(self, entry, weight_name: str | None = None) -> dict | None:
        """Pick the first tuned config whose tags match, from a single dict or a list of candidates.

        A slot holds several candidates when variants share a seq_len (Kimi-K2.7 at 64 heads and K3 at
        96 both want ``640``), because the tags only reject -- they cannot choose. List order is
        priority order; most specific first.
        """
        if entry is None:
            return None
        candidates = entry if isinstance(entry, (list, tuple)) else (entry,)
        kt = self._weight_kt(weight_name) if weight_name is not None else None
        return next((cfg for cfg in candidates if self._cfg_matches(cfg, kt)), None)

    def _resolve_mm_cfg(self, weight_name: str, seq_len_local: int) -> dict | None:
        """Resolve the tuned matmul config for this weight/seq_len, applying the gating tags.
        Returns None when no tuned config applies (caller falls back to defaults)."""
        if not self._is_blackhole:
            return None
        table = self.mm_configs[weight_name]
        entry = table.get(seq_len_local)
        if entry is None:
            # Native chunks smaller than the tuned 5120 (GLM only): derive the missing slot (GLM_SMALL_MM_CFG, default on).
            glm = self.num_heads == 64 and self.q_lora_rank == 2048
            entry = small_rows_entry(table, seq_len_local, weight_name, self.active_seq_len_local if glm else None)
        return self._select_cfg(entry, weight_name)

    def _weight_kt(self, weight_name: str) -> int | None:
        """This model's per-device K for one weight, in tiles, or None if the weight is absent.

        Weights are [K, N] and ``mapper_tp0`` shards K across the TP axis, so ``shape[-2]`` is
        already the per-device extent the matmul program consumes.
        """
        weight = getattr(self, f"{weight_name}_weight", None)
        return None if weight is None else weight.shape[-2] // ttnn.TILE_SIZE

    def _get_act_mem_config(self, weight_name: str, seq_len_local: int) -> ttnn.MemoryConfig:
        """Memory config for the activation (in0) feeding this weight's matmul, as tuned in the mm
        config (act_mem_config). Defaults to DRAM when no tuned config applies."""
        cfg = self._resolve_mm_cfg(weight_name, seq_len_local)
        return cfg["act_mem_config"] if cfg is not None else ttnn.DRAM_MEMORY_CONFIG

    def _get_mm_kwargs(self, weight_name: str, seq_len_local: int) -> dict:
        """Get matmul kwargs from config, falling back to defaults.

        Always carries compute_kernel_config: HiFi2 by default, or HiFi2 with an fp32 dest when the
        tuned config sets ``fp32_dest_acc``. Pinned either way -- a program_config with no explicit
        compute config silently drops the matmul to LoFi.
        """
        cfg = self._resolve_mm_cfg(weight_name, seq_len_local)
        if cfg is None:
            if weight_name in self._BATCHED_MM_DIMS:
                return self._make_batched_mm_kwargs(weight_name, seq_len_local)
            return {
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "dtype": self.MM_DEFAULT_DTYPES[weight_name],
                "compute_kernel_config": self.default_compute_kernel_config,
            }
        return {
            "memory_config": cfg["out_mem_config"],
            "program_config": cfg["program_config"],
            "dtype": cfg["out_dtype"],
            "compute_kernel_config": (
                self.hifi2_fp32_compute_kernel_config
                if cfg.get("fp32_dest_acc")
                else self.default_compute_kernel_config
            ),
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
            "compute_kernel_config": self.default_compute_kernel_config,
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
            matmul_math_fidelity=self.sdpa_matmul_fidelity,
        )

    def _apply_rope_padded(
        self,
        t: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: int,
        metadata: Optional[ttnn.Tensor] = None,
        concat_prefix: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Chunked rotated RoPE via the indexed op. rope_tensors carry the whole-cache,
        block-cyclic-sharded cos/sin (built once via RotarySetup.get_rope_tensors_indexed); the op
        derives this chunk's per-chip shard offset on-device from kv_actual_global -- the same
        update_idxt math the KV-cache writer uses, keeping rotation and cache write consistent.

        Per-element-tensor (trace-safe) path: `metadata` is a 3-tuple of 1-element uint32 tensors
        (slot_id, actual_start, actual_end); rope reads kv_actual_global = actual_start = metadata[1].
        """
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            t,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=metadata[1] if metadata is not None else kv_actual_isl,
            cluster_axis=self.sp_axis,
            concat_prefix=concat_prefix,
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
        actual_end: Optional[int] = None,
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
        # Metadata (trace-safe) path reads slot_idx/kv_actual_global on-device from the metadata tensor.
        self._update_kv_cache(
            kvpe_cache,
            tt_kvpe,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            actual_end=actual_end,
            metadata=metadata,
            tp_axis=self.tp_shard_kv_axis,  # KV dedup: write only this chip's 1/tp window
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
        assert is_tp_deduped_kv_cache(kvpe_cache.storage, (self.sp_factor, self.tp_factor)) == self.tp_shard_kv, (
            f"KVPE cache is {'not ' if self.tp_shard_kv else ''}TP-deduped but this layer has tp_shard_kv="
            f"{self.tp_shard_kv}; allocate the cache to match (set_tp_shard_kv does not re-layout it)"
        )
        # Dedup splits the per-chip depth over both axes.
        cache_rows_global = kvpe_cache.storage.shape[2] * self.sp_factor * (self.tp_factor if self.tp_shard_kv else 1)
        if metadata is not None:
            meta_slot_kwargs = {
                "slot_id": metadata[0],
                "kv_actual_isl_tensor": metadata[1],
                "kv_cache_num_layers": self.layer_num,
                "kv_cache_layer_idx": cache_layer_idx,
            }
            if self.clamp_attention_to_valid_end:
                meta_slot_kwargs["kv_valid_end_tensor"] = metadata[2]
            ring_logical_n = cache_rows_global  # global cache capacity
        else:
            meta_slot_kwargs = {"kv_cache_batch_idx": cache_batch_idx, "kv_actual_isl": kv_actual_isl}
            # Capped at the capacity ring_mla accepts: the last chunk's pad rows can sit past the cache
            # end, and only pad rows read them.
            ring_logical_n = min(kv_actual_isl + chunk_size_global, cache_rows_global)
        if self.tp_shard_kv:
            self._declare_q_sequence_topology(tt_q, seq_len_local)
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
            cluster_axis=None if self.tp_shard_kv else self.sp_axis,
            mesh_device=self.mesh_device,
            # Full mesh asks for a ring; the route decides whether it closes.
            topology=ttnn.Topology.Ring if self.tp_shard_kv else self.sp_ccl_topology,
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
            **self._get_mm_kwargs("q_a_proj", seq_len_local),
        )

        if self._q_distnorm:
            return self._q_a_latent_distnorm(qr, seq_len_local)
        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            qr = self._tp_all_reduce(qr, self._q_a_latent_gather_output, seq_len_local)
        return self._q_a_norm(qr, norm_memory_config)

    def _tp_reduce_scatter(self, x: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        assert (
            seq_len_local == self.active_seq_len_local
        ), f"TP gather was preallocated for {self.active_seq_len_local} local tokens, got {seq_len_local}"
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
            # Default is 2 workers below 1 MB per link; 8 measured 42.6 -> 38.5 us at [640,2048] (glue R0).
            num_workers_per_link=_GLUE_STEM_RS_WORKERS,
        )

    def _tp_all_reduce(self, x: ttnn.Tensor, gather_output: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        x = self._tp_reduce_scatter(x, seq_len_local)
        return ttnn.experimental.high_bw_all_gather(
            x,
            dim=3,
            output_tensor=gather_output,
            cluster_axis=self.tp_axis,
            num_links=self.ccl_num_links,
        )

    def _tp_shard_norm_weight(self, weight: ttnn.Tensor, width: int) -> ttnn.Tensor:
        """Replicated ROW_MAJOR [1, 1, width/32, 32] norm weight -> TP-sharded TILE [1, 1, 1, width/tp] row,
        the affine-weight form dit_fused_distributed_rmsnorm broadcasts (same as TtDistributedRmsNorm)."""
        host = ttnn.to_torch(ttnn.get_device_tensors(weight)[0]).reshape(1, 1, 1, width)
        dims = [None, None]
        dims[self.tp_axis] = 3
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
        )

    def _q_a_latent_distnorm(self, qr: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """RS -> TP-distributed fused RMSNorm on this chip's [S, q_lora/tp] slice -> AG of the normed latent.
        Row-wise RMSNorm commutes with the column gather once the row statistics are all-reduced, which the
        fused op does with one small stats all-gather. Returns the model-owned persistent gather buffer:
        callers must NOT deallocate it."""
        x = self._tp_reduce_scatter(qr, seq_len_local)
        ttnn.deallocate(qr)
        semaphores, stats = self.tt_ccl.get_fused_rmsnorm_resources(
            x, self._q_a_norm_weight_tp, self.tp_axis, self.ccl_num_links
        )
        normed = ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            self.tp_axis,
            self.mesh_device,
            semaphores,
            topology=self.tp_ccl_topology,
            epsilon=self.config.rms_norm_eps,
            weight=self._q_a_norm_weight_tp,
            persistent_output_buffer=stats,
            num_preferred_links=self.ccl_num_links,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        out = ttnn.experimental.high_bw_all_gather(
            normed,
            dim=3,
            output_tensor=self._q_a_latent_gather_output,
            cluster_axis=self.tp_axis,
            num_links=self.ccl_num_links,
        )
        ttnn.deallocate(normed)
        return out

    def _q_a_norm(self, qr: ttnn.Tensor, norm_memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        return ttnn.rms_norm(
            qr,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=norm_memory_config,
            compute_kernel_config=self.default_compute_kernel_config,
        )

    def _padded_wkv_b1_program_config(self, b1) -> ttnn.MatmulMultiCoreReuseProgramConfig:
        """The tuned wkv_b1 batched config with N widened to kv_lora_rank + rope (whole N per core)."""
        n_t = (self.kv_lora_rank + self.qk_rope_head_dim) // ttnn.TILE_SIZE
        sub_w = max(w for w in range(1, 9) if n_t % w == 0)
        grid = b1.compute_with_storage_grid_size
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=b1.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            per_core_M=b1.per_core_M,
            per_core_N=n_t,
        )

    def _kv_l1(self) -> bool:
        """KV stem intermediates in L1 (glue, GLUE_KV_L1): sparse path only, where the fields are consumed by the cache
        write and freed right after it (_prepare_sparse_kv_prefix)."""
        return _GLUE_KV_L1 and self._has_indexer

    def _pad_wkv_b1_weight(self) -> ttnn.Tensor:
        """[.., nope, kv_lora_rank] bfp8 -> [.., nope, kv_lora_rank + rope] with zero rope columns. Whole
        tiles are appended along N, so the bfp8 -> bf16 -> bfp8 round trip keeps every block exact."""
        w = ttnn.typecast(self.wkv_b1_weight, ttnn.bfloat16)
        shape = list(w.shape)
        shape[-1] = self.qk_rope_head_dim
        zeros = ttnn.from_torch(
            torch.zeros(shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        merged = ttnn.concat([w, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(w)
        ttnn.deallocate(zeros)
        out = ttnn.typecast(merged, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return out

    def _concat_qkv_a_weight(self) -> ttnn.Tensor:
        # Both parts are whole tiles wide, so the bf8 round trip keeps every shared-exponent block intact.
        q, kv = self.q_a_proj_weight, self.kv_a_proj_with_mqa_weight
        parts = [ttnn.typecast(q, ttnn.bfloat16), ttnn.typecast(kv, ttnn.bfloat16)]
        if self._qkv_a_pad:
            parts.append(
                ttnn.from_torch(
                    torch.zeros(q.shape[-2], self._qkv_a_pad),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        merged = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in parts:
            ttnn.deallocate(t)
        weight = ttnn.typecast(merged, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return weight

    def _qkv_a_latent(
        self, hidden_states: ttnn.Tensor, seq_len_local: int, norm_memory_config: ttnn.MemoryConfig
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Merged _q_a_latent + the kv_a projection: returns (qr, (kv_nope, kv_rope))."""
        # Tuned config where one exists (GLM-5.3: L1 out, 2D mcast); otherwise the TTNN default to DRAM.
        x = ttnn.linear(
            hidden_states,
            self.qkv_a_proj_weight,
            **self._get_mm_kwargs("qkv_a_proj", seq_len_local),
        )
        x = self._tp_all_reduce(x, self._qkv_a_gather_output, seq_len_local)
        nope_start = self.q_lora_rank
        rope_start = nope_start + self.kv_lora_rank
        rope_end = rope_start + self.qk_rope_head_dim
        qr = ttnn.slice(x, [0, 0, 0, 0], [1, 1, seq_len_local, nope_start])
        kv_nope = ttnn.slice(x, [0, 0, 0, nope_start], [1, 1, seq_len_local, rope_start])
        kv_rope = ttnn.slice(x, [0, 0, 0, rope_start], [1, 1, seq_len_local, rope_end])
        qr_normed = self._q_a_norm(qr, norm_memory_config)
        ttnn.deallocate(qr)
        return qr_normed, (kv_nope, kv_rope)

    def _q_stem(
        self,
        qr: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int],
        seq_len_local: int,
        metadata: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Absorbed-Q stem from the q_a latent: q_b_proj → heads → split → wkv_b1(nope) → RoPE(rope)
        → query assembly. Consumes qr (the indexer, if any, has already read it by this point)."""
        num_heads_local = self.num_heads // self.tp_factor
        if self._llama4_beta is not None:
            # Mistral's per-position query temperature, applied to the normed q_a latent rather than to
            # Q: every step from here to Q (q_b_proj, head split, wkv_b1, RoPE) is linear or a per-row
            # rotation, so s * f(qr) == f(s * qr), and the latent is [S, 1024] in L1 against Q's
            # [8, S, 320] in DRAM. 8.2 us vs 27.1 at 8x4, and a 2.5x smaller scale buffer.
            scaled = ttnn.multiply(
                qr, self._llama4_scale(kv_actual_isl, seq_len_local, metadata), memory_config=qr.memory_config()
            )
            ttnn.deallocate(qr)
            qr = scaled
        tt_q = ttnn.linear(
            qr,
            self.q_b_proj_weight,
            **self._get_mm_kwargs("q_b_proj", seq_len_local),
        )
        if not self._q_distnorm:  # distnorm qr IS the persistent TP gather buffer
            ttnn.deallocate(qr)

        # The Q stem uses interleaved head creation. Write the two channel regions directly.
        inplace = self._q_rope_inplace and seq_len_local == self.active_seq_len_local
        tt_q_nope, tt_q_rope = ttnn.experimental.nlp_create_q_heads_split(
            tt_q,
            num_heads=num_heads_local,
            split_head_dim=self.qk_nope_head_dim,
            memory_config=ttnn.L1_MEMORY_CONFIG if (inplace and _GLUE_Q_HEADS_L1) else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_q)

        if inplace:
            # [q_nope @ wkv_b1 | 0] straight into the Q buffer, then RoPE the q_pe into its last channels.
            tt_q = ttnn.linear(
                tt_q_nope,
                self.wkv_b1_weight_padded,
                program_config=self._wkv_b1_padded_pc,
                memory_config=ttnn.L1_MEMORY_CONFIG if _GLUE_Q_OUT_L1 else ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.default_compute_kernel_config,
            )
            ttnn.deallocate(tt_q_nope)
            ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                tt_q_rope,
                rope_tensors["cos_matrix"],
                rope_tensors["sin_matrix"],
                rope_tensors["trans_matrix"],
                kv_actual_global=metadata[1] if metadata is not None else kv_actual_isl,
                cluster_axis=self.sp_axis,
                output_tensor=tt_q,
            )
            ttnn.deallocate(tt_q_rope)
            return tt_q

        tt_q_nope = ttnn.linear(
            tt_q_nope,
            self.wkv_b1_weight,
            **self._get_mm_kwargs("wkv_b1", seq_len_local),
        )

        if self._use_indexed_rope:
            # The indexed RoPE writer copies the absorbed channels into the same output.
            tt_q = self._apply_rope_padded(
                tt_q_rope, rope_tensors, kv_actual_isl, metadata=metadata, concat_prefix=tt_q_nope
            )
        else:
            rotated_q_rope = self._apply_rope(tt_q_rope, rope_tensors, kv_actual_isl, metadata=metadata)
            tt_q = ttnn.concat([tt_q_nope, rotated_q_rope], dim=-1)
            if rotated_q_rope is not tt_q_rope:
                ttnn.deallocate(rotated_q_rope)
        ttnn.deallocate(tt_q_nope)
        ttnn.deallocate(tt_q_rope)
        return tt_q

    def _llama4_scale(self, kv_actual_isl: Optional[int], seq_len_local: int, metadata) -> ttnn.Tensor:
        """Per-position query temperature for this chunk, shaped like tt_q, SP-sharded.

        Metadata/traced path reads ChunkMetadata.llama4_scale -- a captured graph can only read device
        memory, and write_chunk_metadata refreshes it alongside the scalars. Building a host tensor
        there would bake one chunk's offset into the capture. The host-scalar path builds it fresh and
        caches on (start, seq_len_local) in a dict shared by all layers, since the same offsets recur
        on every layer.

        The geometry is re-derived from this object's own axes rather than through
        rope._llama4_scale_geometry: that helper reads mesh_device.shape[1 - sp_axis] where this reads
        self.tp_factor, which are the same value on a 2-D mesh but reached from different state. Keep
        the two in step by hand -- a divergence shows up only as a copy/shape failure at runtime.

        NOTE ON RESIDENCY: the cache holds one [1, 1, chunk, q_lora_rank] bf16 tensor per distinct
        offset -- 1.31 MB per entry at 8x4 / chunk 5120 (3.28 MB before the scale moved from Q onto the
        q_a latent; the figures below were measured at that size), freed only with the model. Growth is linear in
        context depth, since an offset is visited once per request and never repeats.
        TtPrefillTransformer builds ONE dict and threads it down, so the layers pay that once. Measured
        at 102,400 tokens (20 offsets, L36 chunked): 20 tensors / 0.07 GB per device against 700 /
        2.30 GB per-layer, i.e. 2.23 GB per device off allocated DRAM. 700 and not 720 because chunked
        prefill builds the last layer kv_only and it never reaches _q_stem. Extrapolated, 1,048,576
        (MAX_POSITION_EMBEDDINGS, 204 offsets) is 0.67 GB shared against 23.4 GB, which left no room
        for weights and KV cache. Sharing is sound because every input to the tensor (offset, sp_factor,
        seq_len_local, heads_local, width, beta, orig_max) comes from the chunk, the config or the mesh;
        none varies by layer. The traced path never had the x36 problem: RotarySetup.make_llama4_scale_buffer
        allocates one buffer per runtime and all layers read that single ChunkMetadata.llama4_scale.
        It cannot fill it with rope.refresh_llama4_scale, which builds a host tensor -- a capture
        cannot -- so TtPrefillRuntime._prepare_llama4_scale_offsets pre-builds one buffer per
        chunk-aligned offset at compile() and device-to-device copies the right one in per chunk
        (#55126).

        A shared per-offset SET, not one buffer refreshed in place: an entry is never mutated, so "is
        another layer's enqueued multiply still reading this?" never arises. That is settled only for a
        device-to-device refresh (copy and replay both on cq 0) and open for a host write, which is why
        the sharing stops here.

        An LRU cap is NOT the answer: offsets never repeat within a request, so every chunk would
        miss and rebuild a 21 MB host tensor ([1, 1, 5120, 1024] fp32; 52 MB in the Q-shaped layout
        it was measured with), which measured 3x slower at long context.

        BUILDING THESE AT WARM-UP instead of lazily is the natural next step, and the transformer is
        already the owner that would do it. One entry costs 71.5 ms at 8x4 (11.3 ms host build +
        60.2 ms sharded from_torch, measured over 23 offsets), so the full set is 1.4 s at 102,400
        tokens and 14.6 s at 1,048,576. That is a relocation, not an addition: the same misses are
        paid today at one per chunk on whichever layer runs first. It buys fixed allocation addresses,
        which is what tracing the eager path and closing op2op gaps will need.

        Prebuild only covers the chunk-aligned offsets k * chunk_size_global, though. _q_stem asks
        only for tile alignment (see the assert in _chunked_attn), and a rotated mid-slab start -- a
        continued request resuming at the previous turn's real token count -- is a key no warm-up loop
        can enumerate. Those still miss lazily, so a prebuild pass caps the common case without
        bounding the cache. Computing the scale on device from the offset scalar, the other option
        raised in review, is the one that would.

        Full width rather than [1, 1, S, 1] + broadcast: a width-1 TILE_LAYOUT operand is tile-padded
        to 32, and relying on bcast to read only column 0 is not worth the risk.
        """
        if metadata is not None:
            buf = getattr(metadata, "llama4_scale", None)
            assert buf is not None, (
                "llama4 query scale on the metadata path needs ChunkMetadata.llama4_scale (allocate "
                "with RotarySetup.make_llama4_scale_buffer, refresh via write_chunk_metadata)"
            )
            return buf

        start = kv_actual_isl or 0
        key = (start, seq_len_local)
        cached = self._llama4_cache.get(key)
        if cached is not None:
            return cached

        scale = llama4_scale_host(
            start,
            self.sp_factor,
            seq_len_local,
            1,  # the q_a latent's shape; see rope._llama4_scale_geometry
            self.q_lora_rank,
            self._llama4_beta,
            self._llama4_orig_max,
        )
        shard_dims = [None, None]
        shard_dims[self.sp_axis] = 2
        tensor = ttnn.from_torch(
            scale.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
            ),
        )
        self._llama4_cache[key] = tensor
        return tensor

    def _kv_a_proj(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        # NOTE: input is ideally L1 for chunked, but hidden states memory config is set outside the module
        kv_mm_kwargs = self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local)
        # high_bw_all_gather directly streams its source from DRAM. Kimi's 640-token
        # matmul tune otherwise returns L1, while the fixed-slab TP path is active.
        # Request DRAM from the producer instead of adding a forward-time conversion
        # or falling back to the legacy all-gather.
        if self.tp_factor > 1:
            kv_mm_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            **kv_mm_kwargs,
        )

    def _kv_stem(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int],
        seq_len_local: int,
        return_kv_intermediates: bool,
        kvpe_cache: MlaKvCache,
        metadata: Optional[ttnn.Tensor] = None,
        kv_a: Optional[tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> tuple[ttnn.Tensor | tuple[ttnn.Tensor, ...], Optional[ttnn.Tensor], dict | None]:
        """Shared KV stem. kv_a is the already-reduced (kv_nope, kv_rope) from _qkv_a_latent, if merged.

        Returns packed cache values, or separate fields for fused sparse BF16/scaled-FP8
        cache writes. Sparse attention consumes the written cache; dense attention consumes packed values.
        """
        tt_kv = self._kv_a_proj(hidden_states, seq_len_local) if kv_a is None else None
        if kv_a is not None:
            tt_kv_nope, tt_kv_rope = kv_a
            kv_intermediates = (
                {"tt_kv": ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)} if return_kv_intermediates else None
            )
        # All reduce (skip for single-device TP)
        elif self.tp_factor > 1:
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
            # Reduce the TP partials directly into the two consumer tensors.
            tt_kv_nope, tt_kv_rope = ttnn.experimental.fast_reduce_nc_split(
                tt_kv,
                dim=1,
                split_output_width=self.kv_lora_rank,
                memory_config=ttnn.L1_MEMORY_CONFIG if self._kv_l1() else ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.hifi4_fp32_compute_kernel_config,
            )
            # tt_kv aliases model-owned gather scratch; keep that buffer alive.
            kv_intermediates = (
                {"tt_kv": ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)} if return_kv_intermediates else None
            )
        else:
            # TP=1 has no reduction producer into which to fuse the split.
            kv_intermediates = {"tt_kv": ttnn.clone(tt_kv)} if return_kv_intermediates else None
            tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len_local, self.kv_lora_rank])
            tt_kv_rope = ttnn.slice(
                tt_kv, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim]
            )
            ttnn.deallocate(tt_kv)

        tt_kv_nope = ttnn.rms_norm(
            tt_kv_nope,
            weight=self.kv_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG if self._kv_l1() else ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        tt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl, metadata=metadata)

        if return_kv_intermediates:
            # post-RMSNorm latent ([.., 512]) and post-RoPE k_pe ([.., 64]); clone before concat.
            kv_intermediates["tt_kv_nope"] = ttnn.clone(tt_kv_nope)
            kv_intermediates["tt_kv_rope"] = ttnn.clone(tt_kv_rope)

        if self._has_indexer and not return_kv_intermediates:
            # Keep separate fields alive until the cache writer packs them directly into storage.
            if kvpe_cache.format == MlaKvCacheFormat.BF16_RM:
                return (tt_kv_nope, tt_kv_rope), None, None
            if kvpe_cache.format == MlaKvCacheFormat.SCALED_FP8:
                fields = kvpe_cache.prepare_scaled_fp8_inputs(tt_kv_nope, tt_kv_rope, keep_rope_tiled=True)
                ttnn.deallocate(tt_kv_nope)
                if fields[2] is not tt_kv_rope:
                    ttnn.deallocate(tt_kv_rope)
                return fields, None, None
        tt_kvpe = kvpe_cache.pack(tt_kv_nope, tt_kv_rope, intermediates=kv_intermediates)
        ttnn.deallocate(tt_kv_rope)
        if self._has_indexer:
            # Sparse attention reads latent V from the cache; only dense attention needs this standalone tensor.
            ttnn.deallocate(tt_kv_nope)
            tt_kv_nope = None

        return tt_kvpe, tt_kv_nope, kv_intermediates

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int, out_l1: bool = False) -> ttnn.Tensor:
        kwargs = self._get_mm_kwargs("wkv_b2", seq_len_local)
        if out_l1:
            kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
        return ttnn.linear(
            t,
            self.wkv_b2_weight,
            **kwargs,
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
        declared_topology = kvpe_cache.storage.tensor_topology()
        ttnn.kv_cache.fill_cache_for_user_(kvpe_cache.storage, tt_kvpe, cache_layer_idx)
        kvpe_cache.storage.update_tensor_topology(declared_topology)

    def _update_kv_cache(
        self,
        cache: MlaKvCache,
        values: ttnn.Tensor | tuple[ttnn.Tensor, ...],
        *,
        cache_user_id: int,
        cache_layer_idx: int,
        kv_actual_isl: int,
        actual_end: Optional[int] = None,
        metadata: Optional[ttnn.Tensor] = None,
        tp_axis: Optional[int] = None,
    ) -> None:
        """``actual_end`` (end of this chunk's real tokens) clamps the write to them, so a chunk padding
        past the cache end needs only its real tokens to fit. Omitted, the whole padded slab is written."""
        # TP-sharded writes used to be scalar-only: the reader derived its 1/tp source window from the
        # HOST kv_actual_global, which the metadata path leaves at 0. The reader now reads that value
        # on-device from metadata[1] -- the same tensor the writer reads -- so the two cannot disagree and
        # the combination is supported. The op still validates that the tensor is present.
        # Metadata (trace-safe) path: slot_idx (metadata[0]) + kv_actual_global (metadata[1]) read
        # on-device, each its own 1-element tensor. Scalar path passes host slot/kv_actual_global.
        rope = scales = None
        if isinstance(values, tuple):
            if cache.format == MlaKvCacheFormat.SCALED_FP8:
                values, scales, rope = values
            else:
                values, rope = values
        if metadata is not None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache.storage,
                values,
                metadata[0],  # slot_idx tensor
                metadata[1],  # kv_actual_global tensor
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                cluster_axis=self.sp_axis,
                valid_global=metadata[2],  # actual_end tensor
                tp_axis=tp_axis,
                rope=rope,
                scales=scales,
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
                valid_global=actual_end,
                tp_axis=tp_axis,
                rope=rope,
                scales=scales,
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
        # Sequence-sharded Q with MLA_SEQQ_OUT=cols hands over [1, 1, S, heads*v], already concatenated.
        v_out = (
            attn_out
            if self._seq_q and self._seq_q_out == "cols"
            else ttnn.experimental.nlp_concat_heads(
                attn_out, memory_config=ttnn.L1_MEMORY_CONFIG if _GLUE_CONCAT_L1 else ttnn.DRAM_MEMORY_CONFIG
            )
        )
        if self._use_gate:
            assert hidden_states is not None, "gated MLA needs hidden_states to compute g_proj"
            g = self._output_gate(hidden_states, seq_len_local)
            v_out = ttnn.multiply(v_out, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
        o_kwargs = self._get_mm_kwargs("o_proj", seq_len_local)
        if _MLA_OPROJ_GRID12 and self._is_blackhole:
            o_kwargs = self._oproj_grid12(o_kwargs)
        v_out = ttnn.linear(
            v_out,
            self.o_proj_weight,
            **o_kwargs,
        )
        if self.tp_factor > 1:
            return ttnn.experimental.reduce_scatter_minimal_async(
                v_out,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                # glue track (GLUE_EPI_L1=1): consumed at once by the residual add.
                memory_config=ttnn.L1_MEMORY_CONFIG if _GLUE_EPI_L1 else ttnn.DRAM_MEMORY_CONFIG,
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
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
        return_indexer_indices: bool = False,
        metadata: Optional[ttnn.Tensor] = None,
        force_kv_only: bool = False,
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

        if self.kv_only or force_kv_only:
            return self._forward_kv_only(
                hidden_states,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx,
                kv_actual_isl=actual_start or 0,
                actual_end=actual_end,
                cache_user_id=cache_user_id,
                index_kv_cache=index_kv_cache,
                indexer_indices=indexer_indices,
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

        if os.getenv("MLA_MEMLOG") and not getattr(ttMLA, "_memlog_forward_done", False):
            ttMLA._memlog_forward_done = True  # once per process: everything (weights, caches) is allocated by now
            self._log_dram(f"first forward (layer {self.layer_idx})")

        # Preserve profiler region labels without duplicating them through Python logging.
        ttnn.tracy_message("`TT_SIGNPOST: MLA_START`")

        # q-norm output uses the tuned activation memory_config in every mode (dense and sparse);
        # the next op (q_b_proj) is the same matmul regardless of attention path.
        q_norm_mem_config = self._get_act_mem_config("q_b_proj", seq_len_local)
        # Compute the q_a latent once and share it: the DSA indexer reads it for its queries, then
        # _q_stem consumes it for q_b_proj — so the sparse path does not recompute the q_a stem.
        if self._seq_q and self._seq_q_qa == "allreduce":
            qr_all = self._q_a_latent(hidden_states, seq_len_local, q_norm_mem_config)
            qr, kv_a = ttnn.mesh_partition(qr_all, dim=2, cluster_axis=self.tp_axis), None
            if not self._q_distnorm:  # distnorm qr_all IS the persistent TP gather buffer
                ttnn.deallocate(qr_all)
        elif self._seq_q:
            # Only this TP rank's row quarter of the latent (see _q_a_latent_seq).
            q_norm_mem_config = self._get_act_mem_config("q_b_proj", seq_len_local // self.tp_factor)
            if self._q_defer_l1 and (indexer_indices is not None or _GLUE_Q_DEFER_FULL):
                # Deferred Q: qr outlives the kv stem and the push / top-k || gather region; keep it out of L1.
                q_norm_mem_config = ttnn.DRAM_MEMORY_CONFIG
            qr, kv_a = self._q_a_latent_seq(hidden_states, seq_len_local, q_norm_mem_config), None
        elif self._merge_qkv_a:
            qr, kv_a = self._qkv_a_latent(hidden_states, seq_len_local, q_norm_mem_config)
        else:
            qr, kv_a = self._q_a_latent(hidden_states, seq_len_local, q_norm_mem_config), None

        # DSA dispatch (v3.2 / GLM): the op graph is fixed by CONFIG — self._indexer and self._attention
        # are bound once at construction (sparsity × chunking), never by the runtime sequence length.
        # Host-scalar cache slot/start/length arguments are still rebound on each eager dispatch; a captured
        # device trace therefore remains specific to the values used during capture. (At seq_len <=
        # index_topk the indexer top-k simply selects all available causal keys, so sparse is numerically
        # equal to dense there.) The indexer's forward also writes its K-cache (a no-op on the dense
        # null-indexer), so no separate warm-up write is needed.
        # GLM-5.3 reuse: a shared layer receives a prior full layer's top-k indices and skips its own
        # indexer (its ReuseIndexer.forward would raise). Absent injection -> compute as usual.
        selection_state = None
        if indexer_indices is None and self.tt_ccl.sparse_kv_push is not None:
            self.tt_ccl.sparse_kv_push[0].invalidate()  # this layer refills the shared indices buffer
        if indexer_indices is not None:
            indices = indexer_indices
        elif self._sparse_mla_overlap is not None:
            selection_state = self._indexer.score(
                hidden_states,
                qr,
                seq_len_local,
                start_pos=kv_actual_isl or 0,
                rope_tensors=rope_tensors,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                index_kv_cache=index_kv_cache,
                actual_end=actual_end,
                metadata=metadata,
            )
            indices = None
        else:
            indices = self._indexer.forward(
                hidden_states,
                qr,
                seq_len_local,
                start_pos=kv_actual_isl or 0,
                rope_tensors=rope_tensors,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                index_kv_cache=index_kv_cache,
                actual_end=actual_end,
                metadata=metadata,
            )

        if self._seq_q and self._q_defer_l1 and (indexer_indices is not None or _GLUE_Q_DEFER_FULL):
            # Built inside _finish_sparse_chunked_attn, after the KV write and the prefix push/gather.
            tt_q = _DeferredQ(
                lambda q_b_out, qr=qr: self._q_stem_seq(
                    qr, rope_tensors, kv_actual_isl, metadata=metadata, out_l1=True, q_b_out=q_b_out
                ),
                qr=qr,
            )
        elif self._seq_q:
            tt_q = self._q_stem_seq(qr, rope_tensors, kv_actual_isl, metadata=metadata)
        else:
            tt_q = self._q_stem(qr, rope_tensors, kv_actual_isl, seq_len_local, metadata=metadata)
        tt_kvpe, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            kvpe_cache,
            metadata=metadata,
            kv_a=kv_a,
        )

        attention_kwargs = dict(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            tt_kv_nope=tt_kv_nope,
            indices=indices,
            kvpe_cache=kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            kv_actual_isl=kv_actual_isl,
            actual_end=actual_end,
            metadata=metadata,
            injected=indexer_indices is not None,
        )
        if selection_state is not None:
            attn_out, indices = self._sparse_chunked_attn_overlapped(
                selection_state=selection_state,
                **attention_kwargs,
            )
        else:
            attn_out = self._attention(**attention_kwargs)

        out = self._o_proj_epilogue(attn_out, seq_len_local, hidden_states=hidden_states)
        ttnn.tracy_message("`TT_SIGNPOST: MLA_END`")
        # ``indices`` survives _sparse_mla (it deallocs only re-sharded copies), so it is safe to return
        # for a "full" layer to hand to downstream "shared" layers (GLM-5.3 reuse).
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
        actual_end=None,
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
            actual_end=actual_end,
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
        actual_end=None,
        metadata=None,
        injected=False,
        **_,
    ):
        assert indices is not None, "sparse MLA forward requires indexer top-k indices"
        if injected and self._sparse_kv_push:
            return self._sparse_pushed_attn(
                tt_q=tt_q,
                tt_kvpe=tt_kvpe,
                indices=indices,
                kvpe_cache=kvpe_cache,
                kv_actual_isl=kv_actual_isl,
                cache_layer_idx=cache_layer_idx,
                cache_user_id=cache_user_id,
                seq_len_local=seq_len_local,
                actual_end=actual_end,
                metadata=metadata,
            )
        cache_batch_idx, populated_global = self._prepare_sparse_kv_prefix(
            tt_kvpe=tt_kvpe,
            kvpe_cache=kvpe_cache,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            seq_len_local=seq_len_local,
            actual_end=actual_end,
            metadata=metadata,
        )
        kvpe_dev = self._gather_kvpe_prefix(
            kvpe_cache,
            cache_batch_idx,
            populated_global,
            block_cyclic_chunk_local=seq_len_local,
            metadata=metadata,
        )
        return self._finish_sparse_chunked_attn(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            kvpe_dev=kvpe_dev,
            indices=indices,
            kvpe_cache=kvpe_cache,
            seq_len_local=seq_len_local,
        )

    def _sparse_pushed_attn(
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
        actual_end=None,
        metadata=None,
    ):
        """Sparse attention over only the KV rows `indices` reference, pushed by their owners into a compact buffer.

        Trace-safe on the metadata path: the KV write, the plan's chunk start and the pushed cache slot are all read
        on device from the metadata tensors, so a captured replay serves any chunk start and any user slot."""
        cache_batch_idx, _ = self._prepare_sparse_kv_prefix(
            tt_kvpe=tt_kvpe,
            kvpe_cache=kvpe_cache,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            seq_len_local=seq_len_local,
            actual_end=actual_end,
            # Without it a captured replay writes every chunk at the captured kv_actual_isl.
            metadata=metadata,
        )
        planner, ready_sem, done_sem = self.tt_ccl.get_sparse_kv_push(self.ccl_num_links)
        storage = kvpe_cache.storage
        plan, push_buffer = self._sparse_push_plan(
            indices, kvpe_cache, seq_len_local, kv_actual_isl, self.layer_idx, metadata=metadata
        )
        # When the previous layer's sparse_sdpa already pushed this plan's prefix rows (rows of earlier chunks), push
        # only the remainder: this chunk's own rows, just written above.
        prefetched = planner.prefetched.pop(self.layer_idx, None) is plan
        front, spill = push_buffer
        push = lambda subdevice_id=None: ttnn.experimental.high_bw_all_gather(
            storage,
            dim=2,
            output_tensor=front,
            push_spill=spill,
            cluster_axis=None,
            subdevice_id=subdevice_id,
            num_links=self.ccl_num_links,
            **self._sparse_push_slot_kwargs(storage, cache_batch_idx, metadata),
            push_table=plan.table,
            push_stripe_chunk=plan.stripe_chunk,
            push_max_rows=plan.max_rows,
            push_part=2 if prefetched else 0,
            ready_semaphore=ready_sem,
            data_valid_semaphore=done_sem,
            push_hot_table=plan.hot_table,
        )
        if self._qpush is not None and isinstance(tt_q, _DeferredQ) and self._qpush_fits(tt_q):
            compact = self._qpush_region(tt_q, push)
        else:
            compact = push()
        kvpe_dev = MlaKvCache(format=kvpe_cache.format, storage=compact, geometry=kvpe_cache.geometry)
        return self._finish_sparse_chunked_attn(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            kvpe_dev=kvpe_dev,
            indices=plan.compact_indices,
            kvpe_cache=kvpe_cache,
            seq_len_local=seq_len_local,
            overlap=self._next_layer_prefetch(plan, kvpe_cache, cache_batch_idx, metadata=metadata),
            compact=True,
            kv_spill=spill,
        )

    def _sparse_push_slot_kwargs(self, storage, cache_batch_idx, metadata, prefix=""):
        """Cache-slot arguments of a push. On the metadata path the slot is the USER id tensor (metadata[0]),
        recomposed on device as user * layer_num + layer: a host scalar would be frozen at capture time, so every
        replay would push the captured slot while the metadata-driven KV write lands in the live one."""
        if storage.shape[0] <= 1:
            return {f"{prefix}batch_index": 0} if prefix else {"input_batch_index": None}
        if metadata is not None:
            names = (
                ("push_batch_index_tensor", "push_batch_slot_num_layers", "push_batch_slot_layer_idx")
                if prefix
                else ("input_batch_index_tensor", "batch_slot_num_layers", "batch_slot_layer_idx")
            )
            return dict(zip(names, (metadata[0], self.layer_num, cache_batch_idx % self.layer_num)))
        return {f"{prefix}batch_index": cache_batch_idx} if prefix else {"input_batch_index": cache_batch_idx}

    def _sparse_push_buffer(self, planner, kvpe_cache, layer_idx):
        """The compact buffer of `layer_idx`'s parity as (front, spill), and its DRAM-aligned row page size. With an
        L1 front, `front` holds the first compact rows and `spill` (DRAM) the rest; without one, front is the DRAM
        buffer and spill is None."""
        storage = kvpe_cache.storage
        row_bytes = storage.shape[-1] * (1 if storage.dtype == ttnn.fp8_e4m3 else 2)
        page_bytes = -(-row_bytes // 64) * 64
        dense = self._sparse_kv_gather_buffer
        rows = planner.buffer_rows(dense.shape[2], page_bytes)
        front = planner.front_buffer(layer_idx % 2, dense, page_bytes)
        if front is None:
            return (planner.push_buffer(layer_idx % 2, rows, dense), None), page_bytes
        return (front, planner.spill_buffer(layer_idx % 2, rows, dense, page_bytes)), page_bytes

    def _sparse_push_plan(
        self, indices, kvpe_cache, seq_len_local, kv_actual_isl, layer_idx, metadata=None, run_gather=None
    ):
        """The (cached) push plan for `indices` and the compact buffer of `layer_idx`'s parity."""
        planner = self.tt_ccl.get_sparse_kv_push(self.ccl_num_links)[0]
        push_buffer, page_bytes = self._sparse_push_buffer(planner, kvpe_cache, layer_idx)
        plan = planner.plan(
            indices,
            stripe_chunk=seq_len_local // self.tp_factor,
            page_bytes=page_bytes,
            max_rows=self._sparse_kv_gather_buffer.shape[2],
            chunk_start=kv_actual_isl or 0,
            # Metadata path: the chunk start (global, = kv_actual_global) is read on device, not frozen at capture.
            chunk_start_tensor=metadata[1] if metadata is not None else None,
            run_gather=run_gather,
        )
        return plan, push_buffer

    def _next_layer_prefetch(self, plan, kvpe_cache, cache_batch_idx, metadata=None):
        """sparse_sdpa arguments that fuse the next layer's prefix push into this layer's attention, or None."""
        resources = self._push_prefetch
        if resources is None or not self._sparse_kv_push:
            return None
        planner = self.tt_ccl.get_sparse_kv_push(self.ccl_num_links)[0]
        nxt = self.layer_idx + 1
        (front, spill), _ = self._sparse_push_buffer(planner, kvpe_cache, nxt)
        planner.prefetched[nxt] = plan
        storage = kvpe_cache.storage
        # Attention runs on the overlap profile's top-k grid; the push senders and fabric muxes on its gather grid.
        return dict(
            sub_core_grids=resources.topk_core_grid,
            push_input=storage,
            push_table=plan.table,
            push_output=front,
            push_output_spill=spill,
            push_stripe_chunk=plan.stripe_chunk,
            push_max_rows=plan.max_rows,
            # The next layer's slot: cache slots are user-major, so it is this slot + 1.
            **self._sparse_push_slot_kwargs(storage, cache_batch_idx + 1, metadata, prefix="push_"),
            push_num_links=self.ccl_num_links,
            push_ready_semaphore=resources.ready_semaphore,
            push_data_valid_semaphore=resources.data_valid_semaphore,
            push_core_grid=resources.gather_core_grid,
            push_hot_table=plan.hot_table,
        )

    def _prepare_sparse_kv_prefix(
        self,
        *,
        tt_kvpe,
        kvpe_cache,
        cache_user_id,
        cache_layer_idx,
        kv_actual_isl,
        seq_len_local,
        actual_end=None,
        metadata=None,
    ):
        """Write the current KVPE chunk and return the selected slot plus populated global prefix."""
        cache_batch_idx = self._cache_batch_idx(cache_user_id, cache_layer_idx)
        self._update_kv_cache(
            kvpe_cache,
            tt_kvpe,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            actual_end=actual_end,
            tp_axis=self.tp_shard_kv_axis,  # KV dedup: write only this chip's 1/tp window
            # Same metadata the dense twin passes. Without it a captured replay writes EVERY chunk at the
            # captured kv_actual_isl (None -> 0), stacking all chunks on top of chunk 0's KV region.
            metadata=metadata,
        )
        if self._kv_l1():
            # glue: the L1 KV fields are dead once written; free them before the push / gather region.
            for value in tt_kvpe if isinstance(tt_kvpe, tuple) else (tt_kvpe,):
                ttnn.deallocate(value)
        # The write above is clamped to the chunk's real tokens, so on the last chunk KV is populated to
        # ceil32(actual_end), not the padded window. The gather needs only that prefix — top-k indices never
        # address the unwritten suffix, the indexer being bounded by the same prefix.
        #
        # On the METADATA path this host value is only the capture-time placeholder: actual_end lives in
        # metadata[2] and the reader derives the real extent from metadata[1], so the number computed here
        # never reaches the replayed gather.
        chunk_end_global = kv_actual_isl + seq_len_local * self.sp_factor
        populated_global = (
            chunk_end_global
            if actual_end is None
            else min(chunk_end_global, -(-actual_end // ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
        )
        return cache_batch_idx, populated_global

    def _finish_sparse_chunked_attn(
        self,
        *,
        tt_q,
        tt_kvpe,
        kvpe_dev,
        indices,
        kvpe_cache,
        seq_len_local,
        compact=False,
        overlap=None,
        q_seq=None,
        kv_spill=None,
    ):
        """Consume joined/finalized branch outputs on the restored default full-grid manager."""
        for value in tt_kvpe if isinstance(tt_kvpe, tuple) else (tt_kvpe,):
            if value.is_allocated():
                ttnn.deallocate(value)
        if isinstance(tt_q, _DeferredQ):
            tt_q = tt_q()  # glue: Q built now, after the KV write and the prefix push/gather (L1 Q buffer)

        # Sparse attention runs over latent V; project to v_head_dim afterwards. The prefix is already
        # sliced to this slot (batch-1), so no cache_batch_idx.
        attn_out = self._sparse_mla(
            tt_q,
            kvpe_dev,
            indices,
            block_cyclic_chunk_local=seq_len_local,
            cache_batch_idx=None,
            compact=compact,
            overlap=overlap,
            q_seq=q_seq,
            kv_spill=kv_spill,
        )
        # NOT deallocated: the full-mesh gather writes the model-owned persistent scratch, and freeing
        # that would take the buffer away from every later chunk and layer. (high_bw_all_gather returns a
        # fresh Python wrapper for its supplied output, so wrapper identity cannot decide this.) The old
        # sp==1 non-dedup route did produce a genuinely transient slice; with the SP-only gather gone
        # there is no transient buffer here any more.
        if tt_q.is_allocated():  # an L1 Q is already freed right after its untilize (_sparse_mla)
            ttnn.deallocate(tt_q)
        if self._seq_q:
            # All heads of this TP quarter's rows: project, then restore the head-sharded layout. attn_out is freed
            # before the all-to-all (under MLA_SEQQ_L1 it is an L1 buffer and must not outlive wkv_b2).
            # glue (GLUE_WKVB2_OUT_L1): the bfp8 [1, H, rows, v] wkv_b2 output only feeds the seq->heads all-to-all.
            v = self._apply_wkv_b2(attn_out, attn_out.shape[2], out_l1=_GLUE_WKVB2_OUT_L1)
            ttnn.deallocate(attn_out)
            return self._seq_to_heads(v)
        return self._apply_wkv_b2(attn_out, seq_len_local)

    def _select_and_gather_overlapped(
        self,
        *,
        selection_state,
        kvpe_cache,
        cache_batch_idx,
        populated_global,
        seq_len_local,
        metadata=None,
        q=None,
    ):
        """Run the closed two-program overlap region and return joined branch outputs: (indices, gathered KVPE,
        q resharded for sparse_sdpa or None). With `q`, its head-to-sequence reshard runs on the gather branch after
        the KV gather, hidden behind the longer top-k branch.

        Manager clear is the device-side join. Any exception restores the default manager and resets
        caller-owned semaphores only after draining, so an unknown residue is never reused.
        """
        resources = self._sparse_mla_overlap
        assert resources is not None
        loaded = False
        failed = False
        try:
            trace_controller = getattr(self, "_trace_controller", None)
            if trace_controller is not None:
                trace_controller.sub_device_load(resources.manager_id)
            else:
                self.mesh_device.load_sub_device_manager(resources.manager_id)
            loaded = True
            ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_OVERLAP_START`")
            ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_LOCAL_TOPK`")
            local_indices = self._indexer.select_local(
                selection_state,
                subdevice_id=resources.topk_subdevice_id,
                sub_core_grids=resources.topk_core_grid,
            )
            ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_KV_GATHER`")
            kvpe_dev = self._gather_kvpe_prefix(
                kvpe_cache,
                cache_batch_idx,
                populated_global,
                block_cyclic_chunk_local=seq_len_local,
                overlap_resources=resources,
                metadata=metadata,
            )
            q_seq = None
            if q is not None and self._needs_head_to_seq_reshard and not self._seq_q:
                ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_Q_RESHARD`")
                q_seq = self._sparse_q_seq(q, subdevice_id=resources.gather_subdevice_id)
        except Exception:
            failed = True
            raise
        finally:
            if loaded:
                if trace_controller is not None:
                    trace_controller.sub_device_clear()
                else:
                    self.mesh_device.clear_loaded_sub_device_manager()
                ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_OVERLAP_END`")
            if failed:
                self.tt_ccl.reset_sparse_mla_overlap_semaphores()

        ttnn.tracy_message("`TT_SIGNPOST: SPARSE_MLA_INDEX_REDISTRIBUTION`")
        indices = self._indexer.finalize_distribution(local_indices, selection_state)
        return indices, kvpe_dev, q_seq

    def _sparse_chunked_attn_overlapped(
        self,
        *,
        selection_state,
        tt_q,
        tt_kvpe,
        kvpe_cache,
        cache_layer_idx,
        cache_user_id,
        seq_len_local,
        kv_actual_isl,
        actual_end=None,
        metadata=None,
        **_,
    ):
        """Full-indexer sparse attention with local top-k || SP KV-prefix gather."""
        cache_batch_idx, populated_global = self._prepare_sparse_kv_prefix(
            tt_kvpe=tt_kvpe,
            kvpe_cache=kvpe_cache,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            kv_actual_isl=kv_actual_isl,
            seq_len_local=seq_len_local,
            actual_end=actual_end,
            metadata=metadata,
        )
        indices, kvpe_dev, q_seq = self._select_and_gather_overlapped(
            selection_state=selection_state,
            kvpe_cache=kvpe_cache,
            cache_batch_idx=cache_batch_idx,
            populated_global=populated_global,
            seq_len_local=seq_len_local,
            metadata=metadata,
            q=tt_q,
        )
        # The first shared layer after this one reuses these indices: plan its push now and prefetch its prefix
        # rows during this layer's sparse_sdpa.
        overlap = None
        if self._push_prefetch is not None and self._sparse_kv_push:
            # GLM_QPUSH_FULL: the deferred q_b runs next to the plan tables' all-gather (see _qpush_region).
            run_gather = None
            if self._qpush is not None and isinstance(tt_q, _DeferredQ) and self._qpush_fits(tt_q):
                run_gather = lambda gather: self._qpush_region(tt_q, gather)
            plan, _ = self._sparse_push_plan(
                indices,
                kvpe_cache,
                seq_len_local,
                kv_actual_isl,
                self.layer_idx + 1,
                metadata=metadata,
                run_gather=run_gather,
            )
            overlap = self._next_layer_prefetch(plan, kvpe_cache, cache_batch_idx, metadata=metadata)
        output = self._finish_sparse_chunked_attn(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            kvpe_dev=kvpe_dev,
            indices=indices,
            kvpe_cache=kvpe_cache,
            seq_len_local=seq_len_local,
            overlap=overlap,
            q_seq=q_seq,
        )
        return output, indices

    def _forward_kv_only(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: MlaKvCache,
        cache_layer_idx: int,
        kv_actual_isl: int,
        cache_user_id: int,
        index_kv_cache: Optional[ttnn.Tensor],
        actual_end: Optional[int] = None,
        metadata: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
    ) -> None:
        """Last-layer fast path: fill the migratable KVPE cache, then stop before query/attention/output.

        A sparse full-indexer layer also writes its index-key chunk to the caller-owned ``index_kv_cache``;
        a shared/reuse-indexer layer deliberately skips that write because it owns no indexer state. The
        enclosing block skips FFN/MoE/norm/LM head as well, so this path produces no first-token output.
        """
        ttnn.tracy_message("`TT_SIGNPOST: MLA_START`")
        seq_len_local = hidden_states.shape[2]

        # Sparse decode needs the index key cache for every full-indexer layer even though this fast path
        # skips query construction and scoring. Shared-indexer layers reuse a prior layer's selection and
        if self._has_indexer and not self._indexer_reuse and indexer_indices is None:
            assert index_kv_cache is not None, "sparse kv_only requires the caller-owned index key cache"
            self._indexer.write_k(
                hidden_states,
                seq_len_local,
                kv_actual_isl,
                rope_tensors=rope_tensors,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                index_kbuf=index_kv_cache,
                actual_end=actual_end,
                # Under capture kv_actual_isl is None; without metadata this layer would rope and write its
                # indexer K at position 0 on every chunk -- silent corruption, not a crash.
                metadata=metadata,
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
            actual_end=actual_end,
            metadata=metadata,
            tp_axis=self.tp_shard_kv_axis,  # KV dedup: write only this chip's 1/tp window
        )
        for value in tt_kvpe if isinstance(tt_kvpe, tuple) else (tt_kvpe,):
            ttnn.deallocate(value)

        ttnn.tracy_message("`TT_SIGNPOST: MLA_END`")
        return None

    # ----------------------------------------------------------------------------------------
    # DSA indexer + sparse attention (v3.2 / GLM). Inert unless _has_indexer (dense v3.1 path
    # never reaches these). The full forward above shares the dense/sparse Q/KV stem and epilogue;
    # only sparse-specific gather/attention helpers live below.
    # ----------------------------------------------------------------------------------------

    def _declare_q_sequence_topology(self, tt_q: ttnn.Tensor, seq_len_local: int) -> None:
        """Restate Q (in place) as sequence-on-SP / heads-on-TP; the inherited Shard(0) is stale after the reshape."""
        heads_local = tt_q.shape[1]
        assert tt_q.shape[2] == seq_len_local, (
            f"Q slab {tt_q.shape[2]} is not this chip's chunk rows {seq_len_local}; refusing to declare "
            "a sequence distribution that does not hold"
        )
        assert heads_local * self.tp_factor == self.num_heads, (
            f"Q holds {heads_local} of {self.num_heads} heads at tp={self.tp_factor}; refusing to declare "
            "a head distribution that does not hold"
        )
        current = tt_q.tensor_topology()
        placements = [None, None]
        placements[self.sp_axis] = ttnn.PlacementShard(2)  # sequence
        placements[self.tp_axis] = ttnn.PlacementShard(1)  # heads
        tt_q.update_tensor_topology(
            ttnn.TensorTopology(current.distribution_shape(), placements, current.mesh_coords())
        )

    def set_tp_shard_kv(self, tp_shard_kv: Optional[bool]) -> None:
        """Sparse always dedups (its gather has no other mode); dense opts in via ring_mla's split-KV path.
        Weights and buffers do not depend on it, so a built model can switch layout between runs."""
        if tp_shard_kv is None:
            self.tp_shard_kv = self._has_indexer and self.tp_factor > 1
        else:
            assert tp_shard_kv or not self._has_indexer, "sparse (DSA) MLA requires a TP-deduped KV cache"
            self.tp_shard_kv = tp_shard_kv
            assert not tp_shard_kv or self.tp_factor > 1, (
                f"tp_shard_kv needs a TP axis to dedup across (tp_factor={self.tp_factor}); at tp=1 the "
                "cache has no replicas to remove"
            )
            assert (
                not tp_shard_kv or self._has_indexer or self.is_chunked
            ), "dense tp_shard_kv is the chunked ring_mla split-KV path; single-shot prefill has no striped write"
        # Checked here, not on the first chunk after a long weight load.
        if self.tp_shard_kv and not self._has_indexer:
            assert self.sp_axis == 0 and self.tp_axis == 1, (
                f"dense tp_shard_kv needs SP on mesh axis 0 and TP on axis 1 (split-KV ring_mla), got "
                f"sp_axis={self.sp_axis} tp_axis={self.tp_axis}"
            )
            chunk_local = self.active_seq_len // self.sp_factor
            assert self.active_seq_len % self.sp_factor == 0 and chunk_local % (self.tp_factor * ttnn.TILE_SIZE) == 0, (
                f"dense tp_shard_kv needs a tile-aligned per-chip KV region: chunk {self.active_seq_len} / "
                f"sp {self.sp_factor} / tp {self.tp_factor} is not a multiple of {ttnn.TILE_SIZE}"
            )
            assert self.max_seq_len % self.active_seq_len == 0, (
                f"dense tp_shard_kv needs the cache to hold a whole number of chunks: seq_len {self.max_seq_len} "
                f"is not a multiple of chunk {self.active_seq_len}"
            )
            _fabric = ttnn.get_fabric_config()
            assert _fabric in (
                ttnn.FabricConfig.FABRIC_2D,
                ttnn.FabricConfig.FABRIC_2D_TORUS_X,
                ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            ), (
                f"dense tp_shard_kv requires a 2D fabric config, got {_fabric}: ring_mla's full-mesh KV "
                "gather spans both mesh axes. Open the mesh with FABRIC_2D (or a 2D torus variant)."
            )

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

    # ----------------------------------------------------------------------------------------
    # Sequence-sharded absorbed Q (MLA_SEQ_SHARDED_Q=1 / config.mla_seq_sharded_q / seq_sharded_q=).
    #
    # The head-sharded path builds Q for this chip's S/sp rows x H/tp heads, then all-to-alls heads -> sequence
    # over TP for sparse_sdpa (which needs H % 32 == 0) and back again before wkv_b2. Here each chip builds Q
    # for its own S/(sp*tp) TP-quarter rows x ALL heads straight away (q_b / wkv_b1 / wkv_b2 replicated over
    # TP), exactly like the indexer's own Q: the q_a latent is reduce-scattered over rows instead of
    # all-reduced, so no chip ever holds the other quarters' rows. sparse_sdpa then runs on Q as built, wkv_b2
    # runs on all heads of those rows, and one all-to-all of the v_head_dim-wide (bfp8) result restores the
    # head-sharded [S/sp, H/tp * v] layout o_proj consumes. The heads -> sequence all-to-all is gone and the
    # sequence -> heads one moves the 256-wide bfp8 output instead of the 512-wide bf16 latent. Same FLOPs;
    # costs the replicated weights (+3/4 of q_b, wkv_b1, wkv_b2 per layer).
    # ----------------------------------------------------------------------------------------

    def _resolve_seq_sharded_q(self, requested: Optional[bool]) -> bool:
        if requested is None:
            env = os.getenv("MLA_SEQ_SHARDED_Q")
            requested = (
                env not in (None, "", "0") if env is not None else getattr(self.config, "mla_seq_sharded_q", False)
            )
        if not requested:
            return False
        reasons = []
        if not self._has_indexer:
            reasons.append("dense MLA")
        if self.kv_only:
            reasons.append("kv-only layer")
        if not self._needs_head_to_seq_reshard:
            reasons.append("head shard already fat enough for sparse_sdpa")
        if self._merge_qkv_a or self._use_gate or self._llama4_beta is not None or not self._use_indexed_rope:
            reasons.append("merged qkv_a / output gate / llama4 scale / non-indexed rope")
        if (self.sp_axis, self.tp_axis) != (0, 1) or self.active_seq_len_local % (self.tp_factor * ttnn.TILE_SIZE):
            reasons.append("needs sp_axis=0, tp_axis=1 and whole-tile TP row quarters")
        if reasons:
            logger.debug(f"MLA layer {self.layer_idx}: sequence-sharded Q off ({'; '.join(reasons)})")
            return False
        return True

    def _dram_width_shard(self, w: ttnn.Tensor) -> ttnn.Tensor:
        """Re-lay a DRAM-interleaved [.., K, N] weight out WIDTH_SHARDED over every DRAM bank ([K, N/banks] per bank) by
        an on-device tile copy (bit-identical tiles). Returns w unchanged when N does not split into whole-tile shards."""
        g = self.mesh_device.dram_grid_size()
        banks = g.x * g.y
        n = w.shape[-1]
        height = 1
        for d in list(w.shape)[:-1]:
            height *= d
        if n % (banks * ttnn.TILE_SIZE) != 0:
            return w
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [height, n // banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out = ttnn.to_memory_config(w, mc)
        ttnn.deallocate(w)
        return out

    def _oproj_grid12(self, kwargs: dict) -> dict:
        """mm track (MLA_OPROJ_GRID12=1): the GLM o_proj 2D mcast on the full 12x10 grid instead of 11x10 (N = 192
        tiles = 12 x 16, sub 1x8; R0 -8.9%, sustained 32-chip 12x10 -7%, bit-exact on all chips). Only the tuned 11x10
        2D config with per_core_N 18 is replaced; anything else passes through."""
        pc = kwargs.get("program_config")
        if not isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
            return kwargs
        g = pc.compute_with_storage_grid_size
        n_t = self.o_proj_weight.shape[-1] // ttnn.TILE_SIZE
        dev = self.mesh_device.compute_with_storage_grid_size()
        if (g.x, g.y) != (11, 10) or pc.per_core_N != 18 or dev.x < 12 or n_t % 12 != 0 or (n_t // 12) % 8 != 0:
            return kwargs
        pc12 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 10),
            in0_block_w=pc.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=pc.per_core_M,
            per_core_N=n_t // 12,
            transpose_mcast=pc.transpose_mcast,
            fuse_batch=pc.fuse_batch,
            fused_activation=pc.fused_activation,
        )
        return dict(kwargs, program_config=pc12)

    def _replicate_over_tp(self, w: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        """TP-replicated copy of a TP-sharded weight, by an on-device all_gather along the TP axis (a tile copy, so
        bit-identical to uploading the full weight). Frees the sharded input."""
        shape = list(w.shape)
        lead = 4 - len(shape)
        w4 = ttnn.reshape(w, [1] * lead + shape) if lead else w
        gathered = ttnn.all_gather(
            w4,
            dim=dim + lead if dim >= 0 else dim,
            cluster_axis=self.tp_axis,
            num_links=self.ccl_num_links,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w)
        if lead:
            shape[dim] *= self.tp_factor
            gathered = ttnn.reshape(gathered, shape)
        return gathered

    def _log_dram(self, what: str) -> None:
        mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        print(
            f"MLA_MEMLOG {what}: DRAM banks={mv.num_banks} total/bank={mv.total_bytes_per_bank / 2**20:.1f} MiB "
            f"allocated/bank={mv.total_bytes_allocated_per_bank / 2**20:.1f} MiB free/bank="
            f"{mv.total_bytes_free_per_bank / 2**20:.1f} MiB largest_free/bank="
            f"{mv.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB",
            flush=True,
        )

    def _q_a_latent_seq(
        self, hidden_states: ttnn.Tensor, seq_len_local: int, norm_memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """q_a projection + TP reduce-scatter over ROWS + q_a_layernorm: this chip's [1, 1, S/(sp*tp), q_lora] rows
        of the q_a latent (rows [r*S/(sp*tp), (r+1)*S/(sp*tp)) of its SP slab at TP rank r, the same rows
        mesh_partition(dim=2) over TP selects). Replaces all-reduce + norm of all S/sp rows: both consumers (the
        indexer and _q_stem_seq) read only these rows."""
        qr = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            **self._get_mm_kwargs("q_a_proj", seq_len_local),
        )
        qr_rows = ttnn.experimental.reduce_scatter_minimal_async(
            qr,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
            num_workers_per_link=_GLUE_SEQQ_RS_WORKERS,
        )
        ttnn.deallocate(qr)
        if _GLUE_SEQQ_NORM_CORES:
            out = self._q_a_norm_width_sharded(qr_rows, norm_memory_config, _GLUE_SEQQ_NORM_CORES)
        else:
            out = self._q_a_norm(qr_rows, norm_memory_config)
        ttnn.deallocate(qr_rows)
        return out

    def _q_a_norm_width_sharded(self, qr: ttnn.Tensor, out_memory_config: ttnn.MemoryConfig, cores: int) -> ttnn.Tensor:
        """q_a RMSNorm with the rows width-sharded over `cores` cores (8 per grid row). The interleaved rms_norm runs
        one core per 32-row tile row, i.e. 5 cores for seqq's 160 rows; the sharded one splits the 2048 channels and
        reduces the row statistics across the shard grid (glue track, GLUE_SEQQ_NORM_CORES)."""
        rows, width = qr.shape[-2], qr.shape[-1]
        key = (rows, width, cores)
        cfgs = getattr(self, "_q_norm_shard_cfgs", None)
        if cfgs is None:
            cfgs = self._q_norm_shard_cfgs = {}
        if key not in cfgs:
            gx = min(8, cores)
            gy = cores // gx
            block_w = width // ttnn.TILE_SIZE // cores
            sub_w = max(w for w in (4, 3, 2, 1) if block_w % w == 0)
            cfgs[key] = (
                ttnn.create_sharded_memory_config(
                    shape=(rows, width // cores),
                    core_grid=ttnn.CoreGrid(x=gx, y=gy),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
                ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[gx, gy],
                    subblock_w=sub_w,
                    block_h=rows // ttnn.TILE_SIZE,
                    block_w=block_w,
                    inplace=False,
                ),
            )
        in_cfg, pc = cfgs[key]
        xs = ttnn.to_memory_config(qr, in_cfg)
        o = ttnn.rms_norm(
            xs,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            program_config=pc,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        ttnn.deallocate(xs)
        out = ttnn.sharded_to_interleaved(o, out_memory_config)
        ttnn.deallocate(o)
        return out

    def _qpush_q_b_program_config(self, resources):
        """q_b's 1D mcast_in0 program config moved onto the overlap profile's top-k sub-device (Q-stem || push
        overlap): the same blocking (in0_block_w, per_core_M), the grid anchored at the sub-device's first row through
        allowed_worker_cores, at most as wide as today's (11 columns) and per_core_N widened to fit the fewer cores
        (5 -> 6 tiles: 86 of 88 cores). Each output tile accumulates K in the same order, so the result is bit-identical
        to the full-grid q_b. None when q_b has no 1D mcast_in0 config at this chunk's rows."""
        rows = self.active_seq_len_local // self.tp_factor
        pc = self._seq_qb_ws_pc if (self._seq_qb_ws_pc is not None and rows == self._seq_qb_ws_rows) else None
        if pc is None:
            cfg = self._resolve_mm_cfg("q_b_proj", rows)
            pc = cfg["program_config"] if cfg is not None else None
        if not (isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig) and pc.mcast_in0):
            return None
        box = resources.topk_core_grid.bounding_box()
        gx = min(pc.compute_with_storage_grid_size.x, box.end.x - box.start.x + 1)
        gy = box.end.y - box.start.y + 1
        n_tiles = self.q_b_proj_weight.shape[-1] // ttnn.TILE_SIZE
        per_core_n = -(-n_tiles // (gx * gy))
        sub_h = pc.out_subblock_h
        sub_w = max(w for w in range(1, 9) if per_core_n % w == 0 and w * sub_h <= 8)
        self._qpush_rows = rows
        logger.info(
            f"[qpush overlap] layer {self.layer_idx}: q_b on {gx}x{gy} at ({box.start.x},{box.start.y}), "
            f"per_core_N {pc.per_core_N} -> {per_core_n} ({-(-n_tiles // per_core_n)} cores), push on "
            f"{resources.gather_core_grid.num_cores()} gather cores"
        )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=pc.in0_block_w,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=pc.per_core_M,
            per_core_N=per_core_n,
            fuse_batch=pc.fuse_batch,
            fused_activation=pc.fused_activation,
            mcast_in0=True,
            allowed_worker_cores=ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(box.start.x, box.start.y),
                        ttnn.CoreCoord(box.start.x + gx - 1, box.start.y + gy - 1),
                    )
                ]
            ),
        )

    def _qpush_fits(self, tt_q: "_DeferredQ") -> bool:
        return tt_q.qr is not None and tt_q.q_b_out is None and tt_q.qr.shape[2] == self._qpush_rows

    def _qpush_region(self, tt_q: "_DeferredQ", push):
        """Q-stem || push overlap: issue `push` on the gather sub-device, then the deferred Q stem's q_b on the top-k
        sub-device; the manager clear joins them. Returns push's result; the q_b output goes to tt_q.q_b_out.

        The q_b output is allocated BEFORE the region on purpose: the push program's static CBs (rows 0-1) are then
        checked against it when the push is enqueued, whereas an L1 buffer allocated inside the region is checked only
        against q_b's own cores and could land on L1 the running push still uses. Nothing is freed inside the region."""
        resources, pc = self._qpush
        qr = tt_q.qr
        kw = self._get_mm_kwargs("q_b_proj", qr.shape[2])
        q_b_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, qr.shape[2], self.q_b_proj_weight.shape[-1]]),
            kw["dtype"],
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG if _QPUSH_QB_DRAM else kw["memory_config"],
        )
        q_b_out.update_tensor_topology(qr.tensor_topology())
        trace_controller = getattr(self, "_trace_controller", None)
        if trace_controller is not None:
            trace_controller.sub_device_load(resources.manager_id)
        else:
            self.mesh_device.load_sub_device_manager(resources.manager_id)
        try:
            ttnn.tracy_message("`TT_SIGNPOST: QPUSH_OVERLAP_START`")
            compact = push(resources.gather_subdevice_id)
            ttnn.linear(
                qr,
                self.q_b_proj_weight,
                program_config=pc,
                dtype=kw["dtype"],
                compute_kernel_config=kw["compute_kernel_config"],
                optional_output_tensor=q_b_out,
            )
        finally:
            if trace_controller is not None:
                trace_controller.sub_device_clear()
            else:
                self.mesh_device.clear_loaded_sub_device_manager()
            ttnn.tracy_message("`TT_SIGNPOST: QPUSH_OVERLAP_END`")
        ttnn.deallocate(qr)
        tt_q.qr = None
        tt_q.q_b_out = q_b_out
        return compact

    def _q_stem_seq(
        self,
        qr: ttnn.Tensor,
        rope_tensors: dict,
        kv_actual_isl: Optional[int],
        metadata: Optional[ttnn.Tensor] = None,
        out_l1: bool = False,
        q_b_out: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Absorbed-Q stem over this chip's TP-quarter rows and ALL heads: [1, H, S/(sp*tp), kv_lora + rope].
        With `q_b_out` the q_b matmul has already run (Q-stem || push overlap) and `qr` is freed."""
        if q_b_out is not None:
            tt_q, q_rows = q_b_out, q_b_out.shape[2]
        else:
            q_rows = qr.shape[2]
            qb_kwargs = self._get_mm_kwargs("q_b_proj", q_rows)
            if self._seq_qb_ws_pc is not None and q_rows == self._seq_qb_ws_rows:
                qb_kwargs = dict(qb_kwargs, program_config=self._seq_qb_ws_pc)
            tt_q = ttnn.linear(qr, self.q_b_proj_weight, **qb_kwargs)
            ttnn.deallocate(qr)
        inplace = self._q_rope_inplace_seq and q_rows == self.active_seq_len_local // self.tp_factor
        tt_q_nope, tt_q_rope = ttnn.experimental.nlp_create_q_heads_split(
            tt_q,
            num_heads=self.num_heads,
            split_head_dim=self.qk_nope_head_dim,
            memory_config=ttnn.L1_MEMORY_CONFIG if (inplace and _GLUE_Q_HEADS_L1) else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_q)
        if inplace:
            # glue track (as _q_stem): [q_nope @ wkv_b1 | 0] straight into the [1, H, rows, 576] Q buffer, then the
            # indexed RoPE writes the rotated q_pe into its last channels in place (no 512-channel concat copy).
            tt_q = ttnn.linear(
                tt_q_nope,
                self.wkv_b1_weight_padded,
                program_config=self._wkv_b1_padded_pc_seq,
                memory_config=ttnn.L1_MEMORY_CONFIG if (out_l1 or _GLUE_Q_OUT_L1) else ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.default_compute_kernel_config,
            )
            ttnn.deallocate(tt_q_nope)
            ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                tt_q_rope,
                rope_tensors["cos_matrix"],
                rope_tensors["sin_matrix"],
                rope_tensors["trans_matrix"],
                kv_actual_global=metadata[1] if metadata is not None else kv_actual_isl,
                cluster_axis=self.sp_axis,
                seq_subshard_axis=self.tp_axis,
                output_tensor=tt_q,
            )
            ttnn.deallocate(tt_q_rope)
            return tt_q
        tt_q_nope_abs = ttnn.linear(tt_q_nope, self.wkv_b1_weight, **self._get_mm_kwargs("wkv_b1", q_rows))
        ttnn.deallocate(tt_q_nope)
        # The rows are this TP rank's quarter of the SP slab: seq_subshard_axis adds that window to the rotated
        # block-cyclic offset (the indexer ropes its TP-partitioned queries the same way).
        tt_q = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=metadata[1] if metadata is not None else kv_actual_isl,
            cluster_axis=self.sp_axis,
            seq_subshard_axis=self.tp_axis,
            concat_prefix=tt_q_nope_abs,
        )
        ttnn.deallocate(tt_q_nope_abs)
        ttnn.deallocate(tt_q_rope)
        return tt_q

    def _seq_to_heads(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """[1, H, S/(sp*tp), v] (this TP quarter's rows, all heads) -> the head-sharded layout the epilogue takes.
        MLA_SEQQ_OUT=heads (default): one all-to-all to [1, H/tp, S/sp, v]; =cols: concat heads first and
        all-to-all [1, 1, S/(sp*tp), H*v] -> [1, 1, S/sp, H/tp*v] (the epilogue then skips its concat)."""
        if self._seq_q_out == "cols":
            cat = ttnn.experimental.nlp_concat_heads(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t)
            t, in_dim, out_dim = cat, 2, 3
        else:
            in_dim, out_dim = 2, 1
        out = ttnn.experimental.all_to_all_async_generic(
            t,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=self.tp_axis,
        )
        ttnn.deallocate(t)
        return out

    def indexer_indices_shape(self) -> tuple:
        rows = self.active_seq_len // self.sp_factor
        if self._needs_head_to_seq_reshard:
            rows //= self.tp_factor
        return (1, 1, rows, min(self.config.index_topk, self.max_seq_len))

    def _sparse_mla(
        self,
        q: ttnn.Tensor,
        kvpe: MlaKvCache,
        indices: ttnn.Tensor,
        block_cyclic_chunk_local: int,
        cache_batch_idx: Optional[int] = None,
        compact: bool = False,
        overlap=None,
        q_seq: Optional[ttnn.Tensor] = None,
        kv_spill: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Absorbed MQA over the top-k selected latents (FlashMLA sparse contract: no causal mask —
        ``indices`` already encode it via the 0xFFFFFFFF sentinel). Invoked SPMD on the SP×TP mesh:
        each chip runs the single-chip ``ttnn.transformer.sparse_sdpa`` over its own q shard, so q's
        distribution is preserved — q SP-sharded on seq (dim2), TP-sharded on heads (dim1), out the same.

        q is absorbed ``[1, H/tp, S/sp, geometry.logical_width]`` TILE bf16. ``kvpe`` carries one
        replicated ROW_MAJOR physical cache row per token; its width depends on the explicit cache format.
        Indices are uint32 query-row shards: ``[1,1,S/(sp*tp),k]`` for head-to-sequence
        redistribution, otherwise ``[1,1,S/sp,k]``. Replicated inputs are partitioned when needed.

        block_cyclic_chunk_local: when set, ``kvpe`` is the KVPE cache in its native BLOCK-CYCLIC SP layout
        (not natural order) and ``indices`` are natural positions; sparse_sdpa remaps each index to its
        physical page in-kernel (invP) over the SP mesh axis, so the host reorder is eliminated. It is the
        per-shard chunk length (chunk_size_global / sp).

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
        # The indexer emits matching TP sequence shards directly. The partition fallback below also
        # accepts SP-only or globally replicated indices from callers using the older output contract.
        transpose_head_to_seq = self._needs_head_to_seq_reshard
        # A full layer reshards q inside its overlap region (see _select_and_gather_overlapped). Sequence-sharded Q
        # (_q_stem_seq) is built TP-sequence-sharded with all heads: _sparse_q_seq passes it through, and the
        # caller (_finish_sparse_chunked_attn) restores heads after wkv_b2 instead of here.
        q_seq_sharded = self._sparse_q_seq(q) if q_seq is None else q_seq
        # the op is ROW_MAJOR-only; q comes in TILE. sparse_sdpa requires its q in DRAM: a deferred L1 Q (glue,
        # GLUE_Q_DEFER_L1) is untilized straight to DRAM and freed at once, so it never sits under sparse_sdpa's CBs.
        q_in_l1 = q_seq_sharded.memory_config().buffer_type == ttnn.BufferType.L1
        q_rm = ttnn.to_layout(
            q_seq_sharded, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG if q_in_l1 else None
        )
        if q_seq_sharded is not q or q_in_l1:
            ttnn.deallocate(q_seq_sharded)

        # Indices must match q_rm's sequence sharding. Direct TP sequence shards [1,1,S/(sp*tp),k]
        # already match after head-to-sequence redistribution. Partition legacy globally replicated
        # [1,1,S_global,k] or SP-only [1,1,S/sp,k] inputs over SP and TP as needed.
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
        assert idx.shape[2] == q_rm.shape[2], (
            f"indexer indices carry {idx.shape[2]} query rows but q has {q_rm.shape[2]}; "
            "the indexer's output_tp_sequence_sharded must match this layer's reshard decision"
        )
        # k_chunk_size must be a multiple of 32 that divides TOPK (prod TOPK=2048 → 128).
        k_chunk = next((c for c in (128, 64, 32) if idx.shape[-1] % c == 0), 32)
        sdpa = lambda **grid: ttnn.transformer.sparse_sdpa(
            q_rm,
            kvpe.storage,
            idx,
            v_dim=self.kv_lora_rank,
            kv_format=kvpe.format.sparse_sdpa_format,
            scale=self.scale,
            k_chunk_size=k_chunk,
            # A compact (pushed) buffer is addressed by rewritten slot indices directly: no block-cyclic remap.
            **(
                {}
                if compact
                else dict(
                    block_cyclic_sp_axis=self.sp_axis,
                    block_cyclic_chunk_local=block_cyclic_chunk_local,
                    block_cyclic_cache_tp_sharded=True,
                )
            ),
            cache_batch_idx=cache_batch_idx,
            kv_spill=kv_spill,
            **grid,
        )
        # `overlap`: the next layer's prefix push, fused into this sparse_sdpa program (see _next_layer_prefetch).
        out = sdpa(**(overlap or self._sparse_sdpa_grid(q_rm.shape[2])))
        ttnn.deallocate(q_rm)
        if idx is not indices:
            ttnn.deallocate(idx)
        # back to TILE for the downstream wkv_b2 linear
        ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG if self._seq_q_l1 else None)
        ttnn.deallocate(out)

        if transpose_head_to_seq and not self._seq_q:
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

    def _sparse_q_seq(self, q, subdevice_id=None):
        """q sequence-sharded over TP for sparse_sdpa when the per-chip head shard is too thin (else q itself).
        Optionally on one sub-device."""
        if not self._needs_head_to_seq_reshard or self._seq_q:
            return q
        return ttnn.experimental.all_to_all_async_generic(
            q,
            in_dim=1,
            out_dim=2,
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=self.tp_axis,
            subdevice_id=subdevice_id,
        )  # [1,H,S/(sp·tp),576] — FABRIC_2D path selected at runtime; topology resolves to Linear

    def _sparse_sdpa_grid(self, rows: int) -> dict:
        """sparse_sdpa's core grid for `rows` query rows per chip: the top-k grid when it needs no more rows per core
        than the full grid, else the full grid."""
        if self._sparse_sdpa_cores is None:
            return {}
        grid = self.mesh_device.compute_with_storage_grid_size()
        per_core = lambda cores: -(-rows // cores)
        if per_core(self._sparse_sdpa_cores.num_cores()) > per_core(grid.x * grid.y):
            return {}
        return {"sub_core_grids": self._sparse_sdpa_cores}

    def _gather_kvpe_prefix(
        self,
        kvpe_cache: MlaKvCache,
        cache_batch_idx,
        populated_global: int,
        *,
        block_cyclic_chunk_local: int,
        metadata=None,
        overlap_resources=None,
    ) -> MlaKvCache:
        """On-device read-back of the chunked KVPE prefix for sparse attention, as ONE full-mesh snake
        gather. The cache is ND-sharded / block-cyclic across SP*TP in the op's format (BF16 or packed
        scaled FP8, ROW_MAJOR); sparse_sdpa consumes it replicated and remaps natural-position indices to
        physical block-cyclic pages in-kernel (invP), so the buffer is LEFT in block-cyclic order -- no
        host reorder.

        Each device holds seq_len/(sp*tp) rows in sp*tp row-major order, which is what the snake ring
        reassembles directly -- no TP-inner stage, no intermediate allocation, and the result lands in the
        persistent worst-case scratch rather than being transient. Same linear chip-major buffer that
        sparse_sdpa decodes with block_cyclic_cache_tp_sharded=True.

        SLOT SELECT IN THE GATHER: the persistent cache's per-chip shape is [B, 1, seq_len_local,
        row_width], B = num_users*num_layers (user-major slots). The all-gather offsets its input pages to
        the active (user, layer) cache slot and packs only that slot's valid prefix into the model-owned
        [1, 1, max_seq_len, row_width] output scratch. It never transports the other B-1 slots, avoiding
        the ~5 GB full-cache gather at 78 layers. The gathered KV is batch-1, so sparse_sdpa needs no
        cache_batch_idx.

        The prefix is rounded to a whole block-cyclic slab; sparse SDPA only dereferences current top-k
        indices, so its unwritten suffix is never consumed."""
        storage = kvpe_cache.storage
        slot_lo = cache_batch_idx if storage.shape[0] > 1 else 0
        multi_slot = storage.shape[0] > 1
        stripes = self.sp_factor * self.tp_factor
        # Block-cyclic storage is meaningful only in whole global slabs; the slab is sp-wide because
        # block_cyclic_chunk_local is the per-SP-ROW width (each chip holds 1/tp of it).
        slab_global = block_cyclic_chunk_local * self.sp_factor
        # Trace-safe extent: populated_global GROWS per chunk, so a capture would bake this chunk's value
        # into the gather's runtime args and every replay would re-gather only the captured prefix, leaving
        # later chunks attending an unpopulated tail. Hand over the chunk start instead and let the reader
        # derive the extent (and its own page partition) on-device from the same closed form the host uses.
        extent_kwargs = (
            {"gathered_prefix_tensor": metadata[1], "gathered_slab_global": slab_global}
            if metadata is not None
            else {
                "gathered_dim_size": min(
                    storage.shape[2] * stripes,
                    ((populated_global + slab_global - 1) // slab_global) * slab_global,
                )
            }
        )
        assert metadata is not None or extent_kwargs["gathered_dim_size"] > 0
        # Trace-safe slot select: hand over the 1-element slot_id (USER id) tensor instead of the host
        # scalar, and let the reader recompose user_id * layer_num + layer_idx on-device. A host
        # input_batch_index is re-patched per dispatch and a replay never re-runs that patch, so every
        # replay would gather the slot that was live at capture time -- while the KV WRITE above, being
        # metadata-driven, lands in the correct slot. Wrong-slot attention, with no error.
        # Single-slot caches keep the scalar form: there is no slot to get wrong.
        slot_meta_kwargs = (
            {
                "input_batch_index_tensor": metadata[0],
                "batch_slot_num_layers": self.layer_num,
                "batch_slot_layer_idx": cache_batch_idx % self.layer_num,
            }
            if metadata is not None and multi_slot
            else {"input_batch_index": slot_lo}
        )
        assert self._sparse_kv_gather_buffer is not None
        # Preconditions of the full-mesh gather -- wrong here means a silently misordered gather.
        # Row-major over the mesh equals the sp*tp linearization only for this axis assignment.
        assert self.sp_axis == 0 and self.tp_axis == 1, "full-mesh KVPE gather assumes sp_axis=0, tp_axis=1"
        # A cache still declaring the legacy kv-head-on-TP layout reports too few stripes and would
        # under-gather. This is the assertion that catches an allocator which forgot tp_axis.
        assert self._declared_seq_shard_factor(storage) == stripes, (
            f"TP-deduped KVPE cache declares {self._declared_seq_shard_factor(storage)} dim-2 stripes, "
            f"expected sp*tp = {stripes} -- allocate the cache with tp_axis set"
        )
        # One snake across both mesh axes, so only Fabric2D can route it. FABRIC_1D reaches the op and
        # fails its generic neighbor proof instead, which names the symptom rather than this cause.
        fabric = ttnn.get_fabric_config()
        assert fabric in (
            ttnn.FabricConfig.FABRIC_2D,
            ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ), f"full-mesh KVPE gather requires a 2D fabric config, got {fabric}"
        gather_kwargs = dict(
            dim=2,
            output_tensor=self._sparse_kv_gather_buffer,
            num_links=self.ccl_num_links,
            cluster_axis=None,
            **slot_meta_kwargs,
            **extent_kwargs,
        )
        # Externally scheduled resources (#54740): when the caller is overlapping index selection with
        # this gather, the gather runs on its own sub-device core grid and signals through the two
        # semaphores instead of owning the whole compute rectangle. None = the standalone path.
        if overlap_resources is not None:
            gather_kwargs.update(
                subdevice_id=overlap_resources.gather_subdevice_id,
                sub_core_grids=overlap_resources.gather_core_grid,
                ready_semaphore=overlap_resources.ready_semaphore,
                data_valid_semaphore=overlap_resources.data_valid_semaphore,
            )
        gathered = ttnn.experimental.high_bw_all_gather(storage, **gather_kwargs)
        return MlaKvCache(
            format=kvpe_cache.format,
            storage=gathered,
            geometry=kvpe_cache.geometry,
        )

    @staticmethod
    def _declared_seq_shard_factor(t) -> int:
        """Product of mesh extents over the axes whose placement shards tensor dim 2 (mirrors the op's
        tensor_dim_shard_factor)."""
        return declared_seq_shard_factor(t)
