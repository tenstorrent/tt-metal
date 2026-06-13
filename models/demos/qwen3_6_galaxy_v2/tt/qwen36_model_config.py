# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B Model Configuration for TT Blackhole Galaxy.

Hyperparams diverge from Qwen3-32B (n_q_heads 24 vs 64, intermediate_dim
13824 vs 25600, head_dim 256, partial RoPE rope_dim=64) so this class
subclasses `TtModelArgs` directly rather than `TtQwenModelArgs`.

Two construction-time deltas vs the llama3_70b_galaxy default that are
qwen3.6-specific:

1. `is_qwen36 = True` — read by `llama_*.py` files to branch into qwen3.6
   behavior (QK-norm + partial RoPE + output gate on full_attention;
   layer-dispatch in `llama_decoder.py`).

2. `use_prefetcher = False` + reclaim prefetcher-worker cores. On BH GLX
   the prefetcher is already auto-disabled in `TtModelArgs` (incompatible
   8-DRAM-bank topology), but the existing config still carves col 4 out
   of `sub_core_grids` as if it were a prefetcher sender column. With no
   prefetcher running, col 4 is unused Tensix compute. We reclaim it:
   `sub_core_grids` is now DYNAMIC — derived from the live device compute
   grid as cols `1 .. grid.x-1` × rows `0 .. grid.y-1` (col 0 reserved for
   DRAM). On BH P150 galaxy (grid (12,10)) that is cols 1-11 × 10 = 110
   cores, vs the inherited Wormhole hard-coded 60 (`(1,0)→(6,9)`) and the
   original prefetcher-aware 50-core split. ~83% more compute area.
"""
import math
import os
from pathlib import Path

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, nearest_32
from models.demos.qwen3_6_galaxy_v2.tt.model_config import (
    PREFETCHER_NOC1_GRID,
    CheckpointType,
    LlamaOptimizations,
    TtModelArgs,
    get_core_ranges,
)


class TtQwen36ModelArgs(TtModelArgs):
    """Qwen3.6-27B model configuration for BH GLX 8×4.

    Hybrid decoder: pattern `[lin, lin, lin, full] × 16` = 64 layers. The
    `linear_attention_pattern` is loaded from HF `config.layer_types` and
    is what `llama_decoder.py` switches on to dispatch between
    `TtLlamaAttention` (is_qwen36=True) and `TtQwen36DeltaAttention`.
    """

    # Op-key list inherited from base; QK_NORM_WEIGHTS added (per-head norm tables).
    OP_KEYS = (
        "EMB_WEIGHTS",
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        "QK_NORM_WEIGHTS",
        # DeltaNet-specific
        "DELTA_QKV_OUTPUT",
        "DELTA_CONV_STATE",
        "DELTA_RECURRENT_STATE",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    LOCAL_HF_PARAMS = {
        "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    }

    # Qwen3.6 supports per-head QK-norm (NOT global like olmo). RMSNorm
    # weights are head_dim=256 wide, applied per-head before partial RoPE.
    supports_batched_prefill = False  # to be revisited

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=128 * 1024,
        optimizations=LlamaOptimizations.accuracy,
    ):
        # --- Base / mesh ---
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.device_name = {0: "CPU", 1: "N150", 2: "N300", 8: "T3K", 32: "BH_GLX" if is_blackhole() else "TG"}[
            self.num_devices
        ]
        self.model_name = "Qwen3.6-27B"
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = False  # but architecturally 70B-class for sharding purposes
        self.from_hf_url = True
        self.max_prefill_chunk_size = 32768
        self.use_prefetcher = False  # always off for qwen3.6 v2 (see module docstring)
        self.max_top_k = 32

        # --- Qwen3.6-specific marker (read by llama_*.py is_qwen36 branches) ---
        self.is_qwen36 = True
        self.is_qwen = False
        self.is_olmo = False

        # --- Qwen3.6 architecture ---
        self.dim = 5120
        self.n_q_heads = 24  # full_attention path
        self.n_heads = 24  # alias used by parent class
        # V2-TP: KV heads padded 4 → 8 so the 8-row mesh axis can split heads
        # cleanly (8 / 8 = 1 KV per chip).  Replication is GQA-preserving via
        # `repeat_interleave(2, dim=0)` at weight-load time — verified bit-
        # identical in /tmp/test_gqa_kv_replication.py (V2-TP-1).
        self.n_kv_heads_unpadded = 4  # native HF
        self.n_kv_heads = 8  # padded for 2D-TP head split on rows (cluster_axis=0)
        self.head_dim = 256  # 5120 / 24 ≠ 256: heads are NOT dim/n_heads
        self.rope_dim = 64  # partial RoPE, rotary_factor=0.25
        self.partial_rotary_factor = 0.25
        self.rope_theta = 10_000_000  # MRoPE theta
        self.mrope_section = [11, 11, 10]  # text-mode collapses but kept
        self.rope_scaling_factor = 1.0
        self.orig_context_len = 262144
        self.intermediate_dim = 17_408  # SwiGLU intermediate (Qwen3.6-27B HF config.intermediate_size)
        self.vocab_size = 248_320
        self.padded_vocab_size = 248_832
        self.norm_eps = 1e-6
        self.zero_centered_norm = True  # Qwen3NextRMSNorm: w' = w + 1
        self.qk_norm = True  # per-head, not global like olmo
        self.qk_norm_per_head = True
        self.n_layers = 64
        self.unfuse_res_add = False  # qwen3.6 uses pre-norm (residual fused OK)
        self.pad_logits_to_power_of_2 = True

        # --- Hybrid decoder pattern (lin / full) ---
        # Loaded from HF config.layer_types at construction time; here we
        # set the canonical pattern [lin, lin, lin, full] × 16.
        # Populated from HF config in _set_qwen36_hf_params() below.
        self.linear_attention_pattern = None

        # --- DeltaNet hyperparams ---
        self.linear_num_key_heads = 16
        self.linear_num_value_heads = 48
        self.linear_head_dim = 128
        self.linear_conv_kernel = 4
        self.linear_dt_rank = 256  # placeholder; will read from HF config

        # --- TP factors (matches Qwen3-32B factoring on 8×4 mesh) ---
        # mesh.shape = [8, 4]; we shard heads along cluster_axis=1 (width=4)
        # and dim along cluster_axis=0 (height=8).
        self.dim_tp_factor = 4
        self.intermediate_dim_tp_factor = 8
        self.dim_per_tp = self.dim // self.dim_tp_factor  # 1280
        self.intermediate_dim_per_tp = self.intermediate_dim // self.intermediate_dim_tp_factor  # 2176

        # Base-class compatibility aliases used by some shared helpers.
        self.hidden_dim = self.intermediate_dim  # alias for parent code paths
        self.is_galaxy = True  # 32-device BH GLX / TG
        self.galaxy_type = "6U" if is_blackhole() else "4U"  # informational only
        self.is_multichip = True

        # Padded per-tp widths chosen so width % (24-core ring * tile_size == 768) == 0.
        # 2176 → pad to 5 * 768 = 3840  (intermediate per col, ring-aligned)
        # 1280 → pad to 2 * 768 = 1536  (dim per col, ring-aligned)
        # 3584 → pad to 5 * 768 = 3840  (per-col QKVG width, ring-aligned)
        self.dim_padded_24_cores = 6144  # = 4 cols × 1536
        self.intermediate_dim_per_tp_padded_24_cores = 3840
        self.qkvg_per_col_padded_24_cores = 3840  # 56-channel QKVG padded for ring matmul

        # n_heads alias (matches base TtModelArgs convention used by helpers).
        self.n_heads = self.n_q_heads
        # Full QKV linear size (Q + K + V channels, NOT including the Qwen3.6 gate).
        # head_dim * (2 * n_kv + n_q) = 256 * (2*4 + 24) = 256 * 32 = 8192
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_q_heads)

        # --- Prefetcher carve-out reclamation ---
        # The 70B / qwen3-32B config carved cols 1-3 + 5-6 of the Tensix
        # grid (col 4 reserved as prefetcher-sender). With prefetcher off,
        # col 4 is unused — reclaim it for compute.
        _, _, _, self.pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)

        # Compute grid: DYNAMIC full band derived from the live device grid.
        # Read the actual Tensix compute grid (WH TG: (7,10); BH P150 galaxy:
        # (12,10)) and span cols 1..grid.x-1 × rows 0..grid.y-1. Col 0 is
        # reserved for DRAM-adjacent ops. On BH this gives cols 1-11 × 10 rows
        # = 110 cores (vs the inherited Wormhole hard-coded 60). The old
        # "col 7 = dispatch" reservation was a WH leftover — on BH the dispatch
        # core is the last worker column, already OUTSIDE this band, and col 7
        # is normal compute. Do NOT hard-code 12 — derive it so harvested SKUs
        # (P100 = 7 DRAM banks, etc.) pick the correct width automatically.
        _compute_grid = mesh_device.compute_with_storage_grid_size()
        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0),
                    ttnn.CoreCoord(_compute_grid.x - 1, _compute_grid.y - 1),
                ),
            ]
        )
        # topk path still uses a narrower grid (3-col band).
        self.sub_core_grid_topk = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ]
        )
        self.start_core = ttnn.CoreCoord(1, 0)

        # --- HF checkpoint / cache ---
        HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3.6-27B")
        self.CKPT_DIR = HF_MODEL
        self.TOKENIZER_PATH = HF_MODEL
        cache_root = os.getenv("TT_CACHE_PATH") or os.path.join("model_cache", HF_MODEL)
        self.CACHE_PATH = os.path.join(cache_root, self.device_name)
        self.model_name = HF_MODEL

        logger.info(f"[qwen36] Checkpoint: {self.CKPT_DIR}")
        logger.info(f"[qwen36] Cache: {self.CACHE_PATH}")

        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)

        # V4: extend the parent's bf16/bf8b cache-name map with an fp32 entry so
        # `dtype=ttnn.float32` weight loading has a distinct cache dir. Used by
        # multimodal prefill PCC tests (QWEN36_FP32_WEIGHTS=1).
        self._dtype_cache_name_extra = {ttnn.float32: "tensor_cache_fp32"}
        self.tokenizer_path = self.TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        if "instruct" in self.CKPT_DIR.lower():
            self.instruct = True

        # Load HF params (overwrites layer_types, vocab, etc. with snapshot values).
        self.checkpoint_type = CheckpointType.HuggingFace
        if not dummy_weights:
            self._set_qwen36_hf_params()

        # --- Optimizations ---
        if callable(optimizations):
            self.optimizations = optimizations(self.model_name)
        else:
            self.optimizations = optimizations

        self.dummy_weights = dummy_weights
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"

        self.TG = self.num_devices == 32
        if self.num_devices != 32:
            raise ValueError(f"qwen3.6 v2 only supports BH GLX 8×4 (32 devices). got num_devices={self.num_devices}")

        # Shape / group bookkeeping.
        self.cluster_shape = list(mesh_device.shape) if mesh_device is not None else [8, 4]
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads
        self.batch_size_per_device_group = max(self.max_batch_size // self.num_device_groups, 1)
        # V2-TP: heads now split on cluster_axis=0 (rows=8) for 2D-TP.
        # Per-chip values: Q = 24/8 = 3, KV = 8/8 = 1.
        self.n_local_heads = self.n_q_heads // self.cluster_shape[0]
        self.n_local_kv_heads = self.n_kv_heads // self.cluster_shape[0]
        # V2-DRAM: fused QKVG output width per chip (V2-TP geometry).
        #   Per chip = (n_q + n_q + n_kv_padded + n_kv_padded) / n_rows * head_dim
        #           = (24 + 24 + 8 + 8) / 8 * 256 = 2048
        self.total_per_chip = (2 * self.n_q_heads + 2 * self.n_kv_heads) // self.cluster_shape[0] * self.head_dim

        # --- Model config dict (memory configs by op key) ---
        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})
        self.model_config["USE_PREFETCHER"] = self.use_prefetcher
        # BH_GLX fabric supports at most 2 links (3 hard-faults at fabric.cpp:163, confirmed by
        # tests/test_ccl_num_links_micro.py). The bring-up default was a conservative 1; the whole
        # decode CCL stack is written for up to 3 links (min(3, GALAXY_NUM_LINKS)). DEFAULT NOW 2:
        # PCC bit-identical (RS micro 0.9999956 at 1 and 2 links), 64L A/B +1.7% coherent, combined-wins
        # +3.8% coherent. Set QWEN36_GALAXY_NUM_LINKS=1 to revert. (clamped to 2 max on BH regardless.)
        _bh_links = int(os.environ.get("QWEN36_GALAXY_NUM_LINKS", "2"))
        self.model_config["GALAXY_NUM_LINKS"] = min(2, _bh_links) if is_blackhole() else 4
        self.model_config["CCL_TOPOLOGY"] = ttnn.Topology.Linear if is_blackhole() else ttnn.Topology.Ring
        self.model_config["IS_QWEN36"] = True

        # Populate per-op program configs. Mirrors the structure of qwen_model_config.py
        # but uses qwen3.6 shapes (n_q=24, n_kv=4, head_dim=256, intermediate=13824)
        # and the 60-core full grid (sub_core_grids = (1,0)→(6,9)).
        if mesh_device is not None:
            self._populate_program_configs(mesh_device)

        self.tokenizer = None  # populated when actually needed

    # ------------------------------------------------------------------
    # Per-op program configs (adapted from qwen_model_config.py for
    # qwen3.6 shapes + 60-core grid + prefetcher-off layout)
    # ------------------------------------------------------------------
    def _populate_program_configs(self, mesh_device):
        """Build every program-config / memory-config dict entry that the
        v2 ``tt/*.py`` files read at construction time.

        Adapted from ``qwen_model_config.TtQwenModelArgs.__init__`` Galaxy
        branch.  Diverges where qwen3.6 shapes (head_dim 256, n_q 24, GQA 6:1,
        QKVG fused 56-channel buffer, intermediate 13824, vocab 248832) or the
        60-core full grid (cols 1-6 × rows 0-9) require non-trivial shape
        changes.  Values targeted to be construction-compatible — production
        tuning happens on the first real device run.
        """
        grid = mesh_device.compute_with_storage_grid_size()
        self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)
        # DRAM grid (12 DRAM banks on a horizontal stripe at the top).
        self.dram_weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                )
            }
        )

        # ------------------------------------------------------------------
        # Compute-kernel configs (attributes + dict aliases)
        # ------------------------------------------------------------------
        self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )
        self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # V4-9 (investigated, neither helped — kept here for documentation):
        # • QWEN36_FORCE_HIFI4=1 caused matmul subblock register-count TT_FATAL
        #   (HiFi4 doubles dest-register usage; existing program configs overflow)
        # • QWEN36_HIFI2_EXACT=1 (math_approx_mode=False) produced identical PCC
        #   (matmul code path doesn't use the approximations these flags affect)
        # The bf16+bf8b multimodal-prefill PCC ceiling is ~0.83 last-token vs HF
        # reference, limited by smooth ~0.6%/layer compounding on vision-feature
        # inputs with outlier channels (abs max ~186). Closing further requires
        # retuning matmul subblocks for HiFi4 or reverting bf8b → bf16 matmul
        # output dtype (V2-CONFIG-E) for the prefill path.
        self.model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = self.compute_kernel_config_hifi2
        self.model_config["LO_FI_COMPUTE_CONFIG"] = self.compute_kernel_config_lofi

        # ------------------------------------------------------------------
        # CCL / multichip bookkeeping
        # ------------------------------------------------------------------
        self.num_reduce_scatter_links = 1
        self.num_all_gather_links = 1 if is_blackhole() else 2
        self.ccl_dtype = ttnn.bfloat8_b

        # ------------------------------------------------------------------
        # Core-range sets used by ring matmuls / packet workers
        # ------------------------------------------------------------------
        # 24-core ring (input side). prefetcher cores live in cols 1, 2, 5, 6
        # which are all inside the 60-core sub_core_grids — safe to reuse.
        RING_SIZE = 24
        ring_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID]
        )
        # 24-core ring output side (pf-receiver cores: same shape, mirrored).
        pf_mm_out_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.pf_receiver_cores_list]
        )

        # ------------------------------------------------------------------
        # Decode residual (skip / add) memory layout
        # ------------------------------------------------------------------
        # dim_per_tp = 1280 → 10 tile-cols × 32 per shard on a 5×2 band of cols 1-2.
        num_cores_ln = 10
        core_grid_ln, grid_offset = (5, 2), ttnn.CoreCoord(1, 0)
        residual_core_range = ttnn.CoreRange(
            grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
        )
        self.model_config["DECODE_RESIDUAL_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, self.dim_per_tp // num_cores_ln),  # (1,1,32,128)
            core_grid=ttnn.CoreRangeSet([residual_core_range]),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # SDPA program configs (prefill + decode)
        # ------------------------------------------------------------------
        sdpa_grid = (grid.x, grid.y)
        self.model_config["SDPA_PROGCFG"] = lambda seqlen, chunk_start_idx=0, _g=sdpa_grid: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_g,
            exp_approx_mode=False,
            q_chunk_size=256
            if seqlen >= 2048 and chunk_start_idx == 0
            else 64
            if seqlen < 2048 and chunk_start_idx == 0
            else min(256, chunk_start_idx & -chunk_start_idx)
            if seqlen >= 2048
            else min(64, chunk_start_idx & -chunk_start_idx),
            k_chunk_size=512
            if seqlen >= 2048 and chunk_start_idx == 0
            else 64
            if seqlen < 2048 and chunk_start_idx == 0
            else min(512, (seqlen + chunk_start_idx) & -(seqlen + chunk_start_idx))
            if seqlen >= 2048
            else min(64, (seqlen + chunk_start_idx) & -(seqlen + chunk_start_idx)),
        )
        self.model_config[
            "SDPA_PROGCFG_FLEXIBLE_CHUNK"
        ] = lambda seqlen, page_size, _g=sdpa_grid: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_g,
            exp_approx_mode=False,
            q_chunk_size=min(page_size, 128),
            k_chunk_size=min(page_size, 128),
        )

        # Decode SDPA — use sub_core_grids (60 cores available). 48 / 32 cores.
        self.model_config["PAGED_SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 48, self.sub_core_grids, row_wise=True
            ),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 32, self.sub_core_grids, row_wise=True
            ),
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )
        self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # SDPA output memory config (sharded by batch). qwen3.6 has 6 local
        # Q heads (24/4); the tile padding rounds up to 32.
        self.model_config[
            "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
        ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
            shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # (32, 256)
            core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, max(1, batch_size_per_device_group), self.sub_core_grids, row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # Use-fused-all-gather-matmul flag (decode WO ring)
        # ------------------------------------------------------------------
        self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
            self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
            and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
            and self.num_devices > 1
        )

        # ------------------------------------------------------------------
        # Decode QKV ring (XQKV decode)
        # Per-col QKVG width = 3584 (6Q + 6Gate + 1K + 1V) × 256. Padded to 3840
        # for 24-core ring tile alignment.
        # ------------------------------------------------------------------
        qkvg_n_padded = self.qkvg_per_col_padded_24_cores  # 3840
        self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_padded_24_cores // 4 // RING_SIZE),  # (32, 1536/24 = 64)
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_QKV_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.dim_per_tp,  # 1280
            n=qkvg_n_padded,  # 3840
        )
        self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, qkvg_n_padded // RING_SIZE),  # (32, 160)
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["XQKV_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            self.dim_per_tp,
            qkvg_n_padded,
            RING_SIZE,
            prefetch=self.use_prefetcher,
            untilize_out=True,
        )

        # Packet worker shard for RS+create-heads interim buffer.
        RS_CREATE_HEADS_PACKET_WORKER_CRS = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
            ]
        )
        self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 512),
            core_grid=RS_CREATE_HEADS_PACKET_WORKER_CRS,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # Decode WO ring (attention output → residual)
        # qwen3.6 WO input K = n_local_heads × head_dim = 6 × 256 = 1536 per col.
        # N = dim_per_tp = 1280, padded to 1536 for 24-core ring alignment.
        # ------------------------------------------------------------------
        wo_k_decode = self.n_local_heads * self.head_dim  # 1536
        wo_n_padded = self.dim_padded_24_cores // 4  # 1536
        self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, wo_k_decode // RING_SIZE),  # (32, 64)
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_WO_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=wo_k_decode,
            n=wo_n_padded,
        )
        self.model_config["SHARDED_WO_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, wo_n_padded // RING_SIZE),  # (32, 64)
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["WO_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            wo_k_decode,
            wo_n_padded,
            RING_SIZE,
            prefetch=self.use_prefetcher,
        )

        # ------------------------------------------------------------------
        # Decode FF1/FF3/FF2 ring
        # K = dim_per_tp = 1280, N = intermediate_per_tp_padded = 3840
        # ------------------------------------------------------------------
        ff_n_padded = self.intermediate_dim_per_tp_padded_24_cores  # 3840
        ff_k_native = self.intermediate_dim_per_tp  # 2176 (native, used as W2's K)
        # W1W3 / W2 DRAM-sharded weight memcfgs (read by llama_mlp.py at
        # weight upload time).  K must be the *native* per-device tensor
        # height; only the N dim is padded (the ring matmul tolerates the
        # extra zero-padded output columns).  Matches qwen3-32B precedent.
        # ------------------------------------------------------------------
        # V2-DRAM-P1: standalone DRAM-sharded matmul fast-path for V2-TP
        # attention WQKVG + WO (atupe/qwen35-27b-4xp150 pattern).  These
        # bypass the 24-core ring + prefetcher infrastructure; the matmul
        # kernel itself streams weights from DRAM-sharded banks into L1
        # width-sharded activations.  Used by _forward_decode_qwen36.
        #
        # Per-chip dims (V2-TP geometry):
        #   WQKVG: M=32 (tile-padded T=1), K=1280 (H/cols), N=2048 (qkvg/8)
        #   WO:    M=32,                   K=768  (n_q_pc*hd), N=1280 (H/cols)
        # ------------------------------------------------------------------
        v2tp_M = 32
        v2tp_wqkvg_K = self.dim_per_tp  # 1280
        v2tp_wqkvg_N = self.total_per_chip  # 2048
        v2tp_wo_K = self.n_local_heads * self.head_dim  # 768
        v2tp_wo_N = self.dim_per_tp  # 1280

        self.model_config["V2TP_WQKVG_WEIGHT_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=v2tp_wqkvg_K, n=v2tp_wqkvg_N
        )
        self.model_config["V2TP_WO_WEIGHT_MEMCFG"] = self.create_dram_sharded_mem_config(k=v2tp_wo_K, n=v2tp_wo_N)
        self.model_config["V2TP_WQKVG_ACT_MEMCFG"] = self._v2tp_create_activation_shard_config(v2tp_wqkvg_K)
        self.model_config["V2TP_WO_ACT_MEMCFG"] = self._v2tp_create_activation_shard_config(v2tp_wo_K)
        self.model_config["V2TP_WQKVG_PROGCFG"] = self._v2tp_create_dram_sharded_matmul_progcfg(
            m=v2tp_M, k=v2tp_wqkvg_K, n=v2tp_wqkvg_N
        )
        self.model_config["V2TP_WO_PROGCFG"] = self._v2tp_create_dram_sharded_matmul_progcfg(
            m=v2tp_M, k=v2tp_wo_K, n=v2tp_wo_N
        )

        self.model_config["W1W3_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.dim_per_tp,  # 1280
            n=ff_n_padded,  # 3840
        )
        self.model_config["W2_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=ff_k_native,  # 2176 (must equal physical tensor height per-device)
            n=wo_n_padded,  # 1536
        )
        self.model_config["FF1_3_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            self.dim_per_tp,
            ff_n_padded,
            RING_SIZE,
            prefetch=self.use_prefetcher,
        )
        self.model_config["FF2_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            ff_n_padded,
            wo_n_padded,
            RING_SIZE,
            prefetch=self.use_prefetcher,
        )
        self.model_config["SHARDED_FF12_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_padded_24_cores // 4 // RING_SIZE),  # (32, 64)
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, ff_n_padded // RING_SIZE),  # (32, 96)
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["FF2_IN_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, ff_n_padded // RING_SIZE),  # (32, 96)
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["FF2_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, wo_n_padded // RING_SIZE),  # (32, 64)
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # Decode FF1/FF3/FF2 RING-40 (lower-waste fused-FF2 config set).
        # SEPARATE from the ring-24 keys above — does NOT touch the global
        # RING_SIZE=24 (pervasive in attention QKVG). A clean ring forces
        # padding because native intermediate 2176 (=68 tiles) shares no ring
        # divisor with K=dim_per_tp=1280 (=40 tiles). Ring-40 divides K cleanly
        # (1280/40 = 32 = 1 tile/core, NO pad) and pads N only 2176->2560
        # (vs ring-24's 3840, -76% waste).
        #   FF1/FF3 ring-40: M=32, K=1280, N=2560  (per_core_N=2, in0_block_w=1)
        #   FF2     ring-40: M=32, K=2560, N=1280  (per_core_N=1, in0_block_w=2)
        # num_to_coregrid(40) = CoreGrid(y=5, x=8) = 40 cores -> grid (8,5).
        # ------------------------------------------------------------------
        RING40_SIZE = 40
        ff_n_ring40 = 2560  # intermediate_per_tp 2176 padded to 40-core ring
        # 40-core ring placement for the FUSED all_gather_matmul (ring-40 FF2).
        #
        # CRITICAL (fused-op core placement): the fused
        # `llama_all_gather_matmul_async` places its all-gather interim /
        # `worker_receiver` CCL cores on a dedicated column (col 3, rows 0-3 —
        # `RING40_AG_INTERIM_CRS` below, mirroring the proven ff2_qwen
        # `intermediate_core_range_set = CoreRange((3,0),(3,3))`). The 40 matmul
        # cores MUST be DISJOINT from (a) col 0 (dispatch cores) and (b) col 3
        # (the AG-interim column) or the matmul's `in0_ring_all_gather` reader
        # collides with `worker_receiver` (TT_FATAL Core Overlap at (3,0)).
        #
        # The plain `num_cores_to_corerangeset_in_subcoregrids(start_core, 40,
        # sub_core_grids)` spills onto col 3, so we carve the 40 cores from a
        # sub-grid that EXCLUDES col 0 and col 3: cols {1,2} ∪ {4,5} × rows 0-9
        # = 40 cores. (The matmul's `compute_with_storage_grid_size` bounding box
        # need not contain these cores — for `gather_in0=True` ring matmuls
        # placement follows the sharded memcfg core range set, exactly as
        # ff2_qwen places 24 cores at y∈[0,9] under an (8,3) storage grid.)
        RING40_MM_SUBGRID = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 9)),  # cols 1,2
                ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 9)),  # cols 4,5
            ]
        )
        ring40_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), RING40_SIZE, RING40_MM_SUBGRID, row_wise=True
        )
        # Dedicated AG-interim column (4 cores = cluster_axis=1 ring size), DISJOINT
        # from the 40 matmul cores above and from col 0 (dispatch). Callers pass the
        # ag_memory_config built on this set so the fused op's interim/worker_receiver
        # land here, not on a matmul core.
        self.model_config["RING40_AG_INTERIM_CRS"] = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3))]
        )
        self.model_config["RING40_MM_CRS"] = ring40_core_range_set

        self.model_config["W1W3_RING40_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.dim_per_tp,  # 1280
            n=ff_n_ring40,  # 2560
        )
        self.model_config["W2_RING40_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.intermediate_dim_per_tp,  # 2176 (native per-device tensor height;
            # only N is padded — the ring matmul tolerates the K-side zero pad, same
            # convention as W2_RING_MEMCFG above)
            n=self.dim_per_tp,  # 1280
        )
        self.model_config["FF1_3_RING40_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            self.dim_per_tp,  # K = 1280
            ff_n_ring40,  # N = 2560
            RING40_SIZE,
            prefetch=self.use_prefetcher,
        )
        self.model_config["FF2_RING40_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            ff_n_ring40,  # K = 2560
            self.dim_per_tp,  # N = 1280
            RING40_SIZE,
            prefetch=self.use_prefetcher,
        )
        # Sharded L1 memcfgs (40-core ring widths).
        self.model_config["SHARDED_FF12_RING40_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_per_tp // RING40_SIZE),  # (32, 32) FF12 input (K=1280)
            core_grid=ring40_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_FF12_OUT_RING40_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, ff_n_ring40 // RING40_SIZE),  # (32, 64) FF12 out (N=2560)
            core_grid=ring40_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["FF2_IN_RING40_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, ff_n_ring40 // RING40_SIZE),  # (32, 64) FF2 in (2560)
            core_grid=ring40_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["REDUCE_SCATTER_OUT_RING40_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_per_tp // RING40_SIZE),  # (32, 32) FF2 out (N=1280)
            core_grid=ring40_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # Attention-input sharded memcfg (post-RMSNorm decode input).
        # ------------------------------------------------------------------
        # 60-core full grid: derive an N-core band that holds dim_per_tp.
        # 1280 % (8 × N) == 0 for N ∈ {1,2,4,5,8,10,16,20,40} — pick 8 rows × 4 cols
        # giving 32 cores. (Was 32 cores on the 50-core split grid too.)
        attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)  # CoreGrid
        attn_input_sub_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, 32, self.sub_core_grids, row_wise=True
        )
        # Per-shard width: ensure tile-aligned.
        attn_in_shard_w = max(self.tile_size, nearest_32(self.dim_per_tp // 32))
        self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, attn_in_shard_w),
            core_grid=attn_input_sub_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # MLP input (decode, pre-FF1/FF3) sharded memcfg
        # ------------------------------------------------------------------
        mlp_core_grid = self.dram_shard_core_grid_for_k(self.dim)
        self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            (32, self.dim // mlp_core_grid.num_cores),
            mlp_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # CREATE_HEAD_INPUT / OUTPUT memcfg (nlp_create_heads_decode)
        # ------------------------------------------------------------------
        # Use a 10-core 5x2 band of sub_core_grids cols 1-2.
        shard_spec_n_cores_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, 10, self.sub_core_grids, row_wise=False
        )
        self.model_config["CREATE_HEAD_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_n_cores_grid,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        # qwen3.6 head_dim = 256 (vs. 70B 128). The CREATE_HEAD_OUTPUT shard
        # holds (Q | K | V) heights × head_dim cols. The actual head-dim layout
        # is handled at the op level; here we just lay out 32×128 shards
        # height-first on the full 60-core grid (head-dim re-shaping handled
        # downstream of nlp_create_heads_decode).
        self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_core_grids,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # ------------------------------------------------------------------
        # GATHER_USERS memcfg (post-SDPA gather across users → cluster_axis=0).
        # ------------------------------------------------------------------
        self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 32, self.sub_core_grids, row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ------------------------------------------------------------------
        # Reduce-scatter intermediate / output configs (decode all_reduce)
        # ------------------------------------------------------------------
        PACKET_WORKER_CRS = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
            ]
        )
        self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 512),
            core_grid=PACKET_WORKER_CRS,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # RS-out: 30-core band of sub_core_grids — matches qwen3-32B.
        FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), 30, self.sub_core_grids, row_wise=True
        )
        self.model_config["REDUCE_SCATTER_OUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                FF1_CRS_RS_OUT,
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # ------------------------------------------------------------------
        # Sharded RMSNorm program configs (attn / mlp / lm_head positions).
        # ------------------------------------------------------------------
        self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
        self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)

        # ------------------------------------------------------------------
        # LM-head: padded_vocab=248832 → 248832/4 = 62208 per col (Galaxy 4-way).
        # 62208 / (RING_SIZE=24 × tile=32) = 81 → well-aligned for 24-core ring.
        # ------------------------------------------------------------------
        # Find largest lm_head_num_rows so dim % (32*32*rows) == 0.  For dim=5120
        # → 5120 / 1024 = 5 → use 4 rows.  (Same path as qwen3-32B.)
        lm_head_num_rows = 4
        while self.dim % (32 * 32 * lm_head_num_rows) != 0:
            lm_head_num_rows -= 1
        assert lm_head_num_rows > 0, f"Could not find lm_head_num_rows for dim={self.dim}"
        self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)
        self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            (self.tile_padded_batch_rows, nearest_32(self.dim_per_tp // self.lm_head_core_grid.num_cores)),
            self.lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(self.lm_head_core_grid)

        # LM-head ring grid.  The ring matmul splits the per-col vocab across
        # LM_HEAD_RING_SIZE cores.  This config was hard-coded to 24 cores
        # (qwen3-32B's coordinate lists, inside the old 60-core band) and is now
        # PARAMETRIC + dynamic so it can be widened on the post-60->110 grid via
        # the QWEN36_LM_HEAD_RING_SIZE knob.
        #
        # Sizing math: per-col vocab = padded_vocab/4 = 62208 = 1944 tiles
        # (=2^3 * 3^5).  For ZERO padding the ring size must divide 1944.  The
        # 1D-ring matmul lays its compute cores out on the rectangular
        # num_to_coregrid(N) grid, which is only valid for multiples of 8 (plus
        # 12/20) — so among the clean divisors {54,72,81,108} only 72 = 8x9 is
        # realizable (3x the old 24, exact 1944/72 = 27 tiles/core, no vocab
        # padding, fits inside (12,10)).  81/108/54 are rejected by
        # num_to_coregrid (not 8-multiples) and don't form a rectangle in the
        # grid with col-0 reserved.
        #
        # MEASURED (BH galaxy, ISL-128, identical build): 24 cores = 20.53
        # tok/s/user (48.71 ms/step), 72 cores = 20.53 tok/s/user (48.72
        # ms/step) — IDENTICAL.  128k demo: both 18.5 tok/s/user, coherent.
        # The LM-head is DRAM-weight-bandwidth-bound (the ~80 MB bf8
        # 1280x62208 weight read per token across 8 DRAM banks), NOT
        # core-count-bound, so widening the ring buys nothing.  Default kept at
        # 24 (fewer cores, no contention with neighbouring ops); set
        # QWEN36_LM_HEAD_RING_SIZE=72 to use the wide ring (validated coherent).
        from models.demos.qwen3_6_galaxy_v2.tt.model_config import LM_HEAD_16_GRID, LM_HEAD_32_GRID

        LM_HEAD_RING_SIZE = int(os.environ.get("QWEN36_LM_HEAD_RING_SIZE", "24"))
        # Per-col padded vocab: pad to multiple of LM_HEAD_RING_SIZE * tile.
        per_col_vocab = self.padded_vocab_size // self.cluster_shape[1]  # 62208
        RING_TILE_ALIGN = LM_HEAD_RING_SIZE * self.tile_size  # 24->768, 72->2304
        # per_col_vocab_padded must be tile-aligned BOTH for the RING_SIZE matmul shard
        # (per_col/RING_SIZE % tile == 0 -> %RING_TILE_ALIGN) AND for the 32-core decode
        # RESHARD (LM_HEAD_OUT_RING_RESHARD_MEMCFG: per_col/32 % tile == 0 -> %(32*tile)).
        # Padding only to RING_TILE_ALIGN left per_col/32 = 1944 (not %32) -> the decode
        # lm_head reshard hit tensor_layout.cpp:112 (physical_shard not tile-aligned). Pad
        # to the LCM of both so every derived shard is tile-aligned (62208 -> 64512:
        # 64512/24=2688, 64512/32=2016, both %32).
        _reshard_tile_align = 32 * self.tile_size  # 1024 (32 reshard cores * tile)
        _vocab_align = (RING_TILE_ALIGN * _reshard_tile_align) // math.gcd(RING_TILE_ALIGN, _reshard_tile_align)
        per_col_vocab_padded = ((per_col_vocab + _vocab_align - 1) // _vocab_align) * _vocab_align
        self.lm_head_shape = (self.dim_per_tp, per_col_vocab_padded)  # (1280, 62208)

        # Input/output ring shards live on LM_HEAD_RING_SIZE cores carved from
        # the live (widened) sub_core_grids (cols 1..grid.x-1, col 0 reserved).
        # The 1D-ring matmul itself places compute on num_to_coregrid(RING_SIZE)
        # (= 8x9 for 72), starting at (0,0); ttnn reshards in0/out to these
        # explicit shard grids, so the exact coords here only need to be
        # RING_SIZE distinct cores with tile-aligned shard widths.
        lm_head_ring_core_input_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, LM_HEAD_RING_SIZE, self.sub_core_grids, row_wise=True
        )
        lm_head_ring_core_output_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, LM_HEAD_RING_SIZE, self.sub_core_grids, row_wise=True
        )
        # RESHARD + 16-core input layouts stay on their original (fixed) grids —
        # the post-ring all-reduce uses 32 cores independent of RING_SIZE.
        lm_head_ring_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_32_GRID]
        )
        lm_head_ring_16_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_16_GRID]
        )

        # Input shard: dim_per_tp=1280 padded up to a multiple of
        # RING_SIZE*tile so the K-shard is tile-aligned.  At 72: 2304 / 72 = 32
        # (1 tile/core).  At 24 this was 1536 / 24 = 64 (2 tiles/core).
        lm_head_in_padded = (
            (self.dim_per_tp + RING_TILE_ALIGN - 1) // RING_TILE_ALIGN
        ) * RING_TILE_ALIGN  # 24->1536, 72->2304
        self.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, lm_head_in_padded // LM_HEAD_RING_SIZE),  # 24->(32,64), 72->(32,32)
            core_grid=lm_head_ring_core_input_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_LM_HEAD_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.lm_head_shape[0] // 16),  # (32, 80)
            core_grid=lm_head_ring_16_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, per_col_vocab_padded // LM_HEAD_RING_SIZE),  # 24->(32,2592), 72->(32,864)
            core_grid=lm_head_ring_core_output_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_OUT_RING_RESHARD_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, per_col_vocab_padded // 32),  # (32, 1944)
            core_grid=lm_head_ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_TG_RING_PROGCFG"] = self.matmul_1d_ring_lm_head_config(
            1,
            32,
            lm_head_in_padded,
            per_col_vocab_padded,
            LM_HEAD_RING_SIZE,
            prefetch=False,
        )
        # LM head prefill: matmul_1d_config over (32 × dim_per_tp) × (dim_per_tp × N).
        # Use MinimalMatmulConfig (no dependency on awkward output dims).
        # BH grid retune: the 7x7=49 grid is WH-sized; BH has 12x10=120 cores. QWEN36_LM_HEAD_GRID
        # DEFAULT NOW "12,10" (full BH grid): 1.56x on-op vs 7x7, PCC 0.99994 unchanged
        # (test_lm_head_grid_micro), combined-wins +3.8% coherent. Set "7,7" to revert to the legacy grid.
        _lmh_gx, _lmh_gy = (int(v) for v in os.environ.get("QWEN36_LM_HEAD_GRID", "12,10").split(","))
        self.model_config["LM_HEAD_PREFILL_PROGCFG"] = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(_lmh_gx, _lmh_gy),
        )

        # ------------------------------------------------------------------
        # Prefill MLP / XQKV / WO program configs (sequence-length switch).
        # qwen3.6 intermediate_per_tp = 1728 — adapt block-shape math.
        # ------------------------------------------------------------------
        def w1_w3_prg_config(seq_len, use_w1_w3_interleaved=False):
            if seq_len <= 128:
                return self.matmul_1d_config(
                    seq_len,
                    self.dim_per_tp,
                    self.intermediate_dim_per_tp,
                    grid=ttnn.CoreGrid(x=7, y=4),
                    overwrite_per_core_k=4,
                )
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 8),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=max(1, 8 if seq_len >= 2048 else max(1, seq_len // self.tile_size // 8)),
                per_core_N=math.ceil(self.intermediate_dim_per_tp / self.tile_size / 5),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

        def w2_prg_config(seq_len):
            if seq_len <= 128:
                return self.matmul_1d_config(
                    seq_len,
                    self.intermediate_dim_per_tp,
                    self.dim_per_tp,
                    grid=ttnn.CoreGrid(x=7, y=10),
                    overwrite_per_core_k=2,
                )
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(5, 10),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=max(1, 8 if seq_len >= 2048 else max(1, seq_len // self.tile_size // 8)),
                per_core_N=math.ceil(self.dim_per_tp / self.tile_size / 5),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

        self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = w1_w3_prg_config
        self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = w2_prg_config

        def _prefill_minimal(seq_len, _hk=8, _wk=8):
            # ``subblock_h * subblock_w`` must be <= the matmul's max_dest_volume.
            # Match llama3_70b_galaxy/tt/model_config.py, which keeps the product
            # at <= 8 for EVERY FF minimal-matmul bucket (it never uses 16, even
            # for FF1/FF3 which run under fp32-dest-off). qwen36 previously used
            # (4,4)=16 for FF1/FF3 at seq_len>4096: it does not assert (lofi cap
            # is 16) but produces incorrect results -> garbage output at T=8192.
            # Both ranges below give a product of 8:
            #   seq_len <= 4096: subblock_h=4, subblock_w=2  -> 8
            #   seq_len >  4096: subblock_h=2, subblock_w=4  -> 8
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=_hk if seq_len <= 4096 else _wk,
                subblock_w=2 if seq_len <= 4096 else 4,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 8 if seq_len <= 4096 else 9),
            )

        # subblock_h: 4 for seq_len<=4096 (with subblock_w=2 -> 8), 2 for
        # seq_len>4096 (with subblock_w=4 -> 8). Same for FF1/FF3 and FF2 so
        # both stay within the fp32-dest budget, mirroring llama70b.
        self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"] = lambda seq_len: _prefill_minimal(seq_len, 4, 2)
        self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"] = lambda seq_len: _prefill_minimal(seq_len, 4, 2)

        # Prefill XQKV / WO — qwen3.6 prefill XQKV writes QKVG per-col padded width.
        self.model_config["XQKV_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len,
                self.dim_per_tp,
                qkvg_n_padded,
                grid=ttnn.CoreGrid(x=4, y=10),
                overwrite_per_core_k=8,
            )
            if seq_len <= 128
            else ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=max(1, 8 if seq_len >= 2048 else max(1, seq_len // self.tile_size // 8)),
                per_core_N=math.ceil(qkvg_n_padded / self.tile_size / 7),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )
        )

        def prefill_xqkv_minimal(seq_len):
            if seq_len <= 128:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
                )
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
            )

        self.model_config["XQKV_PREFILL_MINIMAL_PROGCFG"] = prefill_xqkv_minimal

        self.model_config["WO_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len,
                wo_k_decode,
                self.dim_per_tp,
                grid=ttnn.CoreGrid(x=7, y=10),
                overwrite_per_core_k=8,
            )
            if seq_len <= 128
            else ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=max(1, 4 if seq_len >= 1024 else max(1, seq_len // self.tile_size // 8)),
                per_core_N=math.ceil(self.dim_per_tp / self.tile_size / 7),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 1024,
            )
        )

        def prefill_wo_minimal(seq_len):
            if seq_len <= 4096:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=8,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
                )
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=4,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
            )

        self.model_config["WO_PREFILL_MINIMAL_PROGCFG"] = prefill_wo_minimal

        # ------------------------------------------------------------------
        # KV prefill mem-config (sharded by seq_len).
        # ------------------------------------------------------------------
        # V2-TP: KV heads split on rows (cluster_axis=0); 8 padded KV / 8 rows = 1 per chip.
        assert self.n_kv_heads % self.cluster_shape[0] == 0
        self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / max(1, (self.n_kv_heads // self.cluster_shape[0]))
        self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
            (((self.n_kv_heads // self.cluster_shape[0]) * seq_len // (8 * 8)), self.head_dim),
            ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # ------------------------------------------------------------------
    # HF param loading (qwen3.6-specific keys)
    # ------------------------------------------------------------------
    def _set_qwen36_hf_params(self):
        """Read HF config.json to populate:
        - linear_attention_pattern (from config.layer_types)
        - vocab_size, dt_rank, etc.
        """
        import json

        config_path = self.model_base_path / "config.json"
        if not config_path.exists():
            logger.warning(f"[qwen36] config.json not found at {config_path}; using defaults")
            self.linear_attention_pattern = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ] * 16
            return
        with open(config_path) as f:
            hf_cfg = json.load(f)
        text_cfg = hf_cfg.get("text_config", hf_cfg)

        self.linear_attention_pattern = text_cfg.get(
            "layer_types",
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
        )
        if len(self.linear_attention_pattern) != self.n_layers:
            logger.warning(f"[qwen36] layer_types len {len(self.linear_attention_pattern)} != n_layers {self.n_layers}")

        self.vocab_size = text_cfg.get("vocab_size", self.vocab_size)
        self.rope_theta = text_cfg.get("rope_theta", self.rope_theta)
        rope_scaling = text_cfg.get("rope_scaling") or {}
        if "mrope_section" in rope_scaling:
            self.mrope_section = rope_scaling["mrope_section"]

        # DeltaNet block params
        self.linear_num_key_heads = text_cfg.get("linear_num_key_heads", self.linear_num_key_heads)
        self.linear_num_value_heads = text_cfg.get("linear_num_value_heads", self.linear_num_value_heads)
        self.linear_head_dim = text_cfg.get("linear_head_dim", self.linear_head_dim)
        self.linear_conv_kernel = text_cfg.get("linear_conv_kernel_dim", self.linear_conv_kernel)

        # Partial RoPE
        self.partial_rotary_factor = text_cfg.get("partial_rotary_factor", self.partial_rotary_factor)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

    def weight_cache_path(self, dtype):
        # Override parent's bf16/bf8b-only mapping to support fp32 (V4 multimodal
        # PCC push). Falls through to parent for the standard two dtypes.
        if dtype == ttnn.float32:
            cache_name = "tensor_cache_fp32_instruct" if self.instruct else "tensor_cache_fp32"
            return self.model_cache_path / cache_name
        return super().weight_cache_path(dtype)

    # Shape helper required by parent infrastructure.
    def cluster_shape_xy(self):
        return tuple(self.cluster_shape)

    def is_distributed_norm(self, mode):
        return True

    @property
    def ccl_topology(self):
        return self.model_config["CCL_TOPOLOGY"]

    def get_state_dict_prefix(self, module_name, layer_num):
        # standardize_hf_keys_qwen36() converts raw HF keys
        # (model.language_model.layers.{i}.<m>.weight) to the meta-style
        # internal layout the 70B/qwen3-32B/olmo tree consumes
        # (layers.{i}.<m>.weight; top-level keys carry no prefix).
        # llama_decoder / llama_attention / llama_embedding read tensors
        # via this prefix at construction; for is_qwen36 we must match
        # the standardized layout, NOT the raw HF layout.
        #
        # The upstream 70B path also maps class names (TtLlamaMLP/Attention)
        # to the meta module names used in load_checkpoints' rename pass
        # (feed_forward / attention). Mirror that here so TtLlamaMLP's
        # `{prefix}.w1.weight` lookup resolves to `layers.{i}.feed_forward.w1.weight`.
        module_map = {
            "TtLlamaMLP": "feed_forward",
            "TtLlamaAttention": "attention",
            "TtTransformerBlock": "",
            "": "",
        }
        # Allow direct meta names (e.g. "feed_forward") and unknown module names
        # (e.g. the qwen3.6-specific TtQwen36DeltaAttention which sets its own
        # state_dict_prefix internally) to fall through unchanged.
        mapped = module_map.get(module_name, module_name)
        if layer_num is None:
            # Top-level weights (tok_embeddings.weight, norm.weight,
            # output.weight, etc.) — no prefix, no leading dot.
            return mapped
        return f"layers.{layer_num}.{mapped}"

    def create_dram_sharded_mem_config(self, k, n):
        """BH-aware DRAM-sharded memory config for width-sharded weights.

        Upstream ``TtModelArgs.create_dram_sharded_mem_config`` hardcodes
        ``dram_cores = 12`` (WH dram grid). On Blackhole Galaxy the DRAM grid
        is 8x1; using 12 shards trips ``num_shards <= num_cores`` at weight
        upload. Pull the actual DRAM grid width from ``self.dram_weight_grid``
        (set in ``__init__`` from ``mesh_device.dram_grid_size()``).
        """
        dram_cores = self.dram_weight_grid.bounding_box().grid_size().x
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid,
            (k, padded_size // dram_cores),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    # ------------------------------------------------------------------
    # V2-DRAM-P1: standalone DRAM-sharded matmul helpers (atupe pattern).
    # Used by _forward_decode_qwen36 to drive WQKVG + WO via the fast
    # MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig kernel.
    # ------------------------------------------------------------------
    @staticmethod
    def _v2tp_find_largest_divisor(n, max_div=8):
        for d in range(max_div, 0, -1):
            if n % d == 0:
                return d
        return 1

    @staticmethod
    def _v2tp_find_grid(n_tiles, target=32, max_rows=8, max_cols=8):
        """Find (rows, cols) on ≤ max_rows × max_cols grid where n_tiles is
        evenly divided.  Prefer grids closest to ``target`` cores."""
        max_cores = max_rows * max_cols
        possible = [k for k in range(1, max_cores + 1) if n_tiles % k == 0]
        possible.sort(key=lambda x: abs(x - target))
        for cores in possible:
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols
        raise AssertionError(f"Cannot find grid for {n_tiles} tiles in {max_rows}x{max_cols}")

    def _v2tp_create_activation_shard_config(self, k):
        """L1 WIDTH_SHARDED activation memcfg sized to feed the V2-DRAM matmul.

        Picks a grid that evenly divides K_tiles so each core gets a contiguous
        K-stripe of the activation.  Used as the input layout to the
        DRAM-sharded matmul kernel.
        """
        k_tiles = k // self.tile_size
        rows, cols = self._v2tp_find_grid(k_tiles)
        num_cores = rows * cols
        width_per_core = k // num_cores
        return ttnn.create_sharded_memory_config(
            shape=(self.tile_size, width_per_core),
            core_grid=ttnn.CoreGrid(x=cols, y=rows),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _v2tp_create_dram_sharded_matmul_progcfg(self, m, k, n):
        """``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` builder.

        Same shape derivation as atupe/qwen35-27b-4xp150's
        ``create_dram_sharded_matmul_program_config``: pick K-grid via
        ``_find_grid(k_tiles)``, pad N to ``tile_size * dram_cores``, set
        ``per_core_N = n_tiles // num_cores``.  Kernel internally fans out
        the N dim across compute cores.
        """
        dram_cores = self.dram_weight_grid.bounding_box().grid_size().x
        m_tiles = math.ceil(m / self.tile_size)
        k_tiles = math.ceil(k / self.tile_size)
        n_padded = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        n_tiles = n_padded // self.tile_size

        rows, cols = self._v2tp_find_grid(k_tiles)
        num_cores = rows * cols

        k_tiles_per_core = k_tiles // num_cores
        if k_tiles_per_core == 0:
            k_tiles_per_core = k_tiles
            num_cores = 1
        in0_block_w = self._v2tp_find_largest_divisor(k_tiles_per_core)
        per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=m_tiles,
            per_core_N=per_core_N,
            fused_activation=None,
        )
