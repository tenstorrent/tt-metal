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
   `sub_core_grids` becomes `(1,0) → (6,9) = 60 cores`, vs the previous
   `(1,0) → (3,9) + (5,0) → (6,9) = 50 cores`. +20% compute area.
"""
import math
import os
from pathlib import Path

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.qwen3_6_galaxy_v2.tt.model_config import (
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
        self.n_kv_heads = 4  # full_attention path
        self.head_dim = 256  # 5120 / 24 ≠ 256: heads are NOT dim/n_heads
        self.rope_dim = 64  # partial RoPE, rotary_factor=0.25
        self.partial_rotary_factor = 0.25
        self.rope_theta = 10_000_000  # MRoPE theta
        self.mrope_section = [11, 11, 10]  # text-mode collapses but kept
        self.rope_scaling_factor = 1.0
        self.orig_context_len = 262144
        self.intermediate_dim = 13_824  # SwiGLU intermediate
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
        self.intermediate_dim_per_tp = self.intermediate_dim // self.intermediate_dim_tp_factor  # 1728

        # --- Prefetcher carve-out reclamation ---
        # The 70B / qwen3-32B config carved cols 1-3 + 5-6 of the Tensix
        # grid (col 4 reserved as prefetcher-sender). With prefetcher off,
        # col 4 is unused — reclaim it for compute.
        _, _, _, self.pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)

        # Compute grid: cols 1-6 × rows 0-9 = 60 cores (vs 50 with carve-out).
        # col 0 is DRAM, col 7 is dispatch — leave reserved.
        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9)),
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
        self.n_local_heads = self.n_q_heads // self.cluster_shape[1]

        # --- Model config dict (memory configs by op key) ---
        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})
        self.model_config["USE_PREFETCHER"] = self.use_prefetcher
        self.model_config["GALAXY_NUM_LINKS"] = 1 if is_blackhole() else 4
        self.model_config["CCL_TOPOLOGY"] = ttnn.Topology.Linear if is_blackhole() else ttnn.Topology.Ring
        self.model_config["IS_QWEN36"] = True

        # NOTE: Per-op program configs (matmul-2d, sharded norm, etc.) are NOT
        # set up here. They are populated incrementally by V2-4 (full_attention),
        # V2-5 (DeltaNet), V2-6 (CCL persistent buffers). The 60-core grid
        # means several configs will need their `per_core_N` / grid-size
        # parameters re-tuned vs the 50-core 70B values.

        self.tokenizer = None  # populated when actually needed

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

    # Shape helper required by parent infrastructure.
    @property
    def cluster_shape_xy(self):
        return tuple(self.cluster_shape)

    def is_distributed_norm(self, mode):
        return True

    @property
    def ccl_topology(self):
        return self.model_config["CCL_TOPOLOGY"]

    def get_state_dict_prefix(self, module_name, layer_num):
        # Qwen3.6 HF keys: `model.language_model.layers.{i}.<module>.weight`
        return f"model.language_model.layers.{layer_num}.{module_name}"
