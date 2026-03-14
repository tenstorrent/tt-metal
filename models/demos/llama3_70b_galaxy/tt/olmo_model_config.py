# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Model Configuration for TT Galaxy.

Key differences from Qwen3-32B:
- n_q_heads = 40 (not 64), GQA ratio 5:1
- intermediate_size = 27648 (not 25600)
- No QK-norm
- YaRN RoPE (not linear)
- Hybrid sliding window attention (3 sliding + 1 full)
"""

import math
import os
import json
import ttnn
from pathlib import Path
from loguru import logger
from typing import Optional, List

from models.tt_transformers.tt.common import (
    freqs_to_rotation_matrix,
    get_base_model_name,
    get_out_subblock_w,
)
from models.common.utility_functions import nearest_32
from models.demos.llama3_70b_galaxy.tt.load_checkpoints import (
    load_hf_state_dict,
    standardize_hf_keys,
    convert_hf_to_meta,
)
from models.demos.llama3_70b_galaxy.tt.model_config import (
    TtModelArgs,
    CheckpointType,
    get_core_ranges,
    LlamaOptimizations,
    PREFETCHER_NOC1_GRID,
    num_to_coregrid,
)


class TtOlmoModelArgs(TtModelArgs):
    """OLMo-3.1-32B model configuration for TT Galaxy."""

    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
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
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=128 * 1024,
        optimizations=LlamaOptimizations.accuracy,
    ):
        if dummy_weights:
            raise ValueError("Dummy weights not supported for OLMo. Set HF_MODEL env var.")

        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.device_name = {0: "CPU", 1: "N150", 2: "N300", 8: "T3K", 32: "TG"}[self.num_devices]
        self.model_name = "OLMo-3.1-32B"
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = True  # Similar size/structure to 70B
        # OLMo3 model_type not recognized by HF AutoModel, load safetensors directly
        self.from_hf_url = False
        self.max_prefill_chunk_size = 40960
        self.use_prefetcher = False
        self.max_top_k = 32

        # ========================================================================
        # OLMo-specific: Different from Qwen3
        # ========================================================================
        self.dim = 5120
        self.n_q_heads = 40  # OLMo has 40 Q heads (not 64)
        self.n_kv_heads = 8
        self.intermediate_dim = 27648  # OLMo has 27648 (not 25600)

        self.qk_norm = True  # OLMo3 has QK-norm (q_norm: [5120], k_norm: [1024])
        self.is_qwen = False
        self.is_olmo = True
        self.unfuse_res_add = True

        # ========================================================================
        # YaRN RoPE parameters (OLMo uses YaRN, not linear scaling)
        # ========================================================================
        self.rope_type = "yarn"
        self.yarn_attention_factor = 1.2079441541679836
        self.yarn_beta_fast = 32.0
        self.yarn_beta_slow = 1.0

        # ========================================================================
        # Sliding window attention (hybrid: 3 sliding + 1 full)
        # ========================================================================
        self.sliding_window = 4096
        self.sliding_window_pattern = 4  # Every 4th layer is full attention

        # Tensor parallel factors
        self.dim_tp_factor = 4
        self.intermediate_dim_tp_factor = 8

        # Derived dimensions
        self.dim_padded_24_cores = 6144
        self.dim_per_tp = self.dim // self.dim_tp_factor  # 5120 // 4 = 1280
        self.intermediate_dim_per_tp = self.intermediate_dim // self.intermediate_dim_tp_factor  # 27648 // 8 = 3456
        self.intermediate_dim_per_tp_padded_24_cores = 3840

        # Disable prefetcher for OLMo decode debugging
        # Note: This requires OLMo-specific MLP decode path that doesn't use double_matmul
        # if self.num_devices == 32:
        #     self.use_prefetcher = True

        # Prefetcher setup
        _, _, _, self.pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)

        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )
        self.sub_core_grid_topk = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ]
        )
        self.start_core = ttnn.CoreCoord(1, 0)

        # Load from HF_MODEL environment variable
        HF_MODEL = os.getenv("HF_MODEL")
        if not HF_MODEL:
            raise ValueError(
                "HF_MODEL environment variable not set. "
                "Set it to: export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think"
            )

        # Handle HuggingFace cache directory structure (models--org--name/snapshots/hash/...)
        import glob

        base_path = os.path.expanduser(HF_MODEL)
        if os.path.exists(os.path.join(base_path, "snapshots")):
            snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
            if snapshot_dirs:
                base_path = snapshot_dirs[0]
                logger.info(f"Using snapshot directory: {base_path}")

        self.CKPT_DIR = base_path
        self.TOKENIZER_PATH = base_path
        self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
        if not self.CACHE_PATH:
            self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
        else:
            self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
        self.model_name = HF_MODEL

        logger.info(f"OLMo checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"OLMo tokenizer path: {self.TOKENIZER_PATH}")
        logger.info(f"OLMo cache directory: {self.CACHE_PATH}")

        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)
        self.tokenizer_path = self.TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        if "instruct" in self.CKPT_DIR.lower() or "think" in self.CKPT_DIR.lower():
            self.instruct = True

        # Load model params from HF config
        self.checkpoint_type = CheckpointType.HuggingFace
        self._set_hf_params(self.CKPT_DIR)

        if callable(optimizations):
            self.optimizations = optimizations(self.model_name)
        else:
            self.optimizations = optimizations

        self.dummy_weights = dummy_weights
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        # di/dt workaround
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
        if not self.di_dt_workaround:
            logger.info("Disabling di/dt workaround")

        # Galaxy TG setup
        self.TG = self.num_devices == 32
        self.is_galaxy = self.TG  # Alias for compatibility with base class
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        # Memory configs
        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        # Precompute RoPE frequencies (YaRN)
        self.cos, self.sin = self._precompute_yarn_freqs()
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)

        self.tokenizer = None if dummy_weights else self.create_tokenizer()

        # Galaxy configuration
        self.cluster_shape = list(mesh_device.shape)

        if self.num_devices != 32:
            raise ValueError(
                f"Unsupported number of devices: {self.num_devices}. Only 32 devices (Galaxy) are supported."
            )

        self.model_config["GALAXY_NUM_LINKS"] = 4
        self.model_config["CCL_TOPOLOGY"] = ttnn.Topology.Ring

        if mesh_device is not None:
            self.n_local_heads = self.n_heads // self.cluster_shape[1]  # 40 // 8 = 5

            grid = mesh_device.compute_with_storage_grid_size()
            self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

            self.dram_weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                    )
                }
            )

            # Setup TG-specific configs for MLP and Attention
            self._setup_tg_configs()

    def _setup_tg_configs(self):
        """Setup Galaxy TG-specific model configs for MLP and Attention."""
        # OLMo-specific dimensions
        # dim = 5120, dim_per_tp = 1280 (5120 // 4)
        # intermediate = 27648, intermediate_per_tp = 3456 (27648 // 8)
        # padded intermediate_per_tp for 24-core = 3840 (same as Qwen due to rounding)

        # ==== Compute Kernel Configs ====
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

        # CCL configs
        self.num_reduce_scatter_links = 1
        self.num_all_gather_links = 2  # Galaxy configuration
        self.ccl_dtype = ttnn.bfloat8_b
        self.is_multichip = self.num_devices > 1

        # KV cache sharding config
        # Formula: (tile_size * 8 * 8) / (n_kv_heads // cluster_shape[1])
        # For OLMo: (32 * 8 * 8) / (8 // 4) = 2048 / 2 = 1024
        self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])

        # QKV size
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)

        RING_SIZE = 24
        ring_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in PREFETCHER_NOC1_GRID
            ]
        )
        pf_mm_out_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in self.pf_receiver_cores_list
            ]
        )

        self.model_config["USE_PREFETCHER"] = self.use_prefetcher

        # All Gather Matmul for Dense Out (DO)
        self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
            self.ccl_topology() == ttnn.Topology.Ring
            and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
            and self.num_devices > 1
        )

        # Galaxy num links
        self.model_config["GALAXY_NUM_LINKS"] = 1

        # ==== MLP Decode Configs ====
        mlp_core_grid = ttnn.CoreGrid(y=1, x=8)
        self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            (
                32,
                self.dim // mlp_core_grid.num_cores,
            ),
            mlp_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
            m=self.tile_padded_batch_rows,
            k=self.dim,
            n=self.hidden_dim // self.cluster_shape[1],
            num_cores=mlp_core_grid.num_cores,
        )

        mlp2_core_grid = ttnn.CoreGrid(y=1, x=8)
        self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            (
                32,
                self.hidden_dim // self.cluster_shape[1] // mlp2_core_grid.num_cores,
            ),
            mlp2_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
            m=self.tile_padded_batch_rows,
            k=self.hidden_dim // self.cluster_shape[1],
            n=self.dim,
            num_cores=mlp2_core_grid.num_cores,
        )

        # ==== Attention Input Configs ====
        # For OLMo: dim=5120, lm_head_num_rows=8 (computed later, use 8 for now)
        lm_head_num_rows = 8
        attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
        attn_input_sub_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, 32, self.sub_core_grids, row_wise=True
        )
        # OLMo: dim=5120, 5120 // (8*8) // 4 = 20, nearest_32 = 32
        self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, nearest_32(self.dim // (8 * lm_head_num_rows) // 4)),
            core_grid=attn_input_sub_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Attention input ring memory config
        # OLMo dim_padded_24_cores = 1536 (padded), 1536 / 4 / 24 = 16, round to 32
        self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, max(32, self.dim_padded_24_cores // 4 // RING_SIZE)),
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ==== RMSNorm Program Configs ====
        self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
        self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)

        # ==== MLP Ring Configs for Prefetcher ====
        # W1W3: K=1280 (dim/4), N=3840 (padded intermediate/8)
        self.model_config["W1W3_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.dim // 4,  # 1280
            n=self.intermediate_dim_per_tp_padded_24_cores,  # 3840
        )

        # W2: K=3456 (intermediate/8), N=1536 (padded dim/4)
        self.model_config["W2_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=self.intermediate_dim_per_tp,  # 3456
            n=self.dim_padded_24_cores // 4,  # 1536
        )

        self.model_config["FF1_3_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,  # B
            32,  # M
            self.dim // 4,  # K = 1280
            self.intermediate_dim_per_tp_padded_24_cores,  # N = 3840
            RING_SIZE,
        )

        # OLMo decode without prefetcher: num_global_cb_receivers=1
        self.model_config["FF1_3_TG_RING_PROGCFG_NO_PREFETCH"] = self.matmul_1d_ring_config(
            1,  # B
            32,  # M
            self.dim // 4,  # K = 1280
            self.intermediate_dim_per_tp_padded_24_cores,  # N = 3840
            RING_SIZE,
            prefetch=False,
        )

        self.model_config["FF2_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            self.intermediate_dim_per_tp,  # K = 3456
            self.dim_padded_24_cores // 4,  # N = 1536
            RING_SIZE,
        )

        # OLMo decode without prefetcher
        self.model_config["FF2_TG_RING_PROGCFG_NO_PREFETCH"] = self.matmul_1d_ring_config(
            1,
            32,
            self.intermediate_dim_per_tp,  # K = 3456
            self.dim_padded_24_cores // 4,  # N = 1536
            RING_SIZE,
            prefetch=False,
        )

        self.model_config["SHARDED_FF12_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_padded_24_cores // 4 // RING_SIZE),
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.intermediate_dim_per_tp_padded_24_cores // RING_SIZE),
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["SHARDED_FF12_PRE_MUL_RING_REDUCE_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.intermediate_dim_per_tp_padded_24_cores // 30),
            core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 30, self.sub_core_grids, row_wise=True
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        mul_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, 28, self.sub_core_grids, row_wise=True
        )
        self.model_config["MUL_IN_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.intermediate_dim_per_tp_padded_24_cores // 28),
            core_grid=mul_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["FF2_IN_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.intermediate_dim_per_tp_padded_24_cores // RING_SIZE),
            core_grid=ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["FF2_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_padded_24_cores // 4 // RING_SIZE),
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # OLMo-specific FF2 output config for decode: dim_per_tp=1280, not padded 1536
        # 1280 / 10 cores = 128 per shard (tile-aligned)
        OLMO_OUT_RING_SIZE = 10
        olmo_out_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, OLMO_OUT_RING_SIZE, self.sub_core_grids, row_wise=True
        )
        self.model_config["FF2_OUT_RING_MEMCFG_OLMO"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_per_tp // OLMO_OUT_RING_SIZE),  # (32, 128)
            core_grid=olmo_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ==== Reduce Scatter Configs ====
        PACKET_WORKER_CRS = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
            ]
        )
        self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 512),
            core_grid=PACKET_WORKER_CRS,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), 30, self.sub_core_grids, row_wise=True
        )
        # Note: OLMo decode uses host-side reduce_scatter due to L1 constraints
        # This config is kept for compatibility but may not be used for OLMo decode
        self.model_config["REDUCE_SCATTER_OUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                FF1_CRS_RS_OUT,
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # OLMo-specific reduce_scatter output config for decode:
        # intermediate_dim_per_tp_padded=3840, after reduce_scatter (8 devices): 3840/8=480
        # 480 / 15 cores = 32 per core (tile-aligned)
        OLMO_RS_OUT_CORES = 15
        olmo_rs_out_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), OLMO_RS_OUT_CORES, self.sub_core_grids, row_wise=True
        )
        self.model_config["REDUCE_SCATTER_OUT_MEMCFG_OLMO"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                olmo_rs_out_core_range_set,
                [32, 32],  # 15 cores × 32 = 480 total width
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # ==== Decode Residual Config ====
        # OLMo: dim//4 = 1280. For tile-aligned shards:
        # 1280 / 10 cores = 128 per core (tile aligned)
        # Using 5 rows × 2 cols = 10 cores, starting at (1, 0) like base class
        num_cores_ln = 10
        core_grid_ln, grid_offset = (5, 2), ttnn.CoreCoord(1, 0)
        core_range = ttnn.CoreRange(
            grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
        )
        self.model_config["DECODE_RESIDUAL_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, self.dim // 4 // num_cores_ln),  # (1, 1, 32, 128) - tile aligned
            core_grid=ttnn.CoreRangeSet({core_range}),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        # ==== Decode QKV Head Creation Configs ====
        # For llama_rs_create_heads: width-sharded input, height-sharded output
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
        self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_core_grids,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # ==== Prefill MLP Configs ====
        def w1_w3_prg_config(seq_len, use_interleaved=False):
            if seq_len == 128:
                return self.matmul_1d_config(
                    128,
                    self.dim // 4,  # K = 1280
                    self.intermediate_dim_per_tp,  # N = 3456
                    grid=ttnn.CoreGrid(x=7, y=10),
                    overwrite_per_core_k=4,
                )
            if seq_len < 4096:
                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 10),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),
                    per_core_N=math.ceil(self.intermediate_dim_per_tp / 32 / 5),  # 3456 / 32 / 5 = 22
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 1024,
                )
            return None

        self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = w1_w3_prg_config

        def w2_prg_config(seq_len):
            if seq_len == 128:
                return self.matmul_1d_config(
                    128,
                    self.intermediate_dim_per_tp,  # K = 3456
                    self.dim // 4,  # N = 1280
                    grid=ttnn.CoreGrid(x=7, y=10),
                    overwrite_per_core_k=2,
                )
            if seq_len < 4096:
                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 10),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),
                    per_core_N=math.ceil(self.dim // 4 / 32 / 5),  # 1280 / 32 / 5 = 8
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 1024,
                )
            return None

        self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = w2_prg_config

        def prefill_ff1_ff3_minimal_matmul_config(seq_len):
            if seq_len <= 4096:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                )
            elif seq_len <= 8192:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=8,
                    subblock_w=1,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                )
            else:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=8,
                    subblock_w=1,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                )

        self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"] = prefill_ff1_ff3_minimal_matmul_config

        def prefill_ff2_minimal_matmul_config(seq_len):
            if seq_len <= 4096:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                )
            elif seq_len <= 16384:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=8,
                    subblock_w=1,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                )
            else:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=16,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=8,
                    subblock_w=1,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                )

        self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"] = prefill_ff2_minimal_matmul_config

        # ==== Attention Configs ====
        # OLMo Q weight padding for decode: 5 local heads → 8 local heads
        # This is required for fused RoPE which needs num_heads * head_dim = 1024
        # Original: qkv_size = 128 * (2*8 + 40) = 7168, per device = 896
        # Padded: qkv_size = 128 * (2*8 + 64) = 10240, per device = 1280 (same as Llama!)

        # QKV memory config for ring topology
        # Shape: (dim // 4, qkv_size_padded // 8) where qkv_size_padded needs 24-core alignment
        # Use padded Q heads (8 instead of 5) for decode compatibility
        padded_n_heads = 64  # 8 heads per device × 8 devices (padded from 40)
        qkv_size_decode = self.head_dim * (2 * self.n_kv_heads + padded_n_heads)  # 128 * 80 = 10240
        qkv_size_padded = nearest_32(qkv_size_decode)  # 10240 (already aligned)
        qkv_size_per_device = qkv_size_padded // 8  # 1280
        qkv_size_per_device_padded = ((qkv_size_per_device + 767) // 768) * 768  # 1280 → 1536

        # Unpadded QKV/WO sizes for prefill (original 5 Q heads, no padding)
        qkv_size_prefill = self.head_dim * (2 * self.n_kv_heads + self.n_heads)  # 128 * 56 = 7168
        qkv_size_per_device_prefill = qkv_size_prefill // 8  # 896
        wo_k_prefill = self.n_heads * self.head_dim // 8  # 40 * 128 / 8 = 640

        qkv_shape_ring = (self.dim // 4, qkv_size_per_device_padded)  # (1280, 896)
        self.model_config["SHARDED_QKV_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=qkv_shape_ring[0],
            n=qkv_shape_ring[1],
        )

        qkv_out_shard_shape_ring = (32, qkv_size_per_device_padded // RING_SIZE)  # (32, 38) -> pad to 64
        qkv_out_width = max(32, nearest_32(qkv_size_per_device_padded // RING_SIZE))  # At least 32 for tile
        self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, qkv_out_width),
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["XQKV_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1,
            32,
            self.dim // 4,  # K = 1280
            qkv_size_per_device_padded,  # N = 896
            RING_SIZE,
            prefetch=self.use_prefetcher,
            untilize_out=True,
        )

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

        # WO configs
        # OLMo decode: Use 8 padded Q heads (1024 per device) from fused RoPE
        # Original: 5 local Q heads * 128 = 640 per device
        # Padded: 8 Q heads * 128 = 1024 per device
        # Use 8 cores for WO input sharding: 1024 / 8 = 128 per shard (tile-aligned!)
        WO_RING_SIZE = 8  # Use 8 cores instead of 24 for 1024 per device
        wo_k_decode = 8 * self.head_dim  # 8 padded heads * 128 = 1024 per device
        wo_n_padded = self.dim_padded_24_cores // 4  # 1280 -> 1536 padded

        # Create 8-core grid for WO input using valid sub_core_grids
        # sub_core_grids excludes cols 0, 4, 7. Use helper to get 8 valid cores.
        wo_input_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, WO_RING_SIZE, self.sub_core_grids, row_wise=True
        )

        self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, wo_k_decode // WO_RING_SIZE),  # (32, 1024/8) = (32, 128) tile-aligned!
            core_grid=wo_input_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        wo_shape_ring = (wo_k_decode, wo_n_padded)  # (1024, 1536)
        self.model_config["SHARDED_WO_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
            k=wo_shape_ring[0],
            n=wo_shape_ring[1],
        )

        wo_out_shard_shape_ring = (32, wo_n_padded // RING_SIZE)  # (32, 64)
        self.model_config["SHARDED_WO_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=wo_out_shard_shape_ring,
            core_grid=pf_mm_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # OLMo-specific WO output config for decode: dim_per_tp=1280, not padded 1536
        # 1280 / 10 cores = 128 per shard (tile-aligned)
        self.model_config["SHARDED_WO_OUT_RING_MEMCFG_OLMO"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim_per_tp // OLMO_OUT_RING_SIZE),  # (32, 128)
            core_grid=olmo_out_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["WO_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1, 32, wo_shape_ring[0], wo_shape_ring[1], RING_SIZE
        )

        # SDPA program config
        # Use (7, 4) to avoid dispatch core column
        def sdpa_progcfg(seq_len):
            if seq_len <= 256:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 4],
                    q_chunk_size=64,
                    k_chunk_size=64,
                )
            elif seq_len <= 2048:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 4],
                    q_chunk_size=128,
                    k_chunk_size=128,
                )
            else:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 4],
                    q_chunk_size=256,
                    k_chunk_size=256,
                )

        self.model_config["SDPA_PROGCFG"] = sdpa_progcfg

        # ==== Decode SDPA Configs ====
        # Paged SDPA for decode mode
        self.model_config["PAGED_SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(7, 6),  # Use 7 instead of 8 to avoid dispatch core column
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 42, self.sub_core_grids, row_wise=True
            ),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        # Non-paged SDPA for decode mode
        self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(7, 4),  # Use 7 instead of 8 to avoid dispatch core column
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 28, self.sub_core_grids, row_wise=True
            ),
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )

        # SDPA compute kernel config
        self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # SDPA output memory config (sharded by batch)
        # OLMo has 5 local Q heads, pad to tile size (32)
        self.model_config[
            "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
        ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
            shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # (32, 128) padded
            core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, batch_size_per_device_group, self.sub_core_grids, row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # ==== Decode WO Config ====
        self.model_config["WO_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
            1, 32, wo_shape_ring[0], wo_shape_ring[1], RING_SIZE, prefetch=self.use_prefetcher
        )

        # QKV prefill config (uses unpadded QKV weights: 5 Q heads, N=896 per device)
        # 896 / 32 = 28 tiles. Use 4 cores for N (28/4=7 tiles/core)
        qkv_pf_n_tiles = qkv_size_per_device_prefill // 32  # 28 tiles
        qkv_pf_n_cores = 4  # 28 / 4 = 7 evenly
        qkv_pf_per_core_n = qkv_pf_n_tiles // qkv_pf_n_cores  # 7
        self.model_config["XQKV_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len,
                self.dim // 4,
                qkv_size_per_device_prefill,
                grid=ttnn.CoreGrid(x=4, y=10),
                overwrite_per_core_k=8,
            )
            if seq_len == 128
            else (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(qkv_pf_n_cores, 10),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),
                    per_core_N=qkv_pf_per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 2048,
                )
            )
        )

        # WO prefill config (uses unpadded WO: K=640, N=1280)
        # 640/32 = 20 tiles for K, 1280/32 = 40 tiles for N
        wo_n_tiles = self.dim // 4 // 32  # 40 tiles
        wo_n_cores = 5  # 40 / 5 = 8 evenly
        wo_per_core_n = wo_n_tiles // wo_n_cores  # 8
        self.model_config["WO_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len, wo_k_prefill, self.dim // 4, grid=ttnn.CoreGrid(x=4, y=10), overwrite_per_core_k=4
            )
            if seq_len == 128
            else (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(wo_n_cores, 10),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=2,  # 8 % 2 == 0 ✓
                    per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),
                    per_core_N=wo_per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 2048,
                )
            )
        )

        # WO prefill minimal config (for seq_len >= 4096)
        def prefill_wo_minimal_matmul_config(seq_len):
            if seq_len <= 4096:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=8,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
                )
            else:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                )

        self.model_config["WO_PREFILL_MINIMAL_PROGCFG"] = prefill_wo_minimal_matmul_config

        # ==== LM Head Configs (required for model initialization) ====
        # Find lm_head_num_rows such that dim % (32 * num_rows * 4) == 0
        # For OLMo dim=5120: 5120 % (32 * 8 * 4) = 5120 % 1024 = 0
        lm_head_num_rows = 8
        while self.dim % (32 * lm_head_num_rows * 4) != 0:
            lm_head_num_rows -= 1
        assert lm_head_num_rows > 0, f"Could not find lm_head_num_rows for dim={self.dim}"
        self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)

        self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            (
                self.tile_padded_batch_rows,
                nearest_32((self.dim // 4) // self.lm_head_core_grid.num_cores),
            ),
            self.lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # RMS NORM config for LM head
        self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(self.lm_head_core_grid)

        # ==== LM Head Ring Configs ====
        LM_HEAD_RING_SIZE = 24
        # OLMo: vocab=100278, padded for ring matmul
        # N must be divisible by LM_HEAD_RING_SIZE * TILE_SIZE = 24 * 32 = 768
        RING_TILE_ALIGN = LM_HEAD_RING_SIZE * 32  # 768
        padded_vocab_per_device = (
            (self.padded_vocab_size // 8 + RING_TILE_ALIGN - 1) // RING_TILE_ALIGN
        ) * RING_TILE_ALIGN
        self.lm_head_shape = (self.dim // 4, padded_vocab_per_device)  # (1280, 13056)

        # Import grids from parent module
        from models.demos.llama3_70b_galaxy.tt.model_config import (
            LM_HEAD_32_GRID,
            LM_HEAD_16_GRID,
            LM_HEAD_INPUT_GRID,
            LM_HEAD_OUTPUT_GRID,
        )

        lm_head_ring_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_32_GRID]
        )
        lm_head_ring_core_input_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_INPUT_GRID]
        )
        lm_head_ring_core_output_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_OUTPUT_GRID]
        )
        lm_head_ring_16_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in LM_HEAD_16_GRID]
        )

        # Padded shape for ring matmul - already properly aligned via RING_TILE_ALIGN above
        lm_head_out_padded = padded_vocab_per_device  # Already multiple of 768

        lm_head_input_n_cores = 10
        lm_head_input_shard_w = self.dim_per_tp // lm_head_input_n_cores  # 1280 // 10 = 128 (tile-aligned)
        lm_head_input_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, lm_head_input_n_cores, self.sub_core_grids, row_wise=True
        )
        self.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, lm_head_input_shard_w),
            core_grid=lm_head_input_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_LM_HEAD_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.lm_head_shape[0] // 16),  # (32, 80) for OLMo
            core_grid=lm_head_ring_16_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, max(32, lm_head_out_padded // LM_HEAD_RING_SIZE)),
            core_grid=lm_head_ring_core_output_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_OUT_RING_RESHARD_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, max(32, lm_head_out_padded // LM_HEAD_RING_SIZE)),
            core_grid=lm_head_ring_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["LM_HEAD_TG_RING_PROGCFG"] = self.matmul_1d_ring_lm_head_config(
            1,
            32,
            self.dim // 4,  # K = 1280
            lm_head_out_padded,  # N padded for ring
            LM_HEAD_RING_SIZE,
            prefetch=False,
        )
        # Use MinimalMatmulConfig for LM head prefill since it avoids signature issues
        self.model_config["LM_HEAD_PREFILL_PROGCFG"] = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
        )

        # ==== Gather Users Config (required for decode setup) ====
        self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
            shape=(32, 128),  # mesh_cols = 4
            core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 32, self.sub_core_grids, row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def is_distributed_norm(self, mode):
        """Determine if distributed RMSNorm should be used.

        For 2D mesh (Galaxy TG), always use distributed norm.
        For 1D mesh with large hidden dim in prefill, use distributed norm.
        """
        if not self.is_multichip:
            return False
        if all([dim > 1 for dim in list(self.mesh_device.shape)]):  # 2D grid (Galaxy TG)
            return True
        elif self.dim >= 8192 and mode == "prefill":  # Large hidden dim needs distributed
            return True
        return False

    def _precompute_yarn_freqs(self):
        """Precompute YaRN RoPE frequencies for OLMo."""
        import torch

        dim = self.head_dim
        end = self.max_seq_len * 2
        base = self.rope_theta

        # Base frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # YaRN scaling
        inv_freq = inv_freq / self.rope_scaling_factor

        # Compute cos/sin
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        return cos, sin

    def get_layer_type(self, layer_id: int) -> str:
        """Get attention type for a layer (sliding or full)."""
        if hasattr(self, "layer_types_list") and self.layer_types_list:
            return self.layer_types_list[layer_id]
        # Fallback to pattern-based
        if (layer_id + 1) % self.sliding_window_pattern == 0:
            return "full_attention"
        return "sliding_attention"

    def get_sliding_window_size(self, layer_id: int) -> Optional[int]:
        """Get sliding window size for a layer (None for full attention)."""
        if self.get_layer_type(layer_id) == "sliding_attention":
            return self.sliding_window
        return None

    def get_all_layer_types(self) -> List[str]:
        """Get list of attention types for all layers."""
        return [self.get_layer_type(i) for i in range(self.n_layers)]

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors."""
        dram_cores = 12
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def dram_matmul_config(self, m: int, k: int, n: int, num_cores=None):
        """Create DRAM matmul config."""
        if num_cores is None:
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
            assert (
                k % (self.tile_size * num_cores) == 0
            ), f"k must be divisible by tile_size * num_cores: {k} % {self.tile_size * num_cores} != 0"
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=math.ceil(k / (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=None,
        )

    def dram_shard_core_grid_for_k_and_n(self, k: int, n: int):
        """Get core grid for DRAM sharding."""
        rows, cols = self.find_grid_k_n(k // self.tile_size, n // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid_k_n(self, K, N):
        """Find grid for K and N tile counts."""
        max_rows = 4
        max_cols = 8
        max_cores = max_rows * max_cols

        possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
        possible_cores.sort(reverse=True)

        for cores in possible_cores:
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        raise AssertionError(f"Cannot find grid configuration for K={K}, N={N} tiles.")

    def matmul_1d_ring_config(
        self,
        B,
        M,
        K,
        N,
        num_cores,
        prefetch=True,
        untilize_out=False,
    ):
        """Create 1D ring matmul config."""
        M *= B  # Fuse batch always enabled

        in0_block_h = M // ttnn.TILE_SIZE
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // num_cores // ttnn.TILE_SIZE

        num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
        num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
        num_blocks_total = num_blocks_y * num_blocks_x

        if num_blocks_total != num_cores:
            assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

        out_subblock_h = 1
        out_subblock_w = 8
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        # Hardware constraint: out_subblock_w * out_subblock_h <= 4
        while out_subblock_w * out_subblock_h > 4:
            out_subblock_w -= 1
            # Ensure it still divides evenly
            while out_block_w % out_subblock_w != 0 and out_subblock_w > 1:
                out_subblock_w -= 1

        hop_grid = [(3, 6)] if prefetch else []
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )

        # Without prefetch, limit to 7 columns (0-6) to avoid dispatch core on column 7
        if prefetch:
            grid = num_to_coregrid(num_cores)
        else:
            # Use 7-column grid: for 24 cores, use 6x4=24 or adjust
            if num_cores == 24:
                grid = ttnn.CoreGrid(y=4, x=6)  # 6 cols x 4 rows = 24
            elif num_cores == 8:
                grid = ttnn.CoreGrid(y=2, x=4)  # 4 cols x 2 rows = 8
            elif num_cores % 7 == 0:
                grid = ttnn.CoreGrid(y=num_cores // 7, x=7)
            elif num_cores % 6 == 0:
                grid = ttnn.CoreGrid(y=num_cores // 6, x=6)
            else:
                grid = num_to_coregrid(num_cores)  # Fallback

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=hop_core_range_set,
            num_global_cb_receivers=2 if prefetch else 1,
            untilize_out=untilize_out,
        )

        return program_config

    def matmul_1d_config(
        self,
        m,
        k,
        n,
        grid,
        overwrite_per_core_k=None,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
        fused_activation=None,
        fuse_batch=True,
    ):
        """Create 1D matmul config."""
        per_core_M = math.ceil(m / (self.tile_size * grid.y))
        per_core_N = math.ceil(n / (self.tile_size * grid.x))
        per_core_K = math.ceil(k / (self.tile_size * grid.x))
        if overwrite_per_core_k is not None:
            per_core_K = overwrite_per_core_k

        out_subblock_h = 1
        out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)
        if overwrite_subblock_w is not None:
            out_subblock_w = overwrite_subblock_w
        if overwrite_subblock_h is not None:
            out_subblock_h = overwrite_subblock_h

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=per_core_K,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    def _set_hf_params(self, checkpoint_dir):
        """Load parameters from HuggingFace config."""
        # OLMo3 uses a custom model type, so load config.json directly
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            # Try snapshots directory structure
            import glob

            config_files = glob.glob(os.path.join(checkpoint_dir, "snapshots", "*", "config.json"))
            if config_files:
                config_path = config_files[0]

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded OLMo config from {config_path}")
        else:
            raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

        self._set_params_from_dict(config, is_hf=True)

    def _set_params_from_dict(self, params, is_hf=False):
        """Set model parameters from config dict."""
        self.dim = params.get("hidden_size", self.dim)
        self.n_heads = params.get("num_attention_heads", self.n_q_heads)
        self.n_kv_heads = params.get("num_key_value_heads", self.n_kv_heads)
        self.n_layers = params.get("num_hidden_layers", 64)
        self.full_model_n_layers = self.n_layers
        self.norm_eps = params.get("rms_norm_eps", 1e-6)
        self.vocab_size = params.get("vocab_size", 100278)
        # Pad vocab to multiple of (num_vocab_devices * tile_size) = 8*32 = 256
        # so each device's shard (padded_vocab_size // 8) is tile-aligned (multiple of 32).
        # Without this, tile-padding inside each shard shifts vocab token positions after gather,
        # causing ~87% of logit comparisons to compare the wrong tokens (PCC ~0.128).
        # e.g. 100278 -> nearest_32=100288 -> per_device=12536 (not mult of 32, tile-pads to 12544)
        #      after gather: 8*12544=100352 physical but vocab tokens are at wrong indices.
        # Fix: 100278 -> nearest_256=100352 -> per_device=12544 (mult of 32, no extra padding).
        _vocab_shard_tile_align = 8 * 32  # 8 vocab-sharding devices * 32 tile size = 256
        self.padded_vocab_size = (
            (self.vocab_size + _vocab_shard_tile_align - 1) // _vocab_shard_tile_align
        ) * _vocab_shard_tile_align
        self.head_dim = params.get("head_dim", self.dim // self.n_heads)
        self.max_context_len = params.get("max_position_embeddings", 65536)

        # MLP dimensions
        if "intermediate_size" in params:
            self.hidden_dim = params["intermediate_size"]
            self.intermediate_dim = params["intermediate_size"]
            self.ffn_dim_multiplier = None
            self.multiple_of = None
        else:
            self.hidden_dim = self.intermediate_dim

        self.unpadded_hidden_dim = self.hidden_dim

        # Model name
        if "_name_or_path" in params:
            normalized_path = os.path.normpath(params["_name_or_path"])
            if "snapshots" in normalized_path:
                full_model_name = normalized_path.split(os.path.sep)[-3]
                self.model_name = full_model_name.split("--")[-1]
            else:
                self.model_name = os.path.basename(normalized_path)
            logger.info(f"OLMo model name: {self.model_name}")

        # RoPE params (YaRN)
        self.rope_theta = params.get("rope_theta", 500000.0)
        rope_scaling = params.get("rope_scaling", {})
        if rope_scaling:
            self.rope_scaling_factor = rope_scaling.get("factor", 8.0)
            self.orig_context_len = rope_scaling.get("original_max_position_embeddings", 8192)
            self.yarn_attention_factor = rope_scaling.get("attention_factor", 1.2079441541679836)
            self.yarn_beta_fast = rope_scaling.get("beta_fast", 32.0)
            self.yarn_beta_slow = rope_scaling.get("beta_slow", 1.0)
        else:
            self.rope_scaling_factor = 8.0
            self.orig_context_len = 8192
            self.yarn_attention_factor = 1.2079441541679836
            self.yarn_beta_fast = 32.0
            self.yarn_beta_slow = 1.0

        # Alias for YaRN RoPE computation
        self.original_max_position_embeddings = self.orig_context_len

        # Sliding window
        self.sliding_window = params.get("sliding_window", 4096)

        # Layer types from config (OLMo provides explicit list)
        if "layer_types" in params:
            self.layer_types_list = params["layer_types"]
            logger.info(f"Loaded {len(self.layer_types_list)} layer types from config")

    @property
    def use_scaled_rope(self):
        return True  # OLMo always uses YaRN scaling

    @property
    def base_model_name(self):
        return get_base_model_name(self.model_name)

    def __repr__(self):
        return f"""TtOlmoModelArgs(
    model_name={self.model_name}
    dim={self.dim}
    n_layers={self.n_layers}
    n_heads={self.n_heads}
    n_kv_heads={self.n_kv_heads}
    vocab_size={self.vocab_size}
    intermediate_size={self.intermediate_dim}
    head_dim={self.head_dim}
    rope_theta={self.rope_theta}
    rope_type={self.rope_type}
    rope_scaling_factor={self.rope_scaling_factor}
    yarn_attention_factor={self.yarn_attention_factor}
    sliding_window={self.sliding_window}
    qk_norm={self.qk_norm}
)"""

    def get_state_dict_prefix(self, module_name, layer_num):
        """Get state dict prefix for weight loading."""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "TtLlamaMLP": "feed_forward",
            "TtLlamaAttention": "attention",
            "TtTransformerBlock": "",
            "": "",
        }
        return layer_prefix + module_map.get(module_name, "")

    def weight_cache_path(self, dtype):
        """Get weight cache path for given dtype."""
        if self.instruct:
            return (
                self.model_cache_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def load_state_dict(self):
        """Load state dict from HuggingFace checkpoint."""
        assert self.checkpoint_type == CheckpointType.HuggingFace

        if self.from_hf_url:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.CKPT_DIR,
                trust_remote_code=True,
            )
            state_dict = model.state_dict()
        else:
            state_dict = load_hf_state_dict(self.CKPT_DIR)

        state_dict = standardize_hf_keys(state_dict)
        state_dict = convert_hf_to_meta(state_dict, self.head_dim)

        # OLMo QK-norm weights: q_norm [5120], k_norm [1024]
        # These are kept and applied before head splitting (unlike Qwen3's per-head norm)

        return state_dict

    def create_tokenizer(self):
        """Create tokenizer for OLMo.

        OLMo3 uses GPT2Tokenizer. We use from_pretrained with use_fast=True
        to bypass the AutoConfig model_type check.
        """
        from transformers import GPT2Tokenizer

        # Find the actual tokenizer path in snapshots
        import glob

        base_path = os.path.expanduser(self.TOKENIZER_PATH)
        if os.path.exists(os.path.join(base_path, "snapshots")):
            snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
            if snapshot_dirs:
                base_path = snapshot_dirs[0]

        return GPT2Tokenizer.from_pretrained(base_path)
