# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import json
import ttnn
from pathlib import Path
from loguru import logger
import torch
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.llama3.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix
from typing import Tuple
from models.utility_functions import nearest_32
from pathlib import Path
from tqdm import tqdm


def calculate_hidden_dim(dim, ffn_dim_multiplier, multiple_of):
    """Helper function based on logic used in reference model:
    https://github.com/meta-llama/llama-models/blob/e4a6ed52a142bb9b5106dcbf48e41f97f8e7378e/models/llama3/reference_impl/model.py#L227C7-L231C83
    """
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


# TODO Add other default params for models in the family
DEFAULT_LLAMA3_2_3B_PARAMS = {
    "dim": 3072,
    "ffn_dim_multiplier": 1.0,
    "multiple_of": 256,
    "n_heads": 24,
    "n_kv_heads": 8,
    "n_layers": 28,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256,
}

DEFAULT_LLAMA3_1_8B_PARAMS = {
    "dim": 4096,
    "ffn_dim_multiplier": 1.3,
    "multiple_of": 1024,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 32,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256,
}


class TtModelArgs:
    paged_attention_config = None

    # TODO Update these params. In init we update the max_seq_len to 32k if it's a single device
    max_batch_size = 1
    # Context length for Llama models (if single device, reduce to 32k in init)
    max_seq_len = 8192 * 16  # 128k
    kv_seq_len = 8192 * 16  # 128k
    sliding_window = 8192 * 16  # 128k

    tile_size = 32

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
        "DEC_SKIP_OUTPUT",
        "OUTPUT_MM",
    )

    def __init__(self, mesh_device, instruct=False, dummy_weights=False, max_batch_size=1):
        # Add this near the top of the class, with other class attributes
        self.num_devices = mesh_device.get_num_devices()
        device = mesh_device.get_devices()[0]
        device_name = {1: "N150", 2: "N300", 8: "T3K", 32: "TG"}[self.num_devices]

        # A single device cannot fit the full 128k context length
        if self.num_devices == 1:
            self.max_seq_len = 8192 * 4  # 32k
            self.kv_seq_len = 8192 * 4  # 32k
            self.sliding_window = 8192 * 4  # 32k

        # Default folder location for weights and cached files
        LLAMA_DIR = os.getenv("LLAMA_DIR")
        if LLAMA_DIR:
            if any([os.getenv("LLAMA_CKPT_DIR"), os.getenv("LLAMA_TOKENIZER_PATH"), os.getenv("LLAMA_CACHE_PATH")]):
                logger.warning(
                    "LLAMA_DIR is set and will override LLAMA_CKPT_DIR, LLAMA_TOKENIZER_PATH, and LLAMA_CACHE_PATH"
                )
            self.DEFAULT_CKPT_DIR = LLAMA_DIR
            self.DEFAULT_TOKENIZER_PATH = LLAMA_DIR
            self.DEFAULT_CACHE_PATH = os.path.join(LLAMA_DIR, device_name)
        else:
            self.DEFAULT_CKPT_DIR = os.getenv("LLAMA_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/")
            self.DEFAULT_TOKENIZER_PATH = os.getenv(
                "LLAMA_TOKENIZER_PATH", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/"
            )
            self.DEFAULT_CACHE_PATH = os.getenv(
                "LLAMA_CACHE_PATH", f"/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/{device_name}/"
            )

        if not dummy_weights:
            # Assert if all folders and files exist
            assert os.path.exists(
                self.DEFAULT_CKPT_DIR
            ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export LLAMA_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"
            ), f"Tokenizer file {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'} does not exist, please use export LLAMA_TOKENIZER_PATH=..."
            if not os.path.exists(self.DEFAULT_CACHE_PATH):
                os.makedirs(self.DEFAULT_CACHE_PATH)
            assert os.path.exists(
                self.DEFAULT_CACHE_PATH
            ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export LLAMA_CACHE_PATH=..."
            # Check if weights exist in the specified folder. If not warn the user to run the download and untar script.
        #            assert os.path.isfile(
        #                self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        #            ), f"weights consolidated.00.pth file does not exist. Please use the script `models/demos/llama3/scripts/get_weights.py` to download and untar the weights."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")
        if (
            dummy_weights
        ):  # TODO Choose the correct dummy weights for the correct model specified (take model as arg input?)
            logger.info(f"Note: Using dummy weights, weight caching disabled")
            self._set_llama_params_from_dict(DEFAULT_LLAMA3_2_3B_PARAMS)
        else:
            self._set_llama_params(self.DEFAULT_CKPT_DIR)

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        # Enable workarounds by default until di/dt issues are fixed
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
        if not self.di_dt_workaround:
            logger.info("Disabling di/dt workaround, re-enable if you see hangs")

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        self.cos, self.sin = precompute_freqs(
            self.head_dim, self.max_seq_len * 2, self.rope_theta, self.use_scaled_rope
        )  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode

        if device is not None:  # Avoid issue with test_llama_torch.py not having a device
            grid = device.compute_with_storage_grid_size()
            self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

            # DRAM weight grid specs for dram sharding matmuls
            self.dram_weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                    )
                }
            )

            # Compute kernels. FP32 acc is needed for accuracy.
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
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
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256 if seqlen >= 2048 else 64,
                k_chunk_size=256 if seqlen >= 2048 else 64,
            )

            # Update the existing configurations using the new class methods
            self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
            self.model_config["XQKV_DECODE_PROGCFG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows, k=self.dim, n=self.qkv_size // self.num_devices, grid_size=(4, 8)
            )

            def find_largest_divisor(n, max_divisor=8):
                for i in range(max_divisor, 0, -1):
                    if n % i == 0:
                        return i
                return 1  # Fallback to 1 if no divisor found

            divisor = find_largest_divisor(self.n_heads // self.num_devices)
            grid_size_x = self.n_heads // self.num_devices // divisor
            grid_size_y = (self.n_heads // self.num_devices) // grid_size_x
            assert (
                grid_size_x * grid_size_y == self.n_heads // self.num_devices
            ), f"Grid size mismatch: {grid_size_x} * {grid_size_y} != {self.n_heads // self.num_devices}"

            self.model_config["ATTN_OUTPUT_PROGCFG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim // self.num_devices,
                n=self.dim,
                grid_size=(grid_size_x, grid_size_y),
            )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = self.matmul_config(
                m=1024, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=(8, 8)
            )
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = self.matmul_config(
                m=1024, k=self.hidden_dim, n=self.dim, grid_size=(8, 8)
            )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"] = lambda seq_len: self.matmul_config(
                m=seq_len, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=(8, 4)
            )
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"] = lambda seq_len: self.matmul_config(
                m=seq_len, k=self.hidden_dim, n=self.dim, grid_size=(8, 4)
            )

            mlp_grid = (4, 8) if self.num_devices < 8 else (2, 4)
            self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=mlp_grid
            )
            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows, k=self.hidden_dim // self.num_devices, n=self.dim, grid_size=mlp_grid
            )

            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 2048),
                k=self.dim,
                n=self.dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= 2048,
            )

            # Calculate largest number of lm_head_num_rows such that self.dim % (lm_head_num_rows * 8) == 0
            lm_head_num_rows = 8
            while self.dim % (32 * lm_head_num_rows * 8) != 0:
                lm_head_num_rows -= 1
                assert (
                    lm_head_num_rows > 0
                ), f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 8) == 0"
            self.lm_head_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)

            self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    nearest_32(self.dim // self.lm_head_grid.num_cores),
                ),  # Shard shape: [32, 128] -> 1 shard per core
                self.lm_head_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, 8 if seq_len >= 2048 else seq_len // 256
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=math.ceil(self.qkv_size / self.num_devices / 32 / 8),  # N / TILE_WIDTH / grid width
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

            assert self.n_kv_heads % self.num_devices == 0, "n_kv_heads must be divisible by num_devices"
            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (((self.n_kv_heads // self.num_devices) * seq_len // 64), self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["MLP_KERNEL_CONFIG_HIFI2"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,  # full precision for bfp8 @ bfp8
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            self.model_config["MLP_KERNEL_CONFIG_HIFI4"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,  # full precision for bf16 @ bfp8
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=32,
            )

            self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            self.model_config["HEIGHT_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1
            )

            # Useful core grid based on batch size
            if self.max_batch_size == 32:
                grid_by_batch = (8, 4)
            elif self.max_batch_size == 16:
                grid_by_batch = (8, 2)
            elif self.max_batch_size == 8:
                grid_by_batch = (8, 1)
            elif self.max_batch_size == 4:
                grid_by_batch = (4, 1)
            elif self.max_batch_size == 2:
                grid_by_batch = (2, 1)
            elif self.max_batch_size == 1:
                grid_by_batch = (1, 1)
            else:
                raise ValueError(f"Batch size {self.max_batch_size} not supported")
            core_grid_by_batch = ttnn.CoreGrid(y=grid_by_batch[1], x=grid_by_batch[0])
            core_range_set_by_batch = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_by_batch[0] - 1, grid_by_batch[1] - 1),
                    ),
                }
            )

            self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(math.ceil(self.n_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
                core_grid=core_grid_by_batch,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["ROT_MAT_BMM_PROGCFG"] = lambda m, k, n: ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_by_batch,
                in0_block_w=math.ceil(k / 32),
                out_subblock_h=1,
                out_subblock_w=1,  # TODO How to choose this subblock size?
                per_core_M=math.ceil(m / 32),
                per_core_N=math.ceil(n / 32),
            )
            self.model_config["ROT_MAT_MEMCONFIG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    core_range_set_by_batch,
                    [
                        128,
                        128,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )

            # Width sharded
            mlp_core_grid = ttnn.CoreGrid(y=mlp_grid[0], x=mlp_grid[1])
            self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // mlp_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                mlp_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attn_input_grid = ttnn.CoreGrid(y=4, x=8)
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // attn_input_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                attn_input_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Vision model configs
            self.model_config["IMAGE_MLP_FC_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=self.vision_hidden_dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_MLP_PROJ_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_hidden_dim // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_QKV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3)
                // self.num_devices,  # Head dim was padded to nearest 32
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_OUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3) // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["VISION_XATTN_Q_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim,
                n=(self.head_dim * self.n_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= 1024,
            )
            self.model_config["VISION_XATTN_KV_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim,
                n=(self.head_dim * self.n_kv_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= 1024,
            )
            self.model_config["VISION_XATTN_SCORE_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=self.head_dim,
                n=cache_seq_len,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_OUTPUT_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=cache_seq_len,
                n=self.head_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_DENSE_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=seq_len,
                k=self.dim // self.num_devices,
                n=self.dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

    def _set_llama_params_from_dict(self, params):
        # Text params
        self.dim = params["dim"]
        self.ffn_dim_multiplier = params["ffn_dim_multiplier"]
        self.multiple_of = params["multiple_of"]
        self.n_heads = params["n_heads"]
        self.n_kv_heads = params["n_kv_heads"]
        self.n_layers = params["n_layers"]
        self.norm_eps = params["norm_eps"]
        self.rope_theta = params["rope_theta"]
        self.use_scaled_rope = params["use_scaled_rope"]
        self.vocab_size = params["vocab_size"]
        self.head_dim = self.dim // self.n_heads
        self.hidden_dim = calculate_hidden_dim(self.dim, self.ffn_dim_multiplier, self.multiple_of)

        # Vision params
        self.vision_chunk_size = params.get("vision_chunk_size", -1)
        self.vision_max_num_chunks = params.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = params.get("vision_num_cross_attention_layers", -1)

        # Vision constants
        self.vision_dim = 1280
        self.vision_mlp_ratio = 4
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_act_layer = ttnn.UnaryOpType.GELU
        self.vision_dropout = 0.0
        self.vision_attn_n_heads = 16
        self.vision_head_dim = self.vision_hidden_dim // self.vision_attn_n_heads
        self.vision_n_layers = 32
        self.vision_n_global_layers = 8
        self.vision_max_num_tiles = 4
        self.vision_patch_size = 14
        self.vision_in_channels = 3

    def _set_llama_params(self, checkpoint_dir):
        params_file = os.path.join(checkpoint_dir, "params.json")
        assert os.path.exists(params_file), f"params.json file not found at {params_file}"
        with open(params_file, "r") as f:
            params = json.load(f)
        self._set_llama_params_from_dict(params)

    def __repr__(self):
        return f"""ModelArgs(
    dim={self.dim},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_kv_heads={self.n_kv_heads},
    vocab_size={self.vocab_size},
    multiple_of={self.multiple_of},
    ffn_dim_multiplier={self.ffn_dim_multiplier},
    norm_eps={self.norm_eps},
    rope_theta={self.rope_theta},
    use_scaled_rope={self.use_scaled_rope},
    max_batch_size={self.max_batch_size},
    max_seq_len={self.max_seq_len},
    vision_chunk_size={self.vision_chunk_size},
    vision_max_num_chunks={self.vision_max_num_chunks},
    vision_num_cross_attention_layers={self.vision_num_cross_attention_layers}
)"""

    def is_vision(self):
        return self.vision_chunk_size > 0

    def get_state_dict_prefix(self, module_name, layer_num):
        text_prefix = "text_model." if self.is_vision() else ""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "TtLlamaMLP": "feed_forward",
            "TtLlamaAttention": "attention",
            "TtTransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }
        return text_prefix + layer_prefix + module_map[module_name]

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
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
        """Generate or load state_dict for n_layers of the model"""
        if self.dummy_weights:
            reference_model = Transformer(self)
            state_dict = reference_model.state_dict()
            state_dict = {k: torch.randn_like(v) for k, v in state_dict.items()}
        else:
            state_dict = load_llama_state_dict(self.DEFAULT_CKPT_DIR, self.n_layers)
            # state_dict = torch.load(self.consolidated_weights_path, map_location=torch.device("cpu"))

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, 32))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = 12
        tile_size = 32
        padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR, False
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    @staticmethod
    def matmul_config(
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        per_core_M = math.ceil(m / (32 * grid_size[1]))
        per_core_N = math.ceil(n / (32 * grid_size[0]))

        out_subblock_h = 1
        out_subblock_w = 4
        while out_subblock_w > 1:
            if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
                break
            out_subblock_w -= 1

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=1,  # in0_block_w if in0_block_w is not None else max(1, k // (32 * grid_size[0])), # FIXME
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    @staticmethod
    def dram_matmul_config(
        m: int, k: int, n: int, grid_size: Tuple[int, int]
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=math.ceil(k / (32 * grid_size[0] * grid_size[1])),
            per_core_M=math.ceil(m / 32),
            per_core_N=math.ceil(n / (32 * grid_size[0] * grid_size[1])),
            fused_activation=None,
        )


def load_llama_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}

    print(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        if n_layers:
            # Layer range is in the file name, like layers_start-end.pth
            layer_range = ckpt.stem.split("_")[1]
            start_layer, end_layer = map(int, layer_range.split("-"))
            if start_layer > n_layers + start_layer_idx:
                continue
            if end_layer < start_layer_idx:
                continue

        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        checkpoint.update(loaded_ckpt)
    return checkpoint


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    print(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for (
            key,
            value,
        ) in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or "norm" in key:
            checkpoint[key] = value[0]
        else:
            if key == "tok_embeddings.weight" or key == "output.weight":
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint
