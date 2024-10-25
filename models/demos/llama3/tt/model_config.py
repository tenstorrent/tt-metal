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
from models.demos.llama3.tt.llama_common import (
    precompute_freqs,
    freqs_to_rotation_matrix,
    num_to_core_range_set,
    calculate_hidden_dim,
    get_out_subblock_w,
)
from typing import Tuple
from models.utility_functions import nearest_32
from pathlib import Path
from tqdm import tqdm


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

    LOCAL_LLAMA_PARAMS = {
        "LLAMA3_2_1B_PARAMS": "models/demos/llama3/model_params/Llama3.2-1B-Instruct",
        "LLAMA3_2_3B_PARAMS": "models/demos/llama3/model_params/Llama3.2-3B-Instruct",
        "LLAMA3_1_8B_PARAMS": "models/demos/llama3/model_params/Llama3.1-8B-Instruct",
        "LLAMA3_2_11B_PARAMS": "models/demos/llama3/model_params/Llama3.2-11B-Vision-Instruct",
        "LLAMA3_1_70B_PARAMS": "models/demos/llama3/model_params/Llama3.1-70B-Instruct",
    }

    def __init__(self, mesh_device, instruct=False, dummy_weights=False, max_batch_size=1):
        # Add this near the top of the class, with other class attributes
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.device_name = {0: "CPU", 1: "N150", 2: "N300", 8: "T3K", 32: "TG"}[self.num_devices]
        self.is_large_model = False
        self.model_name = "Unknown"  # Llama model name will be dependent on the checkpoint directory

        LLAMA_DIR = os.getenv("LLAMA_DIR")
        if LLAMA_DIR:
            if any([os.getenv("LLAMA_CKPT_DIR"), os.getenv("LLAMA_TOKENIZER_PATH"), os.getenv("LLAMA_CACHE_PATH")]):
                logger.warning(
                    "LLAMA_DIR is set and will override LLAMA_CKPT_DIR, LLAMA_TOKENIZER_PATH, and LLAMA_CACHE_PATH"
                )
            self.DEFAULT_CKPT_DIR = LLAMA_DIR
            self.DEFAULT_TOKENIZER_PATH = LLAMA_DIR
            self.DEFAULT_CACHE_PATH = os.path.join(LLAMA_DIR, self.device_name)
        else:
            self.DEFAULT_CKPT_DIR = os.getenv(
                "LLAMA_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
            )
            self.DEFAULT_TOKENIZER_PATH = os.getenv(
                "LLAMA_TOKENIZER_PATH", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
            )
            self.DEFAULT_CACHE_PATH = os.getenv(
                "LLAMA_CACHE_PATH", f"/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/{self.device_name}/"
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

        # Set the model name based on the checkpoint directory being loaded
        if "3.2-1B" in LLAMA_DIR:
            local_params = "LLAMA3_2_1B_PARAMS"
            self.model_name = "3.2-1B"
        elif "3.2-3B" in LLAMA_DIR:
            local_params = "LLAMA3_2_3B_PARAMS"
            self.model_name = "3.2-3B"
        elif "3.1-8B" in LLAMA_DIR:
            local_params = "LLAMA3_1_8B_PARAMS"
            self.model_name = "3.1-8B"
        elif "3.2-11B" in LLAMA_DIR:
            local_params = "LLAMA3_2_11B_PARAMS"
            self.model_name = "3.2-11B"
        elif "3.1-70B" in LLAMA_DIR:
            local_params = "LLAMA3_1_70B_PARAMS"
            self.model_name = "3.1-70B"
            self.is_large_model = True
        else:
            raise ValueError(f"Unsupported LLAMA model: {LLAMA_DIR}")

        # Load model params
        if not dummy_weights:
            self._set_llama_params(self.DEFAULT_CKPT_DIR)
        else:  # With Dummy weights, set the params from the local copy inside the model folder. This is required for CI pipeline that doesn't mount the external folders.
            self._set_llama_params(self.LOCAL_LLAMA_PARAMS[local_params])

        # Reduce full 128k context length for combinations with memory constraints
        # Currently: n150 8b and t3k 70b with 8b/8b/8b MLPs
        # Default folder location for weights and cached files
        # TODO Generalize for all llama3 weights
        is_8b = self.dim == 4096 and self.n_layers == 32
        is_70b = self.dim == 8192 and self.n_layers == 80
        if self.num_devices == 1 and is_8b or is_70b:
            self.max_seq_len = 8192 * 4  # 32k
            self.kv_seq_len = 8192 * 4  # 32k
            self.sliding_window = 8192 * 4  # 32k

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        # If the weights file contain the keyword `instruct` also set self.instruct to true
        if "instruct" in self.DEFAULT_CACHE_PATH.lower():
            self.instruct = True
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

        device = mesh_device.get_devices()[0] if mesh_device is not None else None
        if device is not None:  # Avoid issue with test_llama_torch.py not having a device
            self.n_local_heads = self.n_heads // self.num_devices

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

            # Compute kernels. FP32 acc does not appear to be needed for accuracy in model tests or demo runs.
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
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

            self.model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = self.compute_kernel_config_hifi2

            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256 if seqlen >= 2048 else 64,
                k_chunk_size=256 if seqlen >= 2048 else 64,
            )

            def find_largest_divisor(n, max_divisor=8):
                for i in range(max_divisor, 0, -1):
                    if n % i == 0:
                        return i
                return 1  # Fallback to 1 if no divisor found

            # nlp_concat_heads_decode will shard the data across this number of cores
            assert (
                self.n_heads % self.num_devices == 0
            ), f"n_heads must be divisible by num_devices: {self.n_heads} % {self.num_devices}"

            self.model_config["ATTN_OUTPUT_PROGCFG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim // self.num_devices,
                n=self.dim,
                num_cores=self.n_heads // self.num_devices,
            )

            # All Gather Matmul for Dense Out (DO)
            # TODO: Is there a better way to decide if fused all gather matmul should be used? And is there a better way to use the flag, instead of passing it into model_config?
            # NOTE: Fused all gather matmul only suppports a core grid of size num_devices x 1
            self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
                self.ccl_topology() == ttnn.Topology.Ring
                and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
            )

            if self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]:
                do_core_grid_size = (8, 1)
                do_per_core_N = (
                    self.dim // self.num_devices // self.tile_size // (do_core_grid_size[0] * do_core_grid_size[1])
                )
                self.model_config[
                    "ATTN_ALL_GATHER_MATMUL_OUTPUT_PROGCFG"
                ] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=do_core_grid_size,
                    in0_block_w=self.dim
                    // self.tile_size
                    // (do_core_grid_size[0] * do_core_grid_size[1]),  # [32 x 8k] x [8k x 1k] = [32 x 1k]
                    out_subblock_h=1,
                    out_subblock_w=get_out_subblock_w(
                        do_per_core_N, out_subblock_h=1
                    ),  # Max out_subblock_w = 4, needs to be divisible by per_core_N
                    per_core_M=self.tile_padded_batch_rows // self.tile_size,
                    per_core_N=do_per_core_N,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_PROGCFG"] = None

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
            self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)

            self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    nearest_32(self.dim // self.lm_head_core_grid.num_cores),
                ),  # Shard shape: [32, 128] -> 1 shard per core
                self.lm_head_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8  # 8 rows
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
                shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
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
            mlp_core_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // mlp_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                mlp_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.hidden_dim // self.num_devices,
                num_cores=mlp_core_grid.num_cores,
            )

            mlp2_core_grid = self.dram_shard_core_grid_for_k(self.hidden_dim // self.num_devices)
            self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.hidden_dim // self.num_devices // mlp2_core_grid.num_cores,
                ),
                mlp2_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.hidden_dim // self.num_devices,
                n=self.dim,
                num_cores=mlp2_core_grid.num_cores,
            )

            attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // attn_input_grid.num_cores,
                ),
                attn_input_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["XQKV_DECODE_PROGCFG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.qkv_size // self.num_devices,
                num_cores=attn_input_grid.num_cores,
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
            self.model_config["VISION_XATTN_KV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=(self.head_dim * self.n_kv_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
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
                # in0_block_w=1, # TODO: Remove this when we get non-causal FlashDecode
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

            self.model_config["VISION_PROJ_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=seq_len,
                k=self.vision_dim * 6,
                n=self.dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

            self.model_config["CROSS_TRANSFORMER_TEXT_OUTPUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=self.vocab_size // 8,  # Magic number. LM Head always contains 8 splits
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )

            self.VISION_MAX_MM_SEQ = nearest_32(self.vision_chunk_ntok)
            # RMS NORM
            self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
            self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)
            self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(self.lm_head_core_grid)

            # All gather matmuls currently only supported on T3K
            # We need it sharded on num_cores = num_devices
            self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    num_to_core_range_set(self.num_devices),
                    [
                        self.tile_padded_batch_rows,
                        self.dim // self.num_devices,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )

            self.is_2d_fracturing = all([dim > 1 for dim in self.mesh_device.shape]) if self.mesh_device else False
            self.is_multichip = self.num_devices > 1

    def is_distributed_norm(self, mode):
        if not self.is_multichip:
            return False
        if all([dim > 1 for dim in self.mesh_device.shape]):  # 2D grid
            return True
        elif self.dim >= 8192 and mode == "prefill":  # Somewhere between 4k and 8k WH runs out of L1 if not distributed
            return True
        return False

    def ccl_topology(self):
        if self.num_devices == 8:  # T3K
            return ttnn.Topology.Ring
        elif self.num_devices > 1:  # All other multi chip devices
            return ttnn.Topology.Linear
        return None

    def prepare_inputs_ttnn_decode(self, x, input_mem_cfg, force_replicated=False):
        """
        Prepare inputs for decode mode.
        x: (batch, seq, dim)
        """
        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh_device)
            if force_replicated
            else ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
        )

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq_len = x.shape[1]
            assert x.shape[2] == self.dim
        elif len(x.shape) == 4:
            seq_len = x.shape[0]
            assert x.shape[1] == 1
            batch = x.shape[2]
            assert x.shape[3] == self.dim

        assert seq_len == 1, "Only supporting decode mode"

        # Support input on device
        if torch.is_tensor(x):  # Input on host -> Use torch
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, dim]
            # Pad small batches to 32
            if batch < 32:
                zeros = torch.zeros(1, seq_len, 32, self.dim)
                zeros[:, :, :batch, :] = x
                x = zeros
        elif len(x.shape) == 3:  # Input on device -> Use ttnn
            x = ttnn.reshape(x, (batch, seq_len, 1, self.dim))  # [batch, seqlen, dim] -> [batch, seqlen, 1, dim]
            x = ttnn.permute(x, (1, 2, 0, 3))  # [seq_len, 1, batch, dim]
        elif len(x.shape) == 4:
            pass  # already in [seq_len, 1, batch, dim]

        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=input_mem_cfg,
            )
        else:  # Convert the row major layout from embedding back to tile layout
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        return x

    def prepare_inputs_ttnn_prefill(self, x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)

        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh_device)
            if force_replicated
            else ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
        )

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

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
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = 32
        self.vision_n_global_layers = 8
        self.vision_max_num_tiles = 4
        self.vision_patch_size = 14
        self.vision_in_channels = 3

    @property
    def vision_chunk_ntok(self):
        """
        Returns the number of tokens per chunk, accounting for the extra class token
        """
        return (self.vision_chunk_size // self.vision_patch_size) ** 2 + 1

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

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        """Generate or load state_dict for n_layers of the model"""
        if self.dummy_weights:
            reference_model = Transformer(self)
            state_dict = reference_model.state_dict()
            state_dict_prefix = self.get_state_dict_prefix("", None)
            state_dict = {f"{state_dict_prefix}{k}": torch.randn_like(v) for k, v in state_dict.items()}
        else:
            state_dict = load_llama_state_dict(self.DEFAULT_CKPT_DIR, self.n_layers)

        keys_dict = list(state_dict.keys())[:]
        remv = [
            f"layers.{i}." for i in list(range(self.n_layers, 32))
        ]  # TODO, this is not generalized to all models. it assumes max layers = 32
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = 12
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR, False
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def matmul_config(
        self,
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
        per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))

        out_subblock_h = 1
        out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

        if in0_block_w is None:
            in0_block_w = min(4, max(1, k // (self.tile_size * grid_size[0])))

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    def dram_shard_core_grid_for_k(self, k: int) -> Tuple[int, int]:
        rows, cols = self.find_grid(k // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid(self, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.
        The grid size is limited to a maximum of 2 rows and 8 columns.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 4
        max_cols = 8  # Maximum number of rows or columns
        max_cores = max_rows * max_cols  # Maximum number of cores (8x2 grid)

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
        possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
        )

    def dram_matmul_config(
        self, m: int, k: int, n: int, num_cores=None
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        if num_cores is None:
            # Default to using 2 rows of 8 cores as this is fastest, but use 1 row if K is too small
            # Warning: this does not handle the case in which K is too large for 2 rows of 8 cores
            # In that case override grid_size or update this logic to do so
            num_cores = self.dram_shard_core_grid_for_k(k).num_cores
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=math.ceil(k / (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=None,
        )

    def create_sharded_norm_config(self, grid):
        """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

        Args:
            grid (ttnn.CoreGrid): Grid specification for the norm operation
        """
        block_w = self.dim // grid.num_cores // self.tile_size
        # Find largest value <= 4 that evenly divides block_w
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=self.tile_padded_batch_rows // self.tile_size,
            block_w=block_w,
            inplace=False,
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

    (f"Loading {len(checkpoints)} checkpoint files")
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
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
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
