# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
from pathlib import Path
from models.utility_functions import is_wormhole_b0
from loguru import logger
import tarfile
import urllib.request


class TtModelArgs:
    """Model args for Mistral 7B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    sliding_window = 4096
    vocab_size = 32000

    # Parameters for our use
    max_batch_size = 8
    max_seq_len = 4096
    kv_seq_len = 4096

    # Default folder location for weights and cached files
    DEFAULT_CKPT_DIR = os.getenv("MISTRAL_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/")
    DEFAULT_TOKENIZER_PATH = os.getenv("MISTRAL_TOKENIZER_PATH", "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/")
    DEFAULT_CACHE_PATH = os.getenv("MISTRAL_CACHE_PATH", "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/")

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
        "LM_HEAD_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DEC_SKIP_OUTPUT",
    )

    def __init__(self, device, instruct=False):
        # Assert if all folders and files exist
        assert os.path.exists(
            self.DEFAULT_CKPT_DIR
        ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export MISTRAL_CKPT_DIR=..."
        assert os.path.isfile(
            self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"
        ), f"Tokenizer file {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'} does not exist, please use export MISTRAL_TOKENIZER_PATH=..."
        assert os.path.exists(
            self.DEFAULT_CACHE_PATH
        ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export MISTRAL_CACHE_PATH=..."
        # Check if weights exist in the specified folder. If not warn the user to run the download and untar script.
        assert os.path.isfile(
            self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        ), f"weights consolidated.00.pth file does not exist. Please use the script `models/demos/wormhole/mistral7b/scripts/get_weights.py` to download and untar the weights."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        if device is not None:  # Avoid issue with test_mistral_torch.py not having a device
            grid_size = device.compute_with_storage_grid_size()
            for i in range(grid_size.y, 0, -1):
                # Force the number of rows in the grid to be a factor of max_batch_size for a valid sharding
                if self.max_batch_size % i == 0:
                    grid_size_y = i
                    break
            assert (
                self.max_batch_size % grid_size_y == 0
            ), f"Number of rows in the grid should be a factor of max_batch_size ({self.max_batch_size})"
            self.max_grid_size = ttnn.CoreGrid(y=grid_size_y, x=grid_size.x)  # (y,x)

            # Add sharded memory config for MLP FF1/FF3
            mlp_shard_config = ttnn.create_sharded_memory_config(
                [self.max_batch_size, self.hidden_dim], self.max_grid_size, ttnn.ShardStrategy.WIDTH
            )
            self.model_config["FF1_OUTPUT_MEMCFG"] = mlp_shard_config
            self.model_config["FF3_OUTPUT_MEMCFG"] = mlp_shard_config

            # Compute kernel shared by attention and MLP. FP32 acc is needed for accuracy
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256 if seqlen > 8192 * 2 else (128 if seqlen >= 8192 else 64),
                k_chunk_size=256 if seqlen > 8192 * 2 else (128 if seqlen >= 8192 else 64),
            )

            self.model_config["PREFILL_MLP_W1_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=4,  # 32, #16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=ttnn.UnaryOpType.SILU,
                fuse_batch=False,
            )

            self.model_config["PREFILL_MLP_W3_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )

            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            self.model_config["PREFILL_MLP_W1_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=1,  # 32, #16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=ttnn.UnaryOpType.SILU,
                fuse_batch=False,
            )

            self.model_config["PREFILL_MLP_W3_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )

            self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, seq_len // (512 if seq_len > 2048 else 256)
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, seq_len // (512 if seq_len > 2048 else 256)
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=24,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (seq_len // 8, self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["MLP_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

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

    def get_compute_kernel_config(self):
        return self.compute_kernel_config
