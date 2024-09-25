# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
from pathlib import Path
from loguru import logger
import torch
import json
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer


class TtModelArgs:
    """Model args for Llama 3.1 8B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    vocab_size = 128256
    ffn_dim_multiplier = 1.3
    multiple_of = 1024
    rope_theta = 500000.0
    use_scaled_rope = True

    # Parameters for our use
    max_batch_size = 1
    # max_seq_len = 131072
    max_seq_len = 8192 * 4
    kv_seq_len = 8192 * 4
    sliding_window = 8192 * 4

    # Default folder location for weights and cached files
    DEFAULT_CKPT_DIR = os.getenv("LLAMA_CKPT_DIR", "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/")
    DEFAULT_TOKENIZER_PATH = os.getenv(
        "LLAMA_TOKENIZER_PATH", "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/"
    )
    DEFAULT_CACHE_PATH = os.getenv("LLAMA_CACHE_PATH", "/proj_sw/user_dev/hf_data/llama/Meta-Llama-3.1-8B-Instruct/")

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

    def __init__(self, device, instruct=False, dummy_weights=False):
        if not dummy_weights:
            # Assert if all folders and files exist
            assert os.path.exists(
                self.DEFAULT_CKPT_DIR
            ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export LLAMA_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"
            ), f"Tokenizer file {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'} does not exist, please use export LLAMA_TOKENIZER_PATH=..."
            assert os.path.exists(
                self.DEFAULT_CACHE_PATH
            ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export LLAMA_CACHE_PATH=..."
            # Check if weights exist in the specified folder. If not warn the user to run the download and untar script.
            assert os.path.isfile(
                self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
            ), f"weights consolidated.00.pth file does not exist. Please use the script `models/demos/wormhole/llama31_8b/scripts/get_weights.py` to download and untar the weights."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")
        if dummy_weights:
            logger.info(f"Note: Using dummy weights, weight caching disabled")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.DEFAULT_CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        self.dummy_weights = dummy_weights

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
            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256 if seqlen > 8192 * 2 else (128 if seqlen >= 8192 else 64),
                k_chunk_size=256 if seqlen > 8192 * 2 else (128 if seqlen >= 8192 else 64),
            )
            # in0: [32, 4096]
            # in1: [4096, 6144]
            self.model_config["XQKV_DECODE_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                # grid_size = [4, 8] # SHARDED_SKIP_INPUT_MEMCFG
                in0_block_w=4,  # K(4096) / TILE_WIDTH(32) /grid_size(32)
                per_core_M=1,  # M(32) / TILE_HEIGHT(32)
                per_core_N=6,  # N(4096) / TILE_WIDTH(32) / grid_size(32)
                fused_activation=None,
            )

            # in0: [32, 4096]
            # in1: [4096, 4096]
            self.model_config["ATTN_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                # grid_size = [4, 8] # nlp_concat_heads_decode has 32 heads, and 1x1 shards
                in0_block_w=4,  # K(4096) / TILE_WIDTH(32) /grid_size(32)
                per_core_M=1,  # M(32) / TILE_HEIGHT(32)
                per_core_N=4,  # N(4096) / TILE_WIDTH(32) / grid_size(32)
                fused_activation=None,
            )

            self.model_config["SHARDED_SKIP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (32, 4096 // 32),  # Shard shape: [32, 128] -> 1 shard per core
                ttnn.CoreGrid(y=4, x=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # in0: [B(seqlen//1024), 1024, 4096]
            # in1: [1, 4096, 14336]
            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,  # K(4096) / TILE_SIZE(32) / 32 how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=4,  # M(1024) / TILE_HEIGHT(32) / Grid_Size_height(8) [2D matmul]
                per_core_N=56,  # N(14336) / TILE_WIDTH(32) / Grid_Size_width (8) [2D matmul]
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )

            # in0: [B(seqlen//1024), 1024, 14336]
            # in1: [1, 14336, 4096]
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=4,  # M(1024) / TILE_HEIGHT(32) / Grid_Size_height(8) [2D matmul]
                per_core_N=16,  # N(4096) / TILE_WIDTH(32) / Grid_Size_width (8) [2D matmul]
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            self.model_config[
                "PREFILL_MLP_W1_W3_PRG_CONFIG_128"
            ] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=2,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=seq_len // 32 // 4,  # M(seqlen) / TILE_HEIGHT(32) / Grid_Size(4) [2D matmul]
                per_core_N=56,  # N(14336) / TILE_WIDTH(32) / Grid_Size_width (8) [2D matmul]
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )

            if self.di_dt_workaround:
                self.model_config[
                    "PREFILL_MLP_W2_PRG_CONFIG_128"
                ] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    in0_block_w=1,  # how much inner dim you take each time
                    out_subblock_h=1,  # Must be divisible by per_core_M
                    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                    per_core_M=seq_len // 32 // 4,  # M(seqlen) / TILE_HEIGHT(32) / Grid_Size(4) [2D matmul]
                    per_core_N=16,  # N(4096) / TILE_WIDTH(32) / Grid_Size_width (8) [2D matmul]
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
            else:
                # Make use of all 56 cores: K = 14336 / TILE_size(32) / out_subblock_w(4) / in0_block_w(2) = 56
                self.model_config[
                    "PREFILL_MLP_W2_PRG_CONFIG_128"
                ] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    in0_block_w=2,  # how much inner dim you take each time
                    out_subblock_h=1,  # Must be divisible by per_core_M
                    out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                    per_core_M=seq_len // 32 // 4,  # M(seqlen) / TILE_HEIGHT(32) / Grid_Size(4) [2D matmul]
                    per_core_N=16,  # N(4096) / TILE_WIDTH(32) / Grid_Size_width (8) [2D matmul]
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )

            # Width sharded
            self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (32, 4096 // 64),  # Shard shape: [32, 64] -> 1 shard per core
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            #     x = [32, 4096]
            # W1/W3 = [4096, 14336]
            self.model_config[
                "DECODE_MLP_W1_W3_PRG_CONFIG"
            ] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                # Grid size = [8, 8]
                in0_block_w=2,  # K(4096) / TILE_WIDTH(32) / Grid_Size(64)
                per_core_M=1,  # M(32) / TILE_HEIGHT(32)
                per_core_N=7,  # N(14336) / TILE_WIDTH(32) / Grid_Size(64)
                fused_activation=None,
            )
            # w2_in = [32,14336]
            #    w2 = [14336, 4096]
            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                # Grid size = [8, 8]
                in0_block_w=7,  # K(14336) / TILE_WIDTH(32) / Grid_Size(64)
                per_core_M=1,  # M(32) / TILE_HEIGHT(32)
                per_core_N=2,  # N(4096) / TILE_WIDTH(32) / Grid_Size(64)
                fused_activation=None,
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

            if self.di_dt_workaround:
                self.model_config["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(7, 8),
                    in0_block_w=1,
                    per_core_M=1,
                    per_core_N=72,  # vocab size = 128k = 4008 tiles. 4008/56cores = 72
                    out_subblock_h=1,
                    out_subblock_w=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                self.model_config["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=72,  # vocab size = 128k = 4008 tiles. 4008/56cores = 72
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )

            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (seq_len // 8, self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["MLP_KERNEL_CONFIG_HIFI2"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,  # full precision for bfp8 @ bfp8
                math_approx_mode=True,
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
                core_grid_by_batch = ttnn.CoreGrid(y=4, x=8)
            elif self.max_batch_size == 16:
                core_grid_by_batch = ttnn.CoreGrid(y=2, x=8)
            elif self.max_batch_size == 8:
                core_grid_by_batch = ttnn.CoreGrid(y=1, x=8)
            elif self.max_batch_size == 4:
                core_grid_by_batch = ttnn.CoreGrid(y=1, x=4)
            elif self.max_batch_size == 2:
                core_grid_by_batch = ttnn.CoreGrid(y=1, x=2)
            elif self.max_batch_size == 1:
                core_grid_by_batch = ttnn.CoreGrid(y=1, x=1)
            else:
                raise ValueError(f"Batch size {self.max_batch_size} not supported")
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

            self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 128),
                core_grid=core_grid_by_batch,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["ROT_MAT_BMM_PROGCFG"] = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_by_batch,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
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

    def load_state_dict(self):
        """Generate or load state_dict for n_layers of the model"""
        if self.dummy_weights:
            reference_model = Transformer(self)
            state_dict = reference_model.state_dict()
            state_dict = {k: torch.randn_like(v) for k, v in state_dict.items()}
        else:
            state_dict = torch.load(self.consolidated_weights_path, map_location=torch.device("cpu"))

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, 32))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict
