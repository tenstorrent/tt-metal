# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
from pathlib import Path
from loguru import logger
import torch
import json
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.wormhole.llama31_8b_N300.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix
from typing import Tuple


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
    paged_attention_config = None

    # Parameters for our use
    max_batch_size = 1
    # max_seq_len = 131072
    max_seq_len = 8192 * 4 * 4  # 128k
    kv_seq_len = 8192 * 4 * 4  # 128k
    sliding_window = 8192 * 4 * 4  # 128k

    # Default folder location for weights and cached files
    LLAMA_DIR = os.getenv("LLAMA_DIR")
    if LLAMA_DIR:
        if any([os.getenv("LLAMA_CKPT_DIR"), os.getenv("LLAMA_TOKENIZER_PATH"), os.getenv("LLAMA_CACHE_PATH")]):
            logger.warning(
                "LLAMA_DIR is set and will override LLAMA_CKPT_DIR, LLAMA_TOKENIZER_PATH, and LLAMA_CACHE_PATH"
            )
        DEFAULT_CKPT_DIR = LLAMA_DIR
        DEFAULT_TOKENIZER_PATH = LLAMA_DIR
        DEFAULT_CACHE_PATH = os.path.join(LLAMA_DIR, "N300")
    else:
        DEFAULT_CKPT_DIR = os.getenv("LLAMA_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/")
        DEFAULT_TOKENIZER_PATH = os.getenv("LLAMA_TOKENIZER_PATH", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/")
        DEFAULT_CACHE_PATH = os.getenv("LLAMA_CACHE_PATH", "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B/N300/")

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

    # Add this near the top of the class, with other class attributes
    num_devices = 2

    def __init__(self, device, instruct=False, dummy_weights=False, max_batch_size=1):
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
        self.max_batch_size = max_batch_size

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
            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256 if seqlen > 2048 else 64,
                k_chunk_size=512 if seqlen > 2048 else 64,
            )

            # Function definitions remain the same
            def matmul_config(
                m: int, k: int, n: int, grid_size: Tuple[int, int], in0_block_w: int = None, fuse_batch: bool = False
            ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
                per_core_M = m // (32 * grid_size[1])
                per_core_N = n // (32 * grid_size[0])

                out_subblock_h = 1
                out_subblock_w = 4
                while out_subblock_w > 1:
                    if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
                        break
                    out_subblock_w -= 1

                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=in0_block_w if in0_block_w is not None else k // (32 * grid_size[0]),
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=fuse_batch,
                )

            def dram_matmul_config(
                m: int, k: int, n: int, grid_size: Tuple[int, int]
            ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
                return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=k // (32 * grid_size[0] * grid_size[1]),
                    per_core_M=m // 32,
                    per_core_N=n // (32 * grid_size[0] * grid_size[1]),
                    fused_activation=None,
                )

            # Update the existing configurations using the new function names
            self.model_config["XQKV_DECODE_PROGCFG"] = dram_matmul_config(
                m=32, k=self.dim, n=3 * self.dim, grid_size=(4, 8)
            )

            self.model_config["ATTN_OUTPUT_PROGCFG"] = dram_matmul_config(
                m=32, k=self.dim // self.num_devices, n=self.dim, grid_size=(2, 8)
            )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = matmul_config(
                m=1024, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=(8, 8)
            )

            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = matmul_config(
                m=1024, k=self.hidden_dim, n=self.dim, grid_size=(8, 8)
            )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"] = lambda seq_len: matmul_config(
                m=seq_len, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=(8, 4)
            )

            if self.di_dt_workaround:
                self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"] = lambda seq_len: matmul_config(
                    m=seq_len, k=self.hidden_dim, n=self.dim, grid_size=(8, 4)
                )
            else:
                self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"] = lambda seq_len: matmul_config(
                    m=seq_len, k=self.hidden_dim, n=self.dim, grid_size=(8, 4)
                )

            self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = dram_matmul_config(
                m=self.n_heads, k=self.dim, n=self.hidden_dim // self.num_devices, grid_size=(4, 8)
            )

            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = dram_matmul_config(
                m=self.n_heads, k=self.hidden_dim // self.num_devices, n=self.dim, grid_size=(4, 8)
            )

            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: matmul_config(
                m=seq_len, k=self.dim, n=self.dim, grid_size=(8, 8), in0_block_w=1, fuse_batch=seq_len <= 2048
            )

            # N = dim * 3
            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, 8 if seq_len >= 2048 else seq_len // 256
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=self.dim * 3 // 2 // 32 // 8,  # previously 12 for 8b :shrug:,  # N / TILE_WIDTH / grid width
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 2048,
            )

            import math

            if self.di_dt_workaround:
                per_core_N = math.ceil((self.vocab_size // self.num_devices) // (7 * 8 * 32))
                self.model_config["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(7, 8),
                    in0_block_w=1,
                    per_core_M=1,
                    per_core_N=per_core_N,  # vocab size / num_devices / (7 * 8 * 32 cores), rounded up
                    out_subblock_h=1,
                    out_subblock_w=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                per_core_N = math.ceil((self.vocab_size // self.num_devices) // (8 * 8 * 32))
                self.model_config["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=per_core_N,  # vocab size / num_devices / (8 * 8 * 32 cores), rounded up
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )

            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (seq_len // 16, self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["MLP_KERNEL_CONFIG_HIFI2"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,  # full precision for bfp8 @ bfp8
                math_approx_mode=False,
                fp32_dest_acc_en=False,
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
            self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (32, self.dim // 32),  # Shard shape: [32, 128] -> 1 shard per core
                ttnn.CoreGrid(y=4, x=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["SHARDED_SKIP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (32, self.dim // 32),  # Shard shape: [32, 128] -> 1 shard per core
                ttnn.CoreGrid(y=4, x=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
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
