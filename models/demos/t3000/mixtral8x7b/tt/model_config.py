# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from loguru import logger
import torch
import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import Transformer


class TtModelArgs:
    # Default Mixtral parameters
    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    vocab_size = 32000

    max_batch_size = 32  # default
    max_seq_len = 32768  # default
    moe = True
    num_experts = 8
    num_experts_per_tok = 2

    # Default folder location for weights and cached files
    DEFAULT_CKPT_DIR = os.getenv("MIXTRAL_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/")
    DEFAULT_TOKENIZER_PATH = os.getenv("MIXTRAL_TOKENIZER_PATH", "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/")
    DEFAULT_CACHE_PATH = os.getenv("MIXTRAL_CACHE_PATH", "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/")

    # Keys to be used by the different modules of Mixtral
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
        "ATTN_CACHE_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        # "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "LM_HEAD_OUTPUT",
        "ATTN_W_LAYOUT",
        # RMS norm
        "NORM_WEIGHTS",
        # MoE
        "GATE_W_LAYOUT",
        "GATE_WEIGHTS",
        "GATE_MM_OUTPUT",
        # Output
        "OUTPUT_W_LAYOUT",
        "OUTPUT_WEIGHTS",
        "OUTPUT_MM",
    )

    def __init__(self, device=None, instruct=False, dummy_weights=False, max_seq_len=32768, max_batch_size=32):
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        if not dummy_weights:
            # Assert if all folders and files exist
            assert os.path.exists(
                self.DEFAULT_CKPT_DIR
            ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export MIXTRAL_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_CKPT_DIR + "/repack_weights.pt"
            ), f"Repacked weights {self.DEFAULT_CKPT_DIR + '/repack_weights.pt'} does not exist, please use export MIXTRAL_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"
            ), f"Tokenizer file {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'} does not exist, please use export MIXTRAL_TOKENIZER_PATH=..."
            assert os.path.exists(
                self.DEFAULT_CACHE_PATH
            ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export MIXTRAL_CACHE_PATH=..."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")
        if dummy_weights:
            logger.info(f"Note: Using dummy weights, weight caching disabled")

        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)
        self.consolidated_weights_path = lambda i: str(self.model_base_path / f"consolidated.{i:02d}.pt")
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.model"
        self.state_dict_path = str(self.model_base_path / "repack_weights.pt")
        self.instruct = instruct
        self.dummy_weights = dummy_weights

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (By default weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        # Set configurations for sharded type
        self.model_config["WIDTH_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
        )
        self.model_config["HEIGHT_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1
        )
        self.model_config["BLOCK_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1
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
        else:
            raise ValueError(f"Batch size {self.max_batch_size} not supported")

        # Create sharded memory configs for different ops
        self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreGrid(y=4, x=6),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["Q_TRANSPOSE_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"] = cached_lambda(
            lambda padded_layer_past_len: ttnn.create_sharded_memory_config(
                shape=(32, padded_layer_past_len),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        )

        self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=core_grid_by_batch,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Create program configs for the different ttlib matmul ops
        self.model_config["ROT_MAT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
        )

        self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"] = cached_lambda(
            lambda padded_layer_past_len: ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
                subblock_w=1,
                block_h=1,  # Shard_height // 32,
                block_w=padded_layer_past_len // 32,  # Dynamic
            )
        )

        self.model_config["QKV_MM_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["SCORES_BATCHED_MM_PROGCFG"] = cached_lambda(
            lambda p: ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=p,
            )
        )

        self.model_config["VALUES_BATCHED_MM_PROGCFG"] = cached_lambda(
            lambda p: ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=p,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
            )
        )

        self.model_config["LM_HEAD_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["FF1_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=7,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )

        self.model_config["FF3_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=7,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["FF2_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=7,  # K = 14336 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            # Issue #8959: Increasing subblock to 2 results in hangs -> Potentially related to di/dt hangs.
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=2,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(7, 6),  # TODO Hanging with full coreGrid (8,8)
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
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
        self.model_config["PREFILL_MLP_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.model_config["WQKV_PREFILL_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=3,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        self.model_config["WO_PREFILL_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        # Chunk values based on what works best empirically
        self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=256 if seqlen >= 8192 * 2 else (128 if seqlen > 128 else 64),
            k_chunk_size=512 if seqlen >= 8192 * 2 else (128 if seqlen > 128 else 64),
        )

        if device is not None:  # Avoid issue with test_mixtral_torch.py not having a device
            grid_size = device.compute_with_storage_grid_size()
            # TODO Lower max grid size (used by MLP) to avoid hangs
            self.max_grid_size = ttnn.CoreGrid(y=7, x=6)  # (y,x)  (y=7, x=8)
            # self.max_grid_size = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)  # (y,x)  (y=7, x=8)
            self.core_grid_attention = (
                ttnn.CoreGrid(y=4, x=8) if (4 <= grid_size.y and 8 <= grid_size.x) else self.max_grid_size
            )

        # Create Compute kernel configs
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_attn_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.model_config["GATE_MM_OUTPUT_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def weight_cache_path(self, dtype):
        if self.dummy_weights:
            return None
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_cache_path
                / {
                    ttnn.bfloat16: "mixtral_tensor_cache_instruct_bf16",
                    ttnn.bfloat8_b: "mixtral_tensor_cache_instruct_bfp8",
                    ttnn.bfloat4_b: "mixtral_tensor_cache_instruct_bfp4",
                }[dtype]
            )
        else:
            return (
                self.model_cache_path
                / {
                    ttnn.bfloat16: "mixtral_tensor_cache_bf16",
                    ttnn.bfloat8_b: "mixtral_tensor_cache_bfp8",
                    ttnn.bfloat4_b: "mixtral_tensor_cache_bfp4",
                }[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_compute_kernel_config(self):
        return self.compute_kernel_config

    def get_compute_kernel_attn_config(self):
        return self.compute_kernel_attn_config

    def load_state_dict(self):
        """Generate or load state_dict for the first n_layers of the model"""
        if self.dummy_weights:
            reference_model = Transformer(args=self)
            state_dict = reference_model.state_dict()
        else:
            state_dict = torch.load(self.state_dict_path)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}" for i in range(self.n_layers, 32)]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict


def cached_lambda(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper
