# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
import json
from pathlib import Path
from loguru import logger
import torch
import ttnn
from models.experimental.grok.reference.model import Grok1Model
from models.experimental.grok.reference.configuration_grok1 import Grok1Config


class TtModelArgs:
    # Grok-1 parameters
    vocab_size = 131072
    hidden_size = 6144
    intermediate_size = 32768
    num_hidden_layers = 1  # 64
    num_attention_heads = 48
    num_key_value_heads = 8
    attn_output_multiplier = 0.08838834764831845
    max_attn_value = 30.0
    max_position_embeddings = 8192
    embedding_multiplier_scale = 78.38367176906169
    output_multiplier_scale = 0.5773502691896257
    rms_norm_eps = 1e-5
    use_cache = True
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    tie_word_embeddings = True
    num_experts_per_tok = 2
    num_experts = 8
    output_router_logits = False
    router_aux_loss_coef = 0.001

    # from Mixtral / our use
    max_batch_size = 32
    head_dim = hidden_size // num_attention_heads
    max_seq_len = max_position_embeddings  # FIXME, just use one variable for this
    dim = hidden_size  # FIXME, just use one variable for this
    moe = True  # FIXME: unused?
    n_layers = num_hidden_layers  # FIXME, just use one variable for this

    # Default folder location for weights and cached files
    DEFAULT_CKPT_DIR = os.getenv("GROK_CKPT_DIR", "/proj_sw/user_dev/hf_data/grok-1")
    DEFAULT_TOKENIZER_PATH = os.getenv("GROK_TOKENIZER_PATH", "/proj_sw/user_dev/hf_data/grok-1")
    DEFAULT_CACHE_PATH = os.getenv("GROK_CACHE_PATH", "/proj_sw/user_dev/hf_data/grok-1")

    # Keys to be used by the different modules of Grok
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
        "NORM_W_LAYOUT",
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

    def __init__(self, device=None, instruct=False, dummy_weights=False):
        if not dummy_weights:
            # Assert if all folders and files exist
            assert os.path.exists(
                self.DEFAULT_CKPT_DIR
            ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export GROK_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_CKPT_DIR + "/pytorch_model.bin.index.json"
            ), f"Repacked weights {self.DEFAULT_CKPT_DIR + '/pytorch_model.bin.index.json'} does not exist, please use export GROK_CKPT_DIR=..."
            assert os.path.isfile(
                self.DEFAULT_TOKENIZER_PATH + "/tokenizer.json"
            ), f"Tokenizer file {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.json'} does not exist, please use export GROK_TOKENIZER_PATH=..."
            assert os.path.exists(
                self.DEFAULT_CACHE_PATH
            ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export GROK_CACHE_PATH=..."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.DEFAULT_TOKENIZER_PATH + '/tokenizer.json'}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")
        if dummy_weights:
            logger.info(f"Note: Using dummy weights, weight caching disabled")

        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)
        self.consolidated_weights_path = lambda i: str(self.model_base_path / f"consolidated.{i:02d}.pt")
        self.tokenizer_path = self.DEFAULT_TOKENIZER_PATH + "/tokenizer.json"
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
        self.model_config["WIDTH_SHARDED_MEMCFG"] = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttnn.experimental.tensor.BufferType.L1
        )
        self.model_config["HEIGHT_SHARDED_MEMCFG"] = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.experimental.tensor.BufferType.L1
        )
        self.model_config["BLOCK_SHARDED_MEMCFG"] = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttnn.experimental.tensor.BufferType.L1
        )

        # Create sharded memory configs for different ops
        self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
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
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        shard_height = 32
        shard_width_hidden_dim_across_32_cores = self.dim // 32  # hidden_size = 6144
        self.model_config["SHARDED_NORM_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width_hidden_dim_across_32_cores),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["SHARDED_NORM_OUTPUT_MEMCFG"] = self.model_config["SHARDED_NORM_INPUT_MEMCFG"]

        # Create program configs for the different ttnn matmul ops
        # TODO: update for 6144 not 4096?
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

        self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"] = cached_lambda(
            lambda padded_layer_past_len: ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
                subblock_w=1,
                block_h=1,  # Shard_height // 32,
                block_w=padded_layer_past_len // 32,  # Dynamic
            )
        )

        self.model_config["GATE_MM_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=24,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        self.model_config["QKV_MM_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=6,
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
            out_subblock_w=3,
            per_core_M=1,
            per_core_N=3,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["FF1_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 6144 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 32768
            fuse_batch=True,
            fused_activation=(ttnn.UnaryOpType.GELU, True),  # FIXME: GET THIS DOCUMENTED
            mcast_in0=True,
        )

        self.model_config["FF3_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 6144 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 32768
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_config["FF2_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            # Issue #8959: Increasing subblock to 2 results in hangs -> Potentially related to di/dt hangs.
            out_subblock_w=3,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=3,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 6144
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

        self.model_config["SHARDED_NORM_PRGM_CFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=3,
            block_h=shard_height // 32,
            block_w=shard_width_hidden_dim_across_32_cores // 32,
            inplace=False,
        )

        if device is not None:  # Avoid issue with test_grok_torch.py not having a device
            grid_size = device.compute_with_storage_grid_size()
            # TODO Lower max grid size (used by MLP) to avoid hangs
            self.max_grid_size = ttnn.CoreGrid(y=7, x=6)  # (y,x)  (y=7, x=8)
            # self.max_grid_size = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)  # (y,x)  (y=7, x=8)
            self.core_grid_attention = (
                ttnn.CoreGrid(y=4, x=8) if (4 <= grid_size.y and 8 <= grid_size.x) else self.max_grid_size
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # LoFi
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_attn_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # HiFi2
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_output_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # HiFi2?
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Create Compute kernel configs
        self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_cache_path
                / {
                    ttnn.bfloat16: "grok_tensor_cache_instruct_bf16",
                    ttnn.bfloat8_b: "grok_tensor_cache_instruct_bfp8",
                    ttnn.bfloat4_b: "grok_tensor_cache_instruct_bfp4",
                }[dtype]
            )
        else:
            return (
                self.model_cache_path
                / {
                    ttnn.bfloat16: "grok_tensor_cache_bf16",
                    ttnn.bfloat8_b: "grok_tensor_cache_bfp8",
                    ttnn.bfloat4_b: "grok_tensor_cache_bfp4",
                }[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_compute_kernel_config(self):
        return self.compute_kernel_config

    def get_compute_kernel_attn_config(self):
        return self.compute_kernel_attn_config

    def get_compute_kernel_output_config(self):
        return self.compute_kernel_output_config

    def load_state_dict(self, start_layer=0):
        """Generate or load state_dict for the first n_layers of the model"""
        # FIXME: PretrainedConfig is from the devil
        grok1_config = Grok1Config.from_json_file("models/experimental/grok/reference/config.json")
        grok1_config.num_hidden_layers = self.n_layers

        if self.dummy_weights:
            reference_model = Grok1Model(config=grok1_config)
            state_dict = reference_model.state_dict()
            state_dict = {f"model.{k}": torch.randn_like(v) for k, v in state_dict.items()}
        else:
            with open(self.model_base_path / "pytorch_model.bin.index.json", "r") as f:
                index = json.load(f)
            required_files = set()
            for layer, file in index["weight_map"].items():
                layer_number = int(layer.split(".")[2]) if layer.startswith("model.layers.") else 0
                if start_layer <= layer_number < self.n_layers + start_layer:
                    required_files.add(file)

            state_dict = {}
            for i, file in enumerate(sorted(required_files)):
                logger.info(f"Loading weight file {i+1}/{len(required_files)}: {file}")
                state_dict.update(torch.load(self.model_base_path / file))

        keys_dict = list(state_dict.keys())[:]
        remv = [
            f"model.layers.{i}." for i in list(range(0, start_layer)) + list(range(self.n_layers + start_layer, 64))
        ]
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
