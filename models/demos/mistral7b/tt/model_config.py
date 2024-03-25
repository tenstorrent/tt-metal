# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from pathlib import Path
from models.utility_functions import is_wormhole_b0


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
    max_batch_size = 32
    max_seq_len = 4096
    kv_seq_len = 512  # TODO Update the initial cache size when scaling up (Should be window_size == 4096)

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
    )

    def __init__(self, device, model_base_path="/mnt/MLPerf/ttnn/models/demos/mistral7b", instruct=False):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        if instruct:  # Load instruct weights and tokenizer (Mistral-7B-Instruct-v0.2)
            self.consolidated_weights_path = str(self.model_base_path / "consolidated_instruct.00.pth")
            self.tokenizer_path = str(self.model_base_path / "tokenizer_instruct.model")
        else:  # Load generative weights and tokenizer (Mistral-7B-v0.1)
            self.consolidated_weights_path = str(self.model_base_path / "consolidated.00.pth")
            self.tokenizer_path = str(self.model_base_path / "tokenizer.model")

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

    def weight_cache_path(self, dtype, instruct=False):
        # Keep the weight cache separate for generative and instruct weights
        if instruct:
            return (
                self.model_base_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_base_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_compute_kernel_config(self):
        return self.compute_kernel_config
