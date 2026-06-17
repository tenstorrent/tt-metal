# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class CosyVoiceModelConfig:
    """Configuration for the CosyVoice-300M model on TT hardware."""

    # Model architecture
    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 4096
    text_token_size: int = 151936  # Qwen2 tokenizer vocab size

    # LLM backbone
    llm_num_layers: int = 24
    llm_num_heads: int = 14
    llm_head_dim: int = 64
    llm_hidden_size: int = 896  # llm_input_size * num_heads
    llm_intermediate_size: int = 4864

    # Flow matching decoder
    flow_input_size: int = 512
    flow_output_size: int = 80  # mel bands
    flow_spk_embed_dim: int = 192
    flow_vocab_size: int = 4096
    flow_input_frame_rate: int = 50
    flow_in_channels: int = 240
    flow_out_channel: int = 80

    # DiT parameters
    dit_num_heads: int = 8
    dit_attention_head_dim: int = 64
    dit_n_blocks: int = 4
    dit_num_mid_blocks: int = 12
    dit_channels: list = None

    # HiFi-GAN vocoder
    hifigan_upsample_rates: list = None
    hifigan_upsample_kernel_sizes: list = None
    hifigan_resblock_kernel_sizes: list = None
    hifigan_resblock_dilation_sizes: list = None

    # TT hardware
    max_seq_len: int = 2048
    batch_size: int = 1
    core_grid: tuple = (8, 8)

    def __post_init__(self):
        if self.dit_channels is None:
            self.dit_channels = [256, 256]
        if self.hifigan_upsample_rates is None:
            self.hifigan_upsample_rates = [5, 4, 4, 2, 2]
        if self.hifigan_upsample_kernel_sizes is None:
            self.hifigan_upsample_kernel_sizes = [10, 8, 8, 4, 4]
        if self.hifigan_resblock_kernel_sizes is None:
            self.hifigan_resblock_kernel_sizes = [3, 7, 11]
        if self.hifigan_resblock_dilation_sizes is None:
            self.hifigan_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def create_model_config(
    batch_size: int = 1,
    max_seq_len: int = 2048,
    mode: str = "decode",
) -> CosyVoiceModelConfig:
    """Create a model config with TTNN memory configurations."""
    config = CosyVoiceModelConfig(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # Memory configs for sharded tensors
    configs = {}

    # Sharded memory config for LLM weights (DRAM)
    configs["dram_weight_config"] = ttnn.DRAM_MEMORY_CONFIG

    # Sharded memory config for activations (L1)
    configs["l1_weight_config"] = ttnn.L1_MEMORY_CONFIG

    # Config for attention KV cache
    configs["kv_cache_config"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, batch_size * max_seq_len, config.llm_hidden_size),
        core_grid=ttnn.CoreGrid(y=config.core_grid[0], x=config.core_grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Config for intermediate activations
    configs["activation_config"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, batch_size, config.llm_hidden_size),
        core_grid=ttnn.CoreGrid(y=config.core_grid[0], x=config.core_grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Data types
    configs["dtype"] = {
        "activations": ttnn.bfloat16,
        "weights": ttnn.bfloat8_b,
    }

    # Layer norm config for sharded multi-core
    configs["layernorm_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[config.core_grid[1], config.core_grid[0]],
        subblock_w=4,
        block_h=batch_size,
        block_w=8,
        inplace=False,
    )

    return config


def get_weight_cache_path(model_name: str, cache_dir: str = "/tmp/tt-metal-weights") -> str:
    """Get the path for weight cache directory."""
    path = os.path.join(cache_dir, model_name)
    os.makedirs(path, exist_ok=True)
    return path
