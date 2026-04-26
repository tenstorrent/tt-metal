# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phi-1 Model Configuration

Defines model hyperparameters and TT-specific configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PhiConfig:
    # Model architecture
    vocab_size: int = 51200
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rotary_dim: int = 32
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_cache: bool = True
    hidden_act: str = "gelu_new"

    # Default TT config (will be updated based on batch size and seq_len)
    _model_config: Dict[str, Any] = None

    def __post_init__(self):
        if self._model_config is None:
            self._model_config = {}

    def update_model_config(self, config_dict: Dict[str, Any]):
        """Update TT-specific model configuration."""
        self._model_config.update(config_dict)

    def get_model_config(self) -> Dict[str, Any]:
        """Return TT model configuration."""
        return self._model_config


def get_model_config(batch_size: int, seq_len: int) -> Dict[str, Any]:
    """
    Generate model config based on batch size and sequence length.
    Follows memory and performance best practices for Wormhole.
    """
    return {
        "BATCH_SIZE": batch_size,
        "SEQ_LEN": seq_len,
        "MAX_SEQ_LEN": 2048,
        # Embedding
        "WORD_EMBEDDING_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        if batch_size == 1
        else ttnn.DRAM_MEMORY_CONFIG,
        "WORD_EMBEDDING_OUTPUT_DTYPE": ttnn.bfloat8_b,
        # Decoder
        "DECODER_ALL_GATHER_OUTPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
        "DECODER_INPUT_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        "DECODER_CROSS_MLP_DENSE_H_TO_4H_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "DECODER_CROSS_MLP_DENSE_4H_TO_H_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "DECODER_CROSS_ATTENTION_QKV_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "DECODER_CROSS_ATTENTION_SOFTMAX_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        "DECODER_CROSS_ATTENTION_OUTPUT_PROJECTIONS_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "DECODER_RESIDUAL_ADD_OUTPUT_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        # SDPA
        "SDPA_OUTPUT_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        "SDPA_PROGCFG": None,
        # MLP
        "MLP_LINEAR_ACT_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        "MLP_W1_W3_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "MLP_W2_OUTPUT_MEMCFG": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # Final LM Head
        "FINAL_LM_HEAD_INPUT_MEMCFG": ttnn.L1_MEMORY_CONFIG,
        "FINAL_LM_HEAD_OUTPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
        # Dtypes
        "DECODER_HIDDEN_DTYPE": ttnn.bfloat8_b,
        "DECODER_MLPDENSE_H_TO_4H_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "DECODER_MLPDENSE_4H_TO_H_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "DECODER_ATTN_OUTPUT_PROJECTIONS_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "DECODER_ATTN_QKV_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "SDPA_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "EMBEDDING_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "ATTN_RESIDUAL_ADD_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "MLP_RESIDUAL_ADD_OUTPUT_DTYPE": ttnn.bfloat8_b,
        "FINAL_LM_HEAD_OUTPUT_DTYPE": ttnn.bfloat16,
        # Sharding
        "DECODER_QKV_SHARD_FACTOR": 8,
        "DECODER_MLP_W1_W3_SHARD_FACTOR": 8,
        "DECODER_MLP_W2_SHARD_FACTOR": 8,
        "ATTN_OUTPUT_PROJ_SHARD_FACTOR": 8,
    }