# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities and configuration for Bark Small TTNN implementation.

Architecture details (from HuggingFace suno/bark-small config.json):
- hidden_size: 768
- num_heads: 12
- num_layers: 12
- intermediate_size: 3072 (4 * hidden_size)
- bias: False (no bias in attention/MLP for causal stages)
- block_size: 1024
"""

from dataclasses import dataclass
from typing import Optional

import torch
import ttnn


@dataclass
class BarkConfig:
    """Configuration mirroring HuggingFace BarkSmall architecture."""

    # Model dimensions (from config.json)
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072  # 4 * hidden_size

    # Bias - causal models (semantic, coarse) have bias=False
    # fine model has bias=True for LayerNorm
    bias: bool = False

    # Vocabulary sizes per sub-model (from config.json)
    semantic_input_vocab_size: int = 129_600
    semantic_output_vocab_size: int = 10_048
    coarse_input_vocab_size: int = 12_096
    coarse_output_vocab_size: int = 12_096
    fine_input_vocab_size: int = 1_056
    fine_output_vocab_size: int = 1_056

    # Codebook configuration
    n_coarse_codebooks: int = 2
    n_fine_codebooks: int = 8
    n_codes_given: int = 1  # fine model: number of codebooks given as input

    # Block size (max position embeddings)
    block_size: int = 1024

    # EnCodec
    encodec_bandwidth: float = 6.0
    sample_rate: int = 24_000
    codebook_size: int = 1024

    # Generation
    semantic_temperature: float = 0.7
    coarse_temperature: float = 0.7
    fine_temperature: float = 0.5

    layer_norm_eps: float = 1e-5
    dropout: float = 0.0


def get_bark_small_config():
    """Return config for Bark Small (80M params per stage)."""
    return BarkConfig()


def load_tt_tensor(
    torch_tensor: torch.Tensor,
    device: ttnn.Device,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: ttnn.DataType = ttnn.bfloat16,
    cache_file_name: Optional[str] = None,
) -> ttnn.Tensor:
    """Load a PyTorch tensor onto the TT device."""
    # Ensure rank 4 for TTNN ops
    while len(torch_tensor.shape) < 4:
        torch_tensor = torch_tensor.unsqueeze(0)

    tt_tensor = ttnn.as_tensor(
        torch_tensor,
        device=device,
        layout=layout,
        memory_config=memory_config,
        dtype=dtype,
        cache_file_name=cache_file_name,
    )
    return tt_tensor
