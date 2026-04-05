# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Wormhole-specific utilities for all-MiniLM-L6-v2.

Provides input preprocessing and TT-device mean pooling for the MiniLM
sentence embedding model.
"""

import torch

import ttnn


def preprocess_inputs(input_ids, token_type_ids, position_ids, attention_mask, device):
    """
    Move tokenized inputs to TT device.

    Args:
        input_ids: [batch, seq] int64
        token_type_ids: [batch, seq] int64
        position_ids: [batch, seq] int64
        attention_mask: [batch, seq] int64 (1 = real token, 0 = pad)
        device: TT device

    Returns:
        tt_input_ids, tt_token_type_ids, tt_position_ids, tt_attn_mask
    """
    batch_size = input_ids.shape[0]

    # Extended attention mask: [batch, 1, 1, seq] with -10000 for padding
    ext_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

    tt_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_position_ids = ttnn.from_torch(
        position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_attn_mask = ttnn.from_torch(
        ext_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_input_ids, tt_token_type_ids, tt_position_ids, tt_attn_mask


def mean_pooling_tt(tt_hidden, attention_mask, batch_size, seq_len, hidden_size):
    """
    Mean pooling on CPU after converting TT tensor back to PyTorch.

    Args:
        tt_hidden: TT tensor [batch, 1, seq, hidden] from model output
        attention_mask: PyTorch tensor [batch, seq]
        batch_size: int
        seq_len: int
        hidden_size: int

    Returns:
        sentence_embeddings: [batch, hidden_size] PyTorch tensor
    """
    tt_hidden_pt = ttnn.to_torch(tt_hidden)
    while tt_hidden_pt.dim() > 3:
        tt_hidden_pt = tt_hidden_pt.squeeze(1)
    tt_hidden_pt = tt_hidden_pt[:batch_size, :seq_len, :hidden_size]

    mask_expanded = attention_mask.unsqueeze(-1).expand(tt_hidden_pt.size()).float()
    sum_embeddings = torch.sum(tt_hidden_pt.float() * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask
