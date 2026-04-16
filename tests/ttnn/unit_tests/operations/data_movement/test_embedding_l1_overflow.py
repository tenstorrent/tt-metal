# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer for ttnn.embedding L1 overflow from GR00T fancy indexing.

The GR00T model's CategorySpecificLinear does self.W[cat_ids] which the
compiler lowers to ttnn.embedding.  The state_encoder layer2 weight has
shape (32, 1024, 1536) which gets flattened to (32, 1572864) as the
embedding table.  With a single-index lookup (indices shape [1, 1]),
the op tries to allocate ~3.2 MB of circular buffers on a single core,
exceeding the 1.5 MB L1 limit.

Original error:
  Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)]
  grow to 3249696 B which is beyond max L1 size of 1499136 B
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


GROOT_EMBEDDING_SHAPES = {
    "state_encoder_layer1_W": {"vocab": 32, "hidden": 64 * 1024},
    "state_encoder_layer1_b": {"vocab": 32, "hidden": 1024},
    "state_encoder_layer2_W": {"vocab": 32, "hidden": 1024 * 1536},
    "state_encoder_layer2_b": {"vocab": 32, "hidden": 1536},
    "action_decoder_layer1_W": {"vocab": 32, "hidden": 1024 * 1024},
    "action_decoder_layer1_b": {"vocab": 32, "hidden": 1024},
    "action_decoder_layer2_W": {"vocab": 32, "hidden": 1024 * 32},
    "action_decoder_layer2_b": {"vocab": 32, "hidden": 32},
    "action_encoder_W1": {"vocab": 32, "hidden": 32 * 1536},
    "action_encoder_W2": {"vocab": 32, "hidden": 3072 * 1536},
    "action_encoder_W3": {"vocab": 32, "hidden": 1536 * 1536},
}


@pytest.mark.parametrize(
    "name",
    list(GROOT_EMBEDDING_SHAPES.keys()),
    ids=list(GROOT_EMBEDDING_SHAPES.keys()),
)
def test_groot_embedding_l1(device, name):
    """
    Reproduce ttnn.embedding with the exact shapes from GR00T's
    CategorySpecificLinear layers.  indices=tensor([0]) → single row lookup.
    """
    cfg = GROOT_EMBEDDING_SHAPES[name]
    vocab_size = cfg["vocab"]
    hidden_dim = cfg["hidden"]
    size_mb = vocab_size * hidden_dim * 2 / 1024 / 1024
    print(f"\n[{name}] weight=({vocab_size}, {hidden_dim}), size={size_mb:.1f} MB")

    torch_indices = torch.tensor([[0]], dtype=torch.int32)
    torch_weights = 0.02 * torch.randn(vocab_size, hidden_dim, dtype=torch.bfloat16)
    torch_output = torch.nn.functional.embedding(torch_indices.to(torch.int64), torch_weights)

    indices = ttnn.to_device(
        ttnn.from_torch(torch_indices, dtype=ttnn.uint32),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.embedding(
        indices,
        weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)


def test_groot_embedding_state_encoder_layer2_W(device):
    """
    Focused reproducer for the exact shape that overflows L1.
    Weight shape: (32, 1572864)  — i.e. (32, 1024*1536)
    Indices: [[0]]
    """
    vocab_size = 32
    hidden_dim = 1024 * 1536  # 1,572,864

    torch_indices = torch.tensor([[0]], dtype=torch.int32)
    torch_weights = 0.02 * torch.randn(vocab_size, hidden_dim, dtype=torch.bfloat16)
    torch_output = torch.nn.functional.embedding(torch_indices.to(torch.int64), torch_weights)

    indices = ttnn.to_device(
        ttnn.from_torch(torch_indices, dtype=ttnn.uint32),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.embedding(
        indices,
        weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)
