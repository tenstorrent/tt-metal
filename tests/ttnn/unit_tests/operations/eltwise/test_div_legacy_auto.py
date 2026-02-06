# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

pytestmark = pytest.mark.use_module_device


def test_div_legacy_auto_sharded_block_format(device):
    """
    Test case 1: Sharded block format triggers auto-legacy for DIV.
    Condition: tensor is sharded AND has BFLOAT8_B dtype (block format).
    This should trigger is_legacy_only=true via any_sharded_block_format().
    """
    torch.manual_seed(42)
    # Use shape that works well with sharding: 32 rows per core, 32 cols
    torch_a = torch.rand(1, 1, 128, 32, dtype=torch.float32) + 0.1  # avoid div by zero
    torch_b = torch.rand(1, 1, 128, 32, dtype=torch.float32) + 0.1
    torch_output = torch_a / torch_b

    # Create sharded memory config using helper
    shard_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),  # shard shape per core
        core_grid=ttnn.CoreGrid(y=4, x=1),  # 4 cores for 128 rows (32 rows each)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create BFLOAT8_B tensors first in interleaved, then shard
    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Move to sharded memory
    a_sharded = ttnn.to_memory_config(a, shard_config)
    b_sharded = ttnn.to_memory_config(b, shard_config)

    # DIV without explicit use_legacy - should auto-select legacy
    result = ttnn.div(a_sharded, b_sharded)
    tt_out = ttnn.to_torch(result)

    pcc = ttnn.pearson_correlation_coefficient(torch_output, tt_out)
    assert pcc >= 0.99, f"PCC {pcc} < 0.99 for sharded block format div"


def test_div_binary_ng_width_broadcast_block_format(device):
    """
    Test case 2: Width broadcast with block format routes DIV to binary_ng.

    Condition: BFLOAT8_B dtype with width broadcast (a[-1]=1, b[-1]>1).
    - is_binary_ng_only() returns true (width broadcast + block format + non-ADD/SUB/MUL)
    - Result: Routes to binary_ng (legacy doesn't support width broadcast for DIV)
    """
    torch.manual_seed(42)
    # Width-only broadcast: a has width=1, b has width=64, but same height=32
    torch_a = torch.rand(1, 1, 32, 1, dtype=torch.float32) + 0.1  # width=1
    torch_b = torch.rand(1, 1, 32, 64, dtype=torch.float32) + 0.1  # width=64
    torch_output = torch_a / torch_b  # broadcasts a across width dimension

    # Create BFLOAT8_B tensors (block format) with width broadcast
    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # DIV without explicit use_legacy - routes to binary_ng (not legacy)
    result = ttnn.div(a, b)
    tt_out = ttnn.to_torch(result)

    pcc = ttnn.pearson_correlation_coefficient(torch_output, tt_out)
    assert pcc >= 0.99, f"PCC {pcc} < 0.99 for width broadcast block format div"


def test_div_binary_ng_height_broadcast_block_format(device):
    """
    Test case 3: Height (row) broadcast with block format forces DIV to binary_ng.

    Condition: BFLOAT8_B dtype with height broadcast (a[-2]=1, b[-2]>1).
    - is_binary_ng_only() returns true (any_non_llk_row_broadcasted + non-ADD/SUB/MUL)
    - Result: Routes to binary_ng (not legacy)
    """
    torch.manual_seed(42)
    # Height broadcast pattern: a has height=1, b has height=32
    torch_a = torch.rand(1, 1, 1, 64, dtype=torch.float32) + 0.1  # height=1
    torch_b = torch.rand(1, 1, 32, 64, dtype=torch.float32) + 0.1  # height=32
    torch_output = torch_a / torch_b  # broadcasts a across height dimension

    # Create BFLOAT8_B tensors (block format) with height broadcast
    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # DIV without explicit use_legacy - goes to binary_ng (NOT legacy)
    result = ttnn.div(a, b)
    tt_out = ttnn.to_torch(result)

    pcc = ttnn.pearson_correlation_coefficient(torch_output, tt_out)
    assert pcc >= 0.99, f"PCC {pcc} < 0.99 for height broadcast block format div"
