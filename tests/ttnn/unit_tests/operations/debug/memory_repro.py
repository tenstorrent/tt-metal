# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _make_l1_tensor_on_core(device, core, num_bytes):
    # bfloat16 (2 B/elem), tile-aligned: shape (1, 1, 32, width) with width % 32 == 0.
    bytes_per_elem = 2
    num_elems = num_bytes // bytes_per_elem
    width = num_elems // 32
    assert width * 32 == num_elems, f"size {num_bytes} not aligned to 32-elem rows"
    assert width % 32 == 0, f"width {width} not tile-aligned"

    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    shard_spec = ttnn.ShardSpec(shard_grid, [32, width], ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.from_torch(
        torch.zeros((1, 1, 32, width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def test_memory_repro(device):
    # 200 KiB L1 tensor sharded on core (0,0) and another on core (1,0).
    # Held for the duration of the op so they contribute to L1 pressure.
    tensor_core_0 = _make_l1_tensor_on_core(device, ttnn.CoreCoord(0, 0), 200 * 1024)
    tensor_core_1 = _make_l1_tensor_on_core(device, ttnn.CoreCoord(1, 0), 200 * 1024)

    input_tensor = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn.memory_repro(input_tensor)
    ttnn.synchronize_device(device)

    del tensor_core_0
    del tensor_core_1
