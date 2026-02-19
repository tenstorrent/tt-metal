# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace

DEEPSEEK_MEM_CONFIG_SHAPE_DTYPE_MEM_CONFIG = [
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                (32, 64),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 32, 1, 64),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                (32, 64),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 32, 16, 64),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
                (32, 32),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 1, 32, 896),
        ttnn.float32,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
                (32, 576),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 4, 1, 576),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))}),
                (32, 576),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 4, 128, 576),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
                (32, 128),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        ttnn.MemoryConfig(
            ttnn.BufferType.L1,
            ttnn.NdShardSpec(
                (32, 128),
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        ),
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.L1_MEMORY_CONFIG,
    ),
]


@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("test_config", DEEPSEEK_MEM_CONFIG_SHAPE_DTYPE_MEM_CONFIG)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_interleaved_to_sharded_deepseek(mesh_device, test_config, layout, enable_trace):
    output_mem_config, shape, dtype, input_mem_config = test_config
    torch.manual_seed(0)
    torch_input = torch.rand(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
        dtype=dtype,
    )

    def run_op():
        return ttnn.interleaved_to_sharded(tt_input, output_mem_config)

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_out_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, tt_out_tensor in zip(coords, ttnn.get_device_tensors(tt_out_tensors)):
        if view is not None and not view.is_local(coord):
            continue
        torch_out = ttnn.to_torch(tt_out_tensor)
        eq, output = comp_equal(torch_out, torch_input)
        assert eq, f"Output mismatch between torch and ttnn all_broadcast: {output}"
