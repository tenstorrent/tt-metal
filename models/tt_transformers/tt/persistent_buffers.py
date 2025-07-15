# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch

import ttnn


# TODO: Cleanup what determines the shapes, does anything else matter?
@dataclass(frozen=True)
class PersistentBuffersConfiguration:
    is_wormhole: bool  # Assuming shapes are different on wormhole and blackhole
    num_devices: int  # Assuming same set of shapes used for a given type of devices
    model_name: str  # Different models have different weights, and therefore different shapes


@dataclass(frozen=True)
class PersistentBufferKey:
    shape: tuple
    dtype: any
    memory_config: any


supported_persistent_buffers_configurations = [
    PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="QwQ-32B",
    ),
    PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen3-32B",
    ),
    PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen2.5-72B",
    ),
    PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Llama-3.1-70B",
    ),
    PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="DeepSeek-R1-Distill-Llama-70B",
    ),
]


def select_and_create_ag_persistent_buffers(mesh_device, persistent_buffers_configuration):
    # The memory config of the persistent output tensor must match the memory config
    # that is expected in the model for the subsequent op after the AG

    assert (
        persistent_buffers_configuration in supported_persistent_buffers_configurations
    ), f"Configuration '{persistent_buffers_configuration}' does not have hardcoded persistent buffers"

    persistent_buffers = {}

    if persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="QwQ-32B",
    ):
        shape = (1, 1, 128, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 320],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 152064)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.L1_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 160],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen3-32B",
    ):
        shape = (1, 1, 128, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 151936)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.L1_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 160],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 5120)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen2.5-72B",
    ):
        shape = (1, 1, 128, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 8192)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 152064)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.L1_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                [32, 1024],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Llama-3.1-70B",
    ):
        shape = (1, 1, 128, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 8192)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 256, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                [32, 1024],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 128256)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.L1_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="DeepSeek-R1-Distill-Llama-70B",
    ):
        shape = (1, 1, 128, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 128, 8192)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 256)
        dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                [32, 1024],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 8192)
        dtype = ttnn.bfloat16
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

        shape = (1, 1, 32, 128256)
        dtype = ttnn.bfloat8_b
        memory_config = ttnn.L1_MEMORY_CONFIG
        persistent_buffer_key = PersistentBufferKey(shape=shape, dtype=dtype, memory_config=memory_config)
        persistent_buffers[persistent_buffer_key] = create_buffer(
            mesh_device=mesh_device, persistent_buffer_key=persistent_buffer_key
        )

    return persistent_buffers


def select_and_create_rs_persistent_buffers(mesh_device, persistent_buffers_configuration):
    # Intermediate persistent buffers for reduce scatter can have any memory config,
    # it does not have to match the memory config of the input tensor or the output tensor

    # The memory config of the persistent output tensor must match the memory config
    # that is expected in the model for the subsequent op after the RS

    # TODO: We are currently using interleaved L1 for all intermediate persistent buffers
    # for reduce scatter. These buffers (or some subset of them) can be changed to be sharded
    # at some point in the future if it's determined that gives a significant perf uplift.

    assert (
        persistent_buffers_configuration in supported_persistent_buffers_configurations
    ), f"Configuration '{persistent_buffers_configuration}' does not have hardcoded persistent buffers"

    persistent_buffers = {}

    if persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="QwQ-32B",
    ):
        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 640)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 320],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 640)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat8_b
        intermediate_shape = (1, 1, 128, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 128, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat8_b
        intermediate_shape = (1, 1, 256, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 256, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 256, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 256, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen3-32B",
    ):
        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 128, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 640)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 640)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat8_b
        intermediate_shape = (1, 1, 128, 5120)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 640)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Qwen2.5-72B",
    ):
        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 1024)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 128, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 1024)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 256, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 256, 1024)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="Llama-3.1-70B",
    ):
        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 1024)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 128, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 1024)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 256, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 256, 1024)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )
    elif persistent_buffers_configuration == PersistentBuffersConfiguration(
        is_wormhole=True,
        num_devices=8,
        model_name="DeepSeek-R1-Distill-Llama-70B",
    ):
        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 32, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 32, 1024)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

        dtype = ttnn.bfloat16
        intermediate_shape = (1, 1, 128, 8192)
        intermediate_memory_config = ttnn.L1_MEMORY_CONFIG
        intermediate_persistent_buffer_key = PersistentBufferKey(
            shape=intermediate_shape, dtype=dtype, memory_config=intermediate_memory_config
        )
        output_shape = (1, 1, 128, 1024)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_persistent_buffer_key = PersistentBufferKey(
            shape=output_shape, dtype=dtype, memory_config=output_memory_config
        )
        persistent_buffers[(intermediate_persistent_buffer_key, output_persistent_buffer_key)] = (
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=intermediate_persistent_buffer_key),
            create_buffer(mesh_device=mesh_device, persistent_buffer_key=output_persistent_buffer_key),
        )

    return persistent_buffers


def create_buffer(mesh_device, persistent_buffer_key):
    return ttnn.from_torch(
        torch.zeros(persistent_buffer_key.shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=persistent_buffer_key.dtype,
        memory_config=persistent_buffer_key.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
