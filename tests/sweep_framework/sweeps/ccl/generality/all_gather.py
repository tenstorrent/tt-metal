# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweeps.ccl.common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()


parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D],
        # TODO this seem to reliably cause hangs, and we can't recover from hangs right now
        #        "fabric_config": [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricConfig.FABRIC_2D],
        "num_links": [1],
        "input_shape": [
            # [1, 1, 32, 32],
            # [1, 1, 32, 1280],
            # [1, 1, 32, 31],
            # [1, 1, 1, 32, 32],
            ##[2, 2, 64, 32],
            ##[2, 2, 1, 32, 64],
            [90, 60],
            # [2, 32, 32],
            # [1, 1, 32, 16384],
            # [1, 1, 1, 2048],  # the following shapes are from training
            # [1, 1, 1, 4096], # https://docs.google.com/spreadsheets/d/18lQ_dJpodMkoDFZjt7TfHdt0cEGsa5GCxxRKDzErGvM/edit?usp=sharing
            # [1, 32, 2048, 8],
            [2, 32, 2048, 8],
            # [1, 32, 2048, 16],
            # start here
            [2, 32, 2048, 16],
            # [1, 32, 4096, 16],
            # [1, 32, 2048, 64],
            ##[2, 32, 2048, 64],
            # [1, 32, 4096, 32],
            ##[8, 32, 4096, 32],
            # [1, 32, 4096, 64],
            ##[8, 32, 4096, 64],
            # [1, 1, 1, 1],
            # [1, 1, 1, 8],
            # [1, 1, 1, 16],
            # [1, 1, 1, 32],
            # [1, 1, 8, 8],
            # [1, 1, 16, 16],
        ],
        "dim": [0, 1, 2, 3, 4],
        "cluster_axis": [0, 1],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "topology": [ttnn.Topology.Linear],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["dim"] >= len(test_vector["input_shape"]):
        return True, "Dim greater than rank"
    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    if (test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT) and (test_vector["input_dtype"] == ttnn.bfloat8_b):
        return True, "Row major not supported for bfloat8_b"
    # invalidated hanging tests for now
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 2048], dim: 1, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.ROW_MAJOR, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 2048]
        and test_vector["dim"] == 1
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 1, 4096], dim: 2, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 1, 4096]
        and test_vector["dim"] == 2
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 8], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (2, 4), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (2, 4)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 2048, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (1, 8), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 32, 4096, 16], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (1, 8)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 32, 4096, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        # and test_vector["input_dtype"] == ttnn.bfloat16
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        # and test_vector["mem_config"] == ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [1, 1, 32, 31], dim: 1, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT8_B, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [1, 1, 32, 31]
        and test_vector["dim"] == 1
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [2, 32, 2048, 8], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [2, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    # mesh_shape: (8, 1), fabric_config: FabricConfig.FABRIC_1D, input_shape: [2, 32, 2048, 16], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [2, 32, 2048, 16]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [2, 32, 2048, 8], dim: 3, cluster_axis: 0, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [2, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 0
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # mesh_shape: (4, 2), fabric_config: FabricConfig.FABRIC_1D, input_shape: [2, 32, 2048, 8], dim: 3, cluster_axis: 1, num_links: 1, input_dtype: DataType.BFLOAT16, layout: Layout.TILE, mem_config: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), num_iters: 1, topology: Topology.Linear
    if (
        test_vector["mesh_shape"] == (4, 2)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [2, 32, 2048, 8]
        and test_vector["dim"] == 3
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"
    # {'mesh_shape': (8, 1), 'fabric_config': <FabricConfig.FABRIC_1D: 1>, 'input_shape': [2, 32, 2048, 16], 'dim': 1, 'cluster_axis': 1, 'num_links': 1, 'input_dtype': <DataType.BFLOAT16: 0>, 'layout': <Layout.TILE: 1>, 'mem_config': MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0), 'num_iters': 1, 'topology': <Topology.Linear: 1>, 'device': None}
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_1D
        and test_vector["input_shape"] == [2, 32, 2048, 16]
        and test_vector["dim"] == 1
        and test_vector["cluster_axis"] == 1
        and test_vector["num_links"] == 1
        and test_vector["layout"] == ttnn.TILE_LAYOUT
        and test_vector["num_iters"] == 1
        and test_vector["topology"] == ttnn.Topology.Linear
    ):
        return True, "hang case"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, layout, device):
    if dtype == ttnn.uint32:
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(input_shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, layout=layout, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(device), device=device
    )

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

    return tt_input, torch_reference


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    dim,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    logger.info("STARTING SWEEP")

    logger.info(vars())
    print("in run function\n")
    with device_context(mesh_shape, fabric_config) as (device, device_err):
        # print the vector being used
        print(
            f"mesh_shape: {mesh_shape}, fabric_config: {fabric_config}, input_shape: {input_shape}, dim: {dim}, cluster_axis: {cluster_axis}, num_links: {num_links}, input_dtype: {input_dtype}, layout: {layout}, mem_config: {mem_config}, num_iters: {num_iters}, topology: {topology}"
        )
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        logger.info("device set up")

        tt_input, torch_reference = _get_tensors(
            input_shape, mesh_shape, dim, cluster_axis, input_dtype, layout, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphore = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    tt_input,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    multi_device_global_semaphore=semaphore,
                    num_links=num_links,
                    memory_config=mem_config,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

            logger.info(f"Done iteration {i}")

        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            logger.info("Bringing tensor back to host")
            tt_output_tensor = ttnn.to_torch(t)
            logger.info("Brought tensor back from host")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, torch_reference)
            else:
                eq, output = comp_pcc(tt_output_tensor, torch_reference)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
            return [(eq, output), e2e_perf]
