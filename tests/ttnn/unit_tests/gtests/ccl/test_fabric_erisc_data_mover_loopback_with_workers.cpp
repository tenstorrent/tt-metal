
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include <umd/device/types/arch.hpp>

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////               LOCAL CHIP TENSOR READ?WRITE (2 INPUT)
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

TEST(WorkerCclCommandProcessingKernels, DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly0) {
    ttnn::Shape tensor_shape({1, 1, 64, 16384});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly1) {
    ttnn::Shape tensor_shape({1, 1, 64, 128});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly2) {
    ttnn::Shape tensor_shape({1, 1, 64, 8192});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 2;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1},  // first input
        {1, 0},  // read in reverse order
        {1, 0},  // read in non-sequential order
        {0, 1}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}

// Hits issues with input tensor copy-back
TEST(
    WorkerCclCommandProcessingKernels,
    DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly_SmallSweep) {
    std::vector<ttnn::Shape> tensor_shapes = {
        ttnn::Shape({1, 1, 64, 8192}),
        ttnn::Shape({1, 4, 64, 768}),
        ttnn::Shape({4, 1, 64, 768}),
        ttnn::Shape({4, 4, 64, 768}),
        ttnn::Shape({1, 1, 64, 768}),
        ttnn::Shape({5, 3, 64, 768})};

    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const std::vector<size_t> slices_per_stage_sweep = {2, 3, 4};
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<std::vector<size_t>> num_workers_per_stage_sweep = {
        {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};

    std::vector<std::vector<std::vector<size_t>>> worker_chunk_read_order = {
        {{}},
        {
            {0},
            {0},
            {0},
            {0},
        },
        {
            {0, 1},
            {1, 0},
            {1, 0},
            {0, 1},
        },
        {
            {2, 0, 1},
            {1, 0, 2},
            {0, 1, 2},
            {2, 1, 0},
        },
        {
            {0, 1, 2, 3},  // first input
            {3, 2, 1, 0},  // read in reverse order
            {2, 0, 3, 1},  // read in non-sequential order
            {1, 2, 3, 0}   // read in non-sequential order
        }};
    std::vector<std::vector<MemoryConfig>> mem_configs_sweep = {
        {
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        },
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
    };

    for (auto& tensor_shape : tensor_shapes) {
        for (auto& num_workers_per_stage : num_workers_per_stage_sweep) {
            for (size_t slices_per_stage : slices_per_stage_sweep) {
                for (auto& mem_configs : mem_configs_sweep) {
                    log_info(
                        tt::LogTest,
                        "tensor shape {} and workers stage {} slices_per_stage {}",
                        tensor_shape,
                        num_workers_per_stage,
                        slices_per_stage);
                    auto pass = RunPipelinedWorkersTest(

                        tensor_shape,
                        split_dim,

                        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will
                        // be configurable)
                        num_stages,
                        num_workers_per_stage,
                        slices_per_stage,
                        data_format,
                        page_size_bytes,
                        cb_packet_size_in_pages,
                        num_packets_per_cb,
                        layout,

                        worker_chunk_read_order[slices_per_stage],
                        mem_configs);

                    ASSERT_TRUE(pass);
                }
            }
        }
    }
}
