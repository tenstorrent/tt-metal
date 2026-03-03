// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that TensorAccessor defaults to aligned_page_size when page_size is
// omitted from the constructor. The test kernel constructs a TensorAccessor from
// TensorAccessorArgs without providing page_size, then writes accessor.page_size
// to a DRAM output buffer. The host compares this with buffer->aligned_page_size().

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* KERNEL_PATH =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/tensor_accessor/report_page_size.cpp";

constexpr uint32_t OUTPUT_PAGE_SIZE = 32;
constexpr CoreCoord KERNEL_CORE(0, 0);

// Row major: 4 rows x 100 cols of uint16 → 200 bytes per page (intentionally non-aligned)
constexpr uint32_t RM_NUM_COLS = 100;
constexpr uint32_t RM_NUM_ROWS = 4;
constexpr uint32_t RM_ELEMENT_SIZE = 2;
constexpr uint32_t RM_PAGE_SIZE = RM_NUM_COLS * RM_ELEMENT_SIZE;
constexpr uint32_t RM_NUM_PAGES = RM_NUM_ROWS;
constexpr uint32_t RM_TOTAL_SIZE = RM_NUM_PAGES * RM_PAGE_SIZE;

// Tilized: 1 tile of bfloat16 → 2048 bytes per page
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_ELEMENT_SIZE = 2;
constexpr uint32_t TILE_PAGE_SIZE = TILE_H * TILE_W * TILE_ELEMENT_SIZE;
constexpr uint32_t TILE_NUM_PAGES = 1;
constexpr uint32_t TILE_TOTAL_SIZE = TILE_NUM_PAGES * TILE_PAGE_SIZE;

void verify_default_page_size(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Buffer* input_buffer) {
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& prog = workload.get_programs().at(device_range);

    auto output_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = OUTPUT_PAGE_SIZE, .page_size = OUTPUT_PAGE_SIZE, .buffer_type = BufferType::DRAM});

    CircularBufferConfig cb_config =
        CircularBufferConfig(OUTPUT_PAGE_SIZE, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, OUTPUT_PAGE_SIZE);
    CreateCircularBuffer(prog, KERNEL_CORE, cb_config);

    std::vector<uint32_t> compile_time_args;
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    auto kernel = CreateKernel(
        prog,
        KERNEL_PATH,
        KERNEL_CORE,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    SetRuntimeArgs(prog, kernel, KERNEL_CORE, {input_buffer->address(), output_buffer->address()});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result;
    detail::ReadFromBuffer(output_buffer, result);

    uint32_t expected_input = static_cast<uint32_t>(input_buffer->aligned_page_size());
    EXPECT_EQ(result[0], expected_input) << "Input TensorAccessor.page_size should default to aligned_page_size. "
                                         << "Expected: " << expected_input << ", Got: " << result[0]
                                         << ", Buffer page_size: " << input_buffer->page_size();

    uint32_t expected_output = static_cast<uint32_t>(output_buffer->aligned_page_size());
    EXPECT_EQ(result[1], expected_output) << "Output TensorAccessor.page_size should default to aligned_page_size. "
                                          << "Expected: " << expected_output << ", Got: " << result[1]
                                          << ", Buffer page_size: " << output_buffer->page_size();
}

std::shared_ptr<Buffer> create_interleaved_buffer(
    IDevice* device, uint32_t page_size, uint32_t num_pages, BufferType buffer_type) {
    return CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = num_pages * page_size, .page_size = page_size, .buffer_type = buffer_type});
}

std::shared_ptr<Buffer> create_legacy_sharded_buffer(
    IDevice* device,
    uint32_t page_size,
    uint32_t num_pages,
    BufferType buffer_type,
    std::array<uint32_t, 2> shard_shape,
    std::array<uint32_t, 2> page_shape,
    std::array<uint32_t, 2> tensor2d_shape_in_pages) {
    CoreRangeSet shard_grid(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))}));
    ShardSpecBuffer shard_spec(
        shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape_in_pages);

    return CreateBuffer(tt_metal::ShardedBufferConfig{
        .device = device,
        .size = num_pages * page_size,
        .page_size = page_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_spec});
}

struct NdShardedBufferResult {
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
    Buffer* device_buffer;
};

NdShardedBufferResult create_nd_sharded_buffer(
    distributed::MeshDevice* mesh_device,
    uint32_t page_size,
    uint32_t total_size,
    BufferType buffer_type,
    Shape tensor_shape,
    Shape shard_shape,
    Shape2D page_shape) {
    CoreRangeSet core_range(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    auto buffer_dist_spec = BufferDistributionSpec::from_shard_spec(
        std::move(tensor_shape), std::move(shard_shape), page_shape, core_range, ShardOrientation::ROW_MAJOR);

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = buffer_type,
        .sharding_args = BufferShardingArgs(buffer_dist_spec),
    };

    distributed::ReplicatedBufferConfig mesh_config{.size = total_size};
    auto mesh_buffer = distributed::MeshBuffer::create(mesh_config, device_local_config, mesh_device);
    Buffer* device_buffer = mesh_buffer->get_reference_buffer();

    return {mesh_buffer, device_buffer};
}

}  // namespace

namespace tt::tt_metal {

// --- Interleaved ---

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeInterleavedDramRowMajor) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_interleaved_buffer(device, RM_PAGE_SIZE, RM_NUM_PAGES, BufferType::DRAM);
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeInterleavedDramTilized) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_interleaved_buffer(device, TILE_PAGE_SIZE, TILE_NUM_PAGES, BufferType::DRAM);
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeInterleavedL1RowMajor) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_interleaved_buffer(device, RM_PAGE_SIZE, RM_NUM_PAGES, BufferType::L1);
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeInterleavedL1Tilized) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_interleaved_buffer(device, TILE_PAGE_SIZE, TILE_NUM_PAGES, BufferType::L1);
        verify_default_page_size(mesh_device, buffer.get());
    }
}

// --- Legacy Sharded ---

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeLegacyShardedDramRowMajor) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_legacy_sharded_buffer(
            device,
            RM_PAGE_SIZE,
            RM_NUM_PAGES,
            BufferType::DRAM,
            {RM_NUM_ROWS, RM_NUM_COLS},
            {1, RM_NUM_COLS},
            {RM_NUM_PAGES, 1});
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeLegacyShardedDramTilized) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_legacy_sharded_buffer(
            device, TILE_PAGE_SIZE, TILE_NUM_PAGES, BufferType::DRAM, {TILE_H, TILE_W}, {TILE_H, TILE_W}, {1, 1});
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeLegacyShardedL1RowMajor) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_legacy_sharded_buffer(
            device,
            RM_PAGE_SIZE,
            RM_NUM_PAGES,
            BufferType::L1,
            {RM_NUM_ROWS, RM_NUM_COLS},
            {1, RM_NUM_COLS},
            {RM_NUM_PAGES, 1});
        verify_default_page_size(mesh_device, buffer.get());
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeLegacyShardedL1Tilized) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        auto buffer = create_legacy_sharded_buffer(
            device, TILE_PAGE_SIZE, TILE_NUM_PAGES, BufferType::L1, {TILE_H, TILE_W}, {TILE_H, TILE_W}, {1, 1});
        verify_default_page_size(mesh_device, buffer.get());
    }
}

// --- ND Sharded ---

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeNdShardedDramRowMajor) {
    for (auto& mesh_device : devices_) {
        auto result = create_nd_sharded_buffer(
            mesh_device.get(),
            RM_PAGE_SIZE,
            RM_TOTAL_SIZE,
            BufferType::DRAM,
            Shape({RM_NUM_ROWS, RM_NUM_COLS}),
            Shape({RM_NUM_ROWS, RM_NUM_COLS}),
            Shape2D(1, RM_NUM_COLS));
        verify_default_page_size(mesh_device, result.device_buffer);
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeNdShardedDramTilized) {
    for (auto& mesh_device : devices_) {
        auto result = create_nd_sharded_buffer(
            mesh_device.get(),
            TILE_PAGE_SIZE,
            TILE_TOTAL_SIZE,
            BufferType::DRAM,
            Shape({TILE_H, TILE_W}),
            Shape({TILE_H, TILE_W}),
            Shape2D(TILE_H, TILE_W));
        verify_default_page_size(mesh_device, result.device_buffer);
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeNdShardedL1RowMajor) {
    for (auto& mesh_device : devices_) {
        auto result = create_nd_sharded_buffer(
            mesh_device.get(),
            RM_PAGE_SIZE,
            RM_TOTAL_SIZE,
            BufferType::L1,
            Shape({RM_NUM_ROWS, RM_NUM_COLS}),
            Shape({RM_NUM_ROWS, RM_NUM_COLS}),
            Shape2D(1, RM_NUM_COLS));
        verify_default_page_size(mesh_device, result.device_buffer);
    }
}

TEST_F(MeshDispatchFixture, TensixTensorAccessorDefaultPageSizeNdShardedL1Tilized) {
    for (auto& mesh_device : devices_) {
        auto result = create_nd_sharded_buffer(
            mesh_device.get(),
            TILE_PAGE_SIZE,
            TILE_TOTAL_SIZE,
            BufferType::L1,
            Shape({TILE_H, TILE_W}),
            Shape({TILE_H, TILE_W}),
            Shape2D(TILE_H, TILE_W));
        verify_default_page_size(mesh_device, result.device_buffer);
    }
}

}  // namespace tt::tt_metal
