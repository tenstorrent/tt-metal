// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <hostdevcommon/tensor_accessor/arg_config.hpp>  // tensor_accessor::ArgConfig (RuntimePageSize / IsDram)
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* KERNEL_PATH =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/tensor_accessor/report_page_size.cpp";

constexpr const char* RUNTIME_PAGE_SIZE_KERNEL_PATH =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/tensor_accessor/report_runtime_page_size.cpp";

// A deliberately synthetic page size for the RuntimePageSize test: distinct from any buffer's
// natural aligned_page_size, so a passing assertion can only mean the value came from the common
// runtime arg (not a stale CTA, and not a re-derivation from the bound buffer).
constexpr uint32_t RUNTIME_PAGE_SIZE_SENTINEL = 0xABCD;

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

void verify_runtime_page_size(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Buffer* input_buffer,
    uint32_t sentinel_page_size) {
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

    // Forge the input accessor's compile-time args: the args_config word with the RuntimePageSize
    // bit set, and NO aligned-page-size word (A-collapse). The legacy TensorAccessorArgs(buffer)
    // path never sets this bit -- only the Metal 2.0 resolver does -- so we build the word by hand.
    // The output accessor is appended normally (static); the kernel reads it at the post-A-collapse
    // offset, which only resolves correctly if the input consumed exactly one CTA word.
    std::vector<uint32_t> compile_time_args;
    const uint32_t input_config_word =
        (tensor_accessor::ArgConfig::IsDram | tensor_accessor::ArgConfig::RuntimePageSize).raw();
    compile_time_args.push_back(input_config_word);
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    auto kernel = CreateKernel(
        prog,
        RUNTIME_PAGE_SIZE_KERNEL_PATH,
        KERNEL_CORE,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    SetRuntimeArgs(prog, kernel, KERNEL_CORE, {input_buffer->address(), output_buffer->address()});
    // The input's dynamic page size rides common runtime arg 0 -- the channel the device accessor
    // reads via get_aligned_page_size() when the RuntimePageSize bit is set.
    SetCommonRuntimeArgs(prog, kernel, {sentinel_page_size});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result;
    detail::ReadFromBuffer(output_buffer, result);

    EXPECT_EQ(result[0], sentinel_page_size)
        << "RuntimePageSize input accessor must report the runtime (common-arg) page size, not a "
           "static CTA value. Expected sentinel "
        << sentinel_page_size << ", got " << result[0];

    uint32_t expected_output = static_cast<uint32_t>(output_buffer->aligned_page_size());
    EXPECT_EQ(result[1], expected_output)
        << "Output accessor page size must be intact -- proves the input's A-collapse (one fewer CTA "
           "word) did not shift the output accessor's CTA offset. Expected "
        << expected_output << ", got " << result[1];
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

// --- Runtime page size (RuntimePageSize relaxation, device-side isolation) ---

TEST_F(MeshDispatchFixture, TensixTensorAccessorRuntimePageSizeInterleavedDram) {
    for (auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        // Input is a real interleaved row-major DRAM buffer; its address is passed to the kernel but
        // its page size is NOT -- the accessor must read the page size from the runtime sentinel.
        auto input = create_interleaved_buffer(device, RM_PAGE_SIZE, RM_NUM_PAGES, BufferType::DRAM);
        verify_runtime_page_size(mesh_device, input.get(), RUNTIME_PAGE_SIZE_SENTINEL);
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
