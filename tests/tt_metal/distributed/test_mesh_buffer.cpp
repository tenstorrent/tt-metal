// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/distributed.hpp>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "env_lib.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/host_buffer.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

static_assert(!std::is_copy_constructible_v<MeshBuffer>, "MeshBuffer should not be copy constructible");
static_assert(!std::is_copy_assignable_v<MeshBuffer>, "MeshBuffer should not be copy assignable");
static_assert(std::is_move_constructible_v<MeshBuffer>, "MeshBuffer should be move constructible");
static_assert(std::is_move_assignable_v<MeshBuffer>, "MeshBuffer should be move assignable");

using MeshBufferTest2x4 = MeshDevice2x4Fixture;
using MeshBufferTest1x2 = MeshDevice1x2Fixture;
using MeshBufferTestSuite = GenericMeshDeviceFixture;

class MeshBufferTest1x2MultiCQ : public MeshDeviceFixtureBase {
protected:
    MeshBufferTest1x2MultiCQ() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 2}, .num_cqs = 2}) {}
};

class ScopedPinnedMemoryCacheLimit {
public:
    explicit ScopedPinnedMemoryCacheLimit(size_t limit_bytes) :
        previous_limit_bytes_(
            tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes()) {
        tt::tt_metal::MetalContext::instance().rtoptions().set_pinned_memory_cache_limit_bytes(limit_bytes);
    }

    ~ScopedPinnedMemoryCacheLimit() {
        tt::tt_metal::MetalContext::instance().rtoptions().set_pinned_memory_cache_limit_bytes(previous_limit_bytes_);
    }

private:
    size_t previous_limit_bytes_;
};

struct DeviceLocalShardedBufferTestConfig {
    Shape2D num_pages_per_core;
    Shape2D num_cores;
    Shape2D page_shape;
    uint32_t element_size = 1;
    TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    Shape2D tensor2d_shape_in_pages() const {
        return {num_pages_per_core.height() * num_cores.height(), num_pages_per_core.width() * num_cores.width()};
    }

    uint32_t num_pages() const { return tensor2d_shape_in_pages().height() * tensor2d_shape_in_pages().width(); }

    std::array<uint32_t, 2> shard_shape() const {
        return {num_pages_per_core.height() * page_shape.height(), num_pages_per_core.width() * page_shape.width()};
    }

    CoreRangeSet shard_grid() const {
        return CoreRangeSet(std::set<CoreRange>(
            {CoreRange(CoreCoord(0, 0), CoreCoord(this->num_cores.height() - 1, this->num_cores.width() - 1))}));
    }

    uint32_t page_size() const { return page_shape.height() * page_shape.width() * element_size; }

    ShardSpecBuffer shard_parameters() const {
        return ShardSpecBuffer(
            this->shard_grid(),
            this->shard_shape(),
            this->shard_orientation,
            this->page_shape,
            this->tensor2d_shape_in_pages());
    }
};

std::vector<uint32_t> expected_shard_host_layout_after_filtered_write(
    Buffer& shard_buffer,
    uint32_t page_size_bytes,
    uint32_t sentinel_value,
    uint32_t new_value,
    const CoreRangeSet& filter) {
    const auto mapping = shard_buffer.get_buffer_page_mapping();
    const uint32_t words_per_page = page_size_bytes / sizeof(uint32_t);
    const uint32_t num_u32 = static_cast<uint32_t>(shard_buffer.size() / sizeof(uint32_t));
    std::vector<uint32_t> expected(num_u32, sentinel_value);
    for (auto it = mapping->begin(); it != mapping->end(); ++it) {
        const auto mp = *it;
        const CoreCoord& core = mapping->all_cores.at(mp.core_id);
        const uint32_t v = (filter.empty() || !filter.contains(core)) ? sentinel_value : new_value;
        const size_t base = static_cast<size_t>(mp.host_page) * words_per_page;
        std::fill(expected.begin() + base, expected.begin() + base + words_per_page, v);
    }
    return expected;
}

// Helper function to print detailed mismatch information between two uint32_t vectors
template <typename T>
void print_vector_mismatch_details(const std::vector<T>& dst_aligned, const std::vector<T>& src) {
    if (dst_aligned == src) {
        return;  // No mismatch, nothing to print
    }

    size_t size = std::min(dst_aligned.size(), src.size());
    std::vector<std::pair<size_t, size_t>> mismatch_ranges;  // pairs of (start_offset, count)

    // Find all ranges of mismatches
    size_t i = 0;
    while (i < size) {
        if (dst_aligned[i] != src[i]) {
            size_t start = i;
            size_t count = 0;
            while (i < size && dst_aligned[i] != src[i]) {
                count++;
                i++;
            }
            mismatch_ranges.emplace_back(start, count);
        } else {
            i++;
        }
    }

    // Print information for each mismatch range
    std::cout << "\n=== DATA MISMATCH DETECTED ===\n";
    std::cout << "Total elements: " << size << "\n";
    std::cout << "Number of mismatch ranges: " << mismatch_ranges.size() << "\n\n";

    for (const auto& [start_offset, mismatch_count] : mismatch_ranges) {
        std::cout << "Mismatch Range: offset=" << start_offset << ", count=" << mismatch_count << "\n";

        // Print header
        std::cout << "Offset   ";
        for (int j = 0; j < 8; j++) {
            std::cout << " | Dst:" << j << "    Src:" << j << "   ";
        }
        std::cout << "|\n";
        std::cout << std::string(190, '-') << "\n";

        // Print up to 128 pairs (or the actual count if less)
        size_t pairs_to_print = std::min(mismatch_count, size_t(128));

        for (size_t offset = 0; offset < pairs_to_print; offset += 8) {
            size_t pairs_in_line = std::min(size_t(8), pairs_to_print - offset);

            // Print offset for this line
            std::cout << std::setw(8) << std::left << (start_offset + offset) << " ";

            // Print 8 pairs (or fewer if at the end)
            for (size_t j = 0; j < pairs_in_line; j++) {
                size_t idx = start_offset + offset + j;
                std::cout << " | 0x" << std::hex << std::setfill('0') << std::setw(8) << dst_aligned[idx] << " 0x"
                          << std::hex << std::setfill('0') << std::setw(8) << src[idx];
                std::cout << std::dec << std::setfill(' ');
            }
            // Fill remaining columns if less than 8 pairs in this line
            for (size_t j = pairs_in_line; j < 8; j++) {
                std::cout << " |                      ";
            }
            std::cout << " |\n";
        }

        if (mismatch_count > 128) {
            std::cout << "... (" << (mismatch_count - 128) << " more mismatched values not shown)\n";
        }
        std::cout << "\n";
    }

    if (dst_aligned.size() != src.size()) {
        std::cout << "WARNING: Size mismatch - dst_aligned.size()=" << dst_aligned.size()
                  << ", src.size()=" << src.size() << "\n";
    }
}

TEST_F(MeshBufferTest2x4, ShardedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ShardedBufferConfig buffer_config{
        .global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 32}};
    EXPECT_EQ(buffer_config.compute_datum_size_bytes(), 2);
    auto sharded_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(sharded_buffer->size(), 16 << 10);
    EXPECT_EQ(sharded_buffer->global_layout(), MeshBufferLayout::SHARDED);
    EXPECT_EQ(sharded_buffer->device_local_size(), 2 << 10);
}

TEST_F(MeshBufferTest2x4, ReplicatedBufferInitialization) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    auto replicated_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    EXPECT_EQ(replicated_buffer->size(), 16 << 10);
    EXPECT_EQ(replicated_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(replicated_buffer->device_local_size(), 16 << 10);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteMeshBufferValidSrcSize) {
    constexpr size_t buffer_size = 16;

    const DeviceLocalBufferConfig device_local_config{
        .page_size = buffer_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    const ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto mesh_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    std::vector<uint8_t> small_src_vec(buffer_size / 2, 0);
    std::vector<uint8_t> exact_src_vec(buffer_size, 0);
    std::vector<uint8_t> large_src_vec(buffer_size * 2, 0);

    EXPECT_THROW(
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, small_src_vec), std::exception);
    EXPECT_NO_THROW(EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, exact_src_vec, true));
    EXPECT_NO_THROW(EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, large_src_vec, true));
}

TEST_F(MeshBufferTest2x4, Deallocation) {
    // Verify that a buffer is deallocated on the MeshDevice when it goes
    // out of scope on host. Create a buffer with a certain config in limited
    // scope. Record its address. Create another buffer with the same config
    // outside the scope. Verify that addresses match.
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};

    // Fetch an address that the allocator provides on first allocation.
    const uint32_t expected_address = [&]() {
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
        EXPECT_TRUE(buffer->is_allocated());
        return buffer->address();
    }();

    // Test that creating and deallocating a MeshBuffer frees the address.
    auto buffer1 = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    EXPECT_TRUE(buffer1->is_allocated());
    EXPECT_EQ(buffer1->address(), expected_address);

    buffer1->deallocate();
    EXPECT_FALSE(buffer1->is_allocated());

    auto buffer2 = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    EXPECT_TRUE(buffer2->is_allocated());
    EXPECT_EQ(buffer2->address(), expected_address);

    // Test deallocation of the view also works.
    auto buffer_view = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get(), buffer2->address());
    EXPECT_TRUE(buffer_view->is_allocated());
    EXPECT_EQ(buffer_view->address(), expected_address);

    buffer_view->deallocate();
    EXPECT_FALSE(buffer_view->is_allocated());
}

TEST(MeshBufferTest, DeallocationWithoutMeshDevice) {
    // Repeated device init takes very long on TG. Lower the number of iterations.
    int iterations = 100;
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        iterations = 10;
    }

    for (int i = 0; i < iterations; i++) {
        MeshDeviceConfig config(MeshShape(1, 1));
        auto mesh_device =
            MeshDevice::create(config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);

        const DeviceLocalBufferConfig device_local_config{
            .page_size = 2048, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const ReplicatedBufferConfig buffer_config{.size = 2048};
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

        mesh_device.reset();
    }
}

TEST(MeshBufferTest, DeallocationWithMeshDeviceClosed) {
    // Repeated device init takes very long on TG. Lower the number of iterations.
    int iterations = 100;
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        iterations = 10;
    }

    for (int i = 0; i < iterations; i++) {
        MeshDeviceConfig config(MeshShape(1, 1));
        auto mesh_device =
            MeshDevice::create(config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);

        const DeviceLocalBufferConfig device_local_config{
            .page_size = 2048, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const ReplicatedBufferConfig buffer_config{.size = 2048};
        auto buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

        mesh_device->close();
    }
}

TEST_F(MeshBufferTest2x4, GetDeviceBuffer) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    auto replicated_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = 16 << 10}, device_local_config, mesh_device_.get());

    // Out of bounds coordinates.
    EXPECT_ANY_THROW(replicated_buffer->get_device_buffer(MeshCoordinate{2, 4}));

    EXPECT_NO_THROW(replicated_buffer->get_device_buffer(MeshCoordinate{1, 3}));
}

TEST_F(MeshBufferTestSuite, MoveConstructor) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    auto original_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    const auto original_address = original_buffer->address();
    const auto original_size = original_buffer->size();
    const auto original_device_local_size = original_buffer->device_local_size();

    EXPECT_TRUE(original_buffer->is_allocated());

    MeshBuffer moved_buffer(std::move(*original_buffer));

    EXPECT_TRUE(moved_buffer.is_allocated());
    EXPECT_EQ(moved_buffer.address(), original_address);
    EXPECT_EQ(moved_buffer.size(), original_size);
    EXPECT_EQ(moved_buffer.device_local_size(), original_device_local_size);

    EXPECT_FALSE(original_buffer->is_allocated());
}

TEST_F(MeshBufferTestSuite, MoveAssignment) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ReplicatedBufferConfig buffer_config{.size = 16 << 10};
    auto source_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());

    const auto source_address = source_buffer->address();
    const auto source_size = source_buffer->size();
    const auto source_device_local_size = source_buffer->device_local_size();

    EXPECT_TRUE(source_buffer->is_allocated());

    const ReplicatedBufferConfig target_buffer_config{.size = 8 << 10};
    auto target_buffer = MeshBuffer::create(target_buffer_config, device_local_config, mesh_device_.get());
    const auto target_original_address = target_buffer->address();

    EXPECT_TRUE(target_buffer->is_allocated());
    EXPECT_NE(target_buffer->address(), source_address);

    *target_buffer = std::move(*source_buffer);

    EXPECT_TRUE(target_buffer->is_allocated());
    EXPECT_EQ(target_buffer->address(), source_address);
    EXPECT_EQ(target_buffer->size(), source_size);
    EXPECT_EQ(target_buffer->device_local_size(), source_device_local_size);

    EXPECT_FALSE(source_buffer->is_allocated());

    auto new_buffer = MeshBuffer::create(target_buffer_config, device_local_config, mesh_device_.get());
    EXPECT_EQ(new_buffer->address(), target_original_address);
}

class DeviceLocalMeshBufferShardingTest
    : public MeshBufferTest2x4,
      public testing::WithParamInterface<
          std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>, TensorMemoryLayout>> {};

TEST_P(DeviceLocalMeshBufferShardingTest, ShardingTest) {
    auto [num_pages_per_core, page_shape, shard_strategy] = GetParam();
    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();

    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = num_pages_per_core,
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = page_shape,
        .mem_config = shard_strategy};
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
        .bottom_up = false};

    uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    ReplicatedBufferConfig global_buffer_config{
        .size = buf_size,
    };

    auto buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    std::vector<uint32_t> src_vec(buf_size / sizeof(uint32_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, MeshCoordinate(logical_y, logical_x));
        }
    }

    for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));
            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DeviceLocalMeshBufferShardingTests,
    DeviceLocalMeshBufferShardingTest,
    ::testing::Combine(
        // num_pages_per_core
        ::testing::Values(
            std::array<uint32_t, 2>{1, 1},
            std::array<uint32_t, 2>{3, 137},
            std::array<uint32_t, 2>{67, 4},
            std::array<uint32_t, 2>{7, 11},
            std::array<uint32_t, 2>{2, 2}),
        // page_shape
        ::testing::Values(
            std::array<uint32_t, 2>{1, 1024},
            std::array<uint32_t, 2>{1, 2048},
            std::array<uint32_t, 2>{1, 4},
            std::array<uint32_t, 2>{32, 32},
            std::array<uint32_t, 2>{1, 120}),
        // shard_strategy
        ::testing::Values(
            TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::BLOCK_SHARDED)));

TEST_F(MeshBufferTest2x4, SweepShardAndConcat) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    std::vector<Shape2D> global_buffer_shapes = {
        {64, 128}, {128, 128}, {32, 1024}, {1024, 32}, {512, 64}, {2048, 2048}};
    std::vector<Shape2D> shard_shapes = {{32, 32}, {32, 64}, {32, 128}, {128, 32}, {128, 32}, {512, 1024}};
    for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
        for (int i = 0; i < global_buffer_shapes.size(); i++) {
            Shape2D global_buffer_shape = global_buffer_shapes[i];
            Shape2D shard_shape = shard_shapes[i];

            uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

            ShardedBufferConfig sharded_config{
                .global_size = global_buffer_size,
                .global_buffer_shape = global_buffer_shape,
                .shard_shape = shard_shape,
                .shard_orientation = shard_orientation,
            };

            auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
            std::vector<uint32_t> src_vec =
                std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
            std::iota(src_vec.begin(), src_vec.end(), 0);
            EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
            std::vector<uint32_t> dst_vec = {};
            EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer);

            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

TEST_F(MeshBufferTestSuite, ConfigValidation) {
    const DeviceLocalBufferConfig device_local_config{
        .page_size = 1024, .buffer_type = BufferType::DRAM, .bottom_up = false};

    // Unaligned shard shape
    EXPECT_ANY_THROW(MeshBuffer::create(
        ShardedBufferConfig{.global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {32, 120}},
        device_local_config,
        mesh_device_.get()));

    // Number of shards exceeds the number of devices
    EXPECT_ANY_THROW(MeshBuffer::create(
        ShardedBufferConfig{.global_size = 16 << 10, .global_buffer_shape = {64, 128}, .shard_shape = {16, 16}},
        device_local_config,
        mesh_device_.get()));

    // Buffer with a global shape of 64x128 distributed across a 2x4 or 2x1 Mesh.
    auto buffer = MeshBuffer::create(
        ShardedBufferConfig{
            .global_size = 16 << 10,
            .global_buffer_shape = {128, 256},
            .shard_shape = {128 / mesh_device_->num_rows(), 256 / mesh_device_->num_cols()}},
        device_local_config,
        mesh_device_.get());
}

TEST_F(MeshBufferTestSuite, InterleavedShardsReadWrite) {
    constexpr uint32_t NUM_ITERS = 100;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    for ([[maybe_unused]] auto buffer_type : {BufferType::L1, BufferType::DRAM}) {
        DeviceLocalBufferConfig per_device_buffer_config{
            .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};

        std::uniform_int_distribution<int> gen_num_tiles(1, 1024);
        std::mt19937 rng(seed);
        for (int i = 0; i < NUM_ITERS; i++) {
            uint32_t num_random_tiles = gen_num_tiles(rng);
            ReplicatedBufferConfig global_buffer_config = {
                .size = num_random_tiles * single_tile_size,
            };

            std::shared_ptr<MeshBuffer> buf =
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

            std::vector<uint32_t> src_vec(num_random_tiles * single_tile_size / sizeof(uint32_t), 0);
            std::iota(src_vec.begin(), src_vec.end(), i);
            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    WriteShard(mesh_device_->mesh_command_queue(), buf, src_vec, MeshCoordinate(logical_y, logical_x));
                }
            }

            for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
                for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                    if (!mesh_device_->impl().is_local(MeshCoordinate(logical_y, logical_x))) {
                        continue;
                    }
                    std::vector<uint32_t> dst_vec = {};
                    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, MeshCoordinate(logical_y, logical_x));
                    EXPECT_EQ(dst_vec, src_vec);
                }
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, RowMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{64, 256}, {128, 128}, {256, 2048}, {32, 512}, {512, 1024}};

    for (auto global_buffer_shape : global_buffer_shapes) {
        Shape2D shard_shape = {0, global_buffer_shape.width() / mesh_device_->num_cols()};
        // Mesh-Level Sharding Parameters for the MeshBufferView that will be read to verify correctness
        Shape2D global_buffer_read_shape = {
            global_buffer_shape.height() * mesh_device_->num_rows(), global_buffer_shape.width()};
        Shape2D shard_read_shape = {
            global_buffer_shape.height(), global_buffer_shape.width() / mesh_device_->num_cols()};

        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
        auto shard_orientation = ShardOrientation::ROW_MAJOR;

        ShardedBufferConfig sharded_config{
            .global_size = global_buffer_size,
            .global_buffer_shape = global_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = shard_orientation,
        };
        // Initialize the ShardedBufferConfig for reading and verifying replicated data
        ShardedBufferConfig sharded_read_view_config{
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint32_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = shard_orientation};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
        std::vector<uint32_t> dst_vec =
            std::vector<uint32_t>(global_buffer_read_shape.height() * global_buffer_read_shape.width(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);

        for (int i = 0; i < dst_vec.size(); i++) {
            EXPECT_EQ(dst_vec[i], i % (src_vec.size()));
        }
    }
}

TEST_F(MeshBufferTestSuite, ColMajorShardingAndReplication) {
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    std::vector<Shape2D> global_buffer_shapes = {{256, 64}, {1024, 1024}, {128, 32}, {512, 64}, {2048, 256}};

    for (auto global_buffer_shape : global_buffer_shapes) {
        Shape2D shard_shape = {global_buffer_shape.height() / mesh_device_->num_rows(), 0};
        uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
        Shape2D global_buffer_read_shape = {
            global_buffer_shape.height(), global_buffer_shape.width() * mesh_device_->num_cols()};
        Shape2D shard_read_shape = {
            global_buffer_shape.height() / mesh_device_->num_rows(), global_buffer_shape.width()};

        ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;

        ShardedBufferConfig sharded_config{
            .global_size = global_buffer_size,
            .global_buffer_shape = global_buffer_shape,
            .shard_shape = shard_shape,
            .shard_orientation = shard_orientation,
        };

        ShardedBufferConfig sharded_read_view_config{
            .global_size = global_buffer_read_shape.height() * global_buffer_read_shape.width() * sizeof(uint32_t),
            .global_buffer_shape = global_buffer_read_shape,
            .shard_shape = shard_read_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR};

        auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), 0);

        auto mesh_buffer_read_view = MeshBuffer::create(
            sharded_read_view_config, per_device_buffer_config, mesh_device_.get(), mesh_buffer->address());

        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
        std::vector<uint32_t> dst_vec =
            std::vector<uint32_t>(global_buffer_read_shape.height() * global_buffer_read_shape.width(), 0);
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer_read_view);
        for (int i = 0; i < dst_vec.size(); i++) {
            EXPECT_EQ(
                (i / global_buffer_read_shape.width()) * global_buffer_shape.width() + i % global_buffer_shape.width(),
                dst_vec[i]);
        }
    }
}

TEST_F(MeshBufferTestSuite, MultiShardReadWrite) {
    constexpr uint32_t NUM_ITERS = 50;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    std::uniform_int_distribution<int> gen_num_datums(32, 128);
    std::mt19937 rng(seed);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

    uint32_t rows = mesh_device_->num_rows();
    uint32_t cols = mesh_device_->num_cols();
    uint32_t num_devices = rows * cols;

    for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
        for (int i = 0; i < NUM_ITERS; i++) {
            Shape2D global_buffer_shape = {
                gen_num_datums(rng) * constants::TILE_HEIGHT * rows,
                gen_num_datums(rng) * constants::TILE_WIDTH * cols};
            Shape2D shard_shape = {global_buffer_shape.height() / rows, global_buffer_shape.width() / cols};
            uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
            ShardedBufferConfig sharded_config{
                .global_size = global_buffer_size,
                .global_buffer_shape = global_buffer_shape,
                .shard_shape = shard_shape,
                .shard_orientation = shard_orientation,
            };
            auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

            std::vector<uint32_t> src_vec =
                std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
            std::iota(src_vec.begin(), src_vec.end(), i);
            std::unordered_map<distributed::MeshCoordinate, std::vector<uint32_t>> dst_vec = {};
            std::vector<distributed::ShardDataTransfer> input_shards = {};
            std::vector<distributed::ShardDataTransfer> output_shards = {};

            for (const auto& coord : coord_range) {
                input_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
            }
            for (const auto& coord : coord_range) {
                dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                output_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
            }

            mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
            mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

            for (auto& dst : dst_vec) {
                EXPECT_EQ(dst.second, src_vec);
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, MultiShardReadWriteMultiThread) {
    constexpr uint32_t NUM_ITERS = 50;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", 0);
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    std::vector<std::thread> threads;
    for (int thread_idx = 0; thread_idx < 2; thread_idx += 1) {
        threads.push_back(std::thread([&]() {
            std::uniform_int_distribution<int> gen_num_datums(32, 128);
            std::mt19937 rng(seed);

            DeviceLocalBufferConfig per_device_buffer_config{
                .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

            distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

            uint32_t rows = mesh_device_->num_rows();
            uint32_t cols = mesh_device_->num_cols();
            uint32_t num_devices = rows * cols;

            for (auto shard_orientation : {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
                for (int i = 0; i < NUM_ITERS; i++) {
                    Shape2D global_buffer_shape = {
                        gen_num_datums(rng) * constants::TILE_HEIGHT * rows,
                        gen_num_datums(rng) * constants::TILE_WIDTH * cols};
                    Shape2D shard_shape = {global_buffer_shape.height() / rows, global_buffer_shape.width() / cols};
                    uint32_t global_buffer_size =
                        global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);
                    ShardedBufferConfig sharded_config{
                        .global_size = global_buffer_size,
                        .global_buffer_shape = global_buffer_shape,
                        .shard_shape = shard_shape,
                        .shard_orientation = shard_orientation,
                    };
                    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

                    std::vector<uint32_t> src_vec =
                        std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                    std::iota(src_vec.begin(), src_vec.end(), i);
                    std::unordered_map<distributed::MeshCoordinate, std::vector<uint32_t>> dst_vec = {};
                    std::vector<distributed::ShardDataTransfer> input_shards = {};
                    std::vector<distributed::ShardDataTransfer> output_shards = {};

                    for (const auto& coord : coord_range) {
                        input_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
                    }
                    for (const auto& coord : coord_range) {
                        dst_vec[coord] = std::vector<uint32_t>(global_buffer_size / num_devices / sizeof(uint32_t), 0);
                        output_shards.push_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
                    }

                    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
                    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

                    for (auto& dst : dst_vec) {
                        EXPECT_EQ(dst.second, src_vec);
                    }
                }
            }
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

TEST_F(MeshBufferTestSuite, EnqueueReadShardsWithPinnedMemoryFullRange) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(ttsl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);
    std::shared_ptr<experimental::PinnedMemory> pinned_shared = std::move(pinned_unique);

    // Read back using enqueue_read_shards with pinned_memory populated
    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_aligned))
                             .region(BufferRegion(0, bytes_per_device));
    experimental::ShardDataTransferSetPinnedMemory(read_transfer, pinned_shared);
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(MeshBufferTestSuite, EnqueueReadWithDistributedHostBufferAndPinnedMemory) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(ttsl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_shared = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);

    // Create DistributedHostBuffer and add the HostBuffer as a shard
    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
    distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });

    // Read back using enqueue_read with DistributedHostBuffer
    mesh_device_->mesh_command_queue().enqueue_read(
        mesh_buffer, distributed_host_buffer, std::nullopt, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheUpgradesCoverageForSameHostBuffer) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    if (mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Requires at least two devices in the mesh";
        return;
    }

    constexpr int device_read_align = 64;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be equal to PCIE read alignment: " << hal.get_read_alignment(HalMemType::HOST)
        << std::endl;

    const uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const auto num_words = single_tile_size / sizeof(uint32_t);
    auto src =
        std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(num_words, 0);
    std::iota(src->begin(), src->end(), 0);

    HostBuffer host_buffer(ttsl::Span<uint32_t>(src->data(), num_words), MemoryPin(src));

    const auto first_coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto single_coord_range = MeshCoordinateRangeSet(MeshCoordinateRange(first_coord, first_coord));
    auto single_coord_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, single_coord_range, host_buffer, true);
    ASSERT_TRUE(single_coord_pin);
    EXPECT_EQ(single_coord_pin->get_device_ids().size(), 1);
    single_coord_pin.reset();

    const auto full_range = MeshCoordinateRangeSet(MeshCoordinateRange(mesh_device_->shape()));
    auto full_mesh_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, full_range, host_buffer, true);
    ASSERT_TRUE(full_mesh_pin);
    EXPECT_EQ(full_mesh_pin->get_device_ids().size(), mesh_device_->num_devices());

    auto repeated_full_mesh_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, full_range, host_buffer, true);
    ASSERT_TRUE(repeated_full_mesh_pin);
    EXPECT_EQ(repeated_full_mesh_pin.get(), full_mesh_pin.get());
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheReadOnlyRequestReusesReadWriteMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    ASSERT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    EXPECT_EQ(read_only.get(), read_write.get());
    EXPECT_EQ(read_only->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheCreatesBestSupportedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));

    auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
        *mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned);
    const auto expected_access = pinning_params.supports_read_only ? experimental::PinnedMemoryDeviceAccess::ReadOnly
                                                                   : experimental::PinnedMemoryDeviceAccess::ReadWrite;
    EXPECT_EQ(pinned->get_device_access(), expected_access);
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheReadWriteRequestReplacesUnreferencedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    std::weak_ptr<experimental::PinnedMemory> old_mapping = read_only;
    read_only.reset();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    EXPECT_TRUE(old_mapping.expired());
    EXPECT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheReadWriteRequestRejectsHeldReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    EXPECT_EQ(read_write, nullptr);
}

TEST_F(MeshBufferTestSuite, PinnedMemoryCacheEvictsOldestEntryToStayWithinLimit) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    if (pinning_params.max_pins < 2) {
        GTEST_SKIP() << "Requires enough pin slots to keep two cache entries resident";
        return;
    }

    constexpr int device_read_align = 64;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be equal to PCIE read alignment: " << hal.get_read_alignment(HalMemType::HOST)
        << std::endl;

    const uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const auto num_words = single_tile_size / sizeof(uint32_t);
    using AlignedVector = std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>;
    auto src0 = std::make_shared<AlignedVector>(num_words, 0);
    auto src1 = std::make_shared<AlignedVector>(num_words, 1);
    auto src2 = std::make_shared<AlignedVector>(num_words, 2);

    HostBuffer host_buffer0(ttsl::Span<uint32_t>(src0->data(), num_words), MemoryPin(src0));
    HostBuffer host_buffer1(ttsl::Span<uint32_t>(src1->data(), num_words), MemoryPin(src1));
    HostBuffer host_buffer2(ttsl::Span<uint32_t>(src2->data(), num_words), MemoryPin(src2));

    const auto first_coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto single_coord_range = MeshCoordinateRangeSet(MeshCoordinateRange(first_coord, first_coord));
    ScopedPinnedMemoryCacheLimit cache_limit(2 * single_tile_size);

    auto first_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, single_coord_range, host_buffer0, true);
    ASSERT_TRUE(first_pin);
    std::weak_ptr<experimental::PinnedMemory> first_weak = first_pin;
    first_pin.reset();

    auto second_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, single_coord_range, host_buffer1, true);
    ASSERT_TRUE(second_pin);
    std::weak_ptr<experimental::PinnedMemory> second_weak = second_pin;
    second_pin.reset();

    auto third_pin =
        experimental::PinnedMemoryCache::instance().try_pin(*mesh_device_, single_coord_range, host_buffer2, true);
    ASSERT_TRUE(third_pin);
    third_pin.reset();

    EXPECT_TRUE(first_weak.expired());
    EXPECT_FALSE(second_weak.expired());
}

TEST_F(MeshBufferTestSuite, EnqueueReadShardsWithPinnedMemoryFullRangeUnaligned) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Prepare source data and write it to a single shard (0,0)
    std::vector<uint8_t> src(bytes_per_device, 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer =
        distributed::ShardDataTransfer{coord}.host_data(src.data()).region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    constexpr size_t unaligned_shift = 3;
    // Prepare destination buffer and pin the entire destination range for the target shard
    auto dst = std::make_shared<vector_aligned<uint8_t>>(bytes_per_device + unaligned_shift, 0);
    uint8_t* dst_ptr_unaligned = dst->data() + unaligned_shift;

    // Create HostBuffer on top of dst
    HostBuffer host_buffer(ttsl::Span<uint8_t>(dst_ptr_unaligned, bytes_per_device), MemoryPin(dst));

    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_shared = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true);

    // Read back using enqueue_read_shards with pinned_memory populated
    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_unaligned))
                             .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint8_t> dst_aligned(dst_ptr_unaligned, dst_ptr_unaligned + bytes_per_device);

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteShardsWithPinnedMemoryFullRange) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be equal to PCIE read alignment: " << hal.get_read_alignment(HalMemType::HOST)
        << std::endl;
    // Prepare write source buffer and pin the entire destination range for the target shard
    auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src->begin(), src->end(), 0);
    // Create HostBuffer on top of src
    HostBuffer host_buffer(ttsl::Span<uint32_t>(src->data(), bytes_per_device / sizeof(uint32_t)), MemoryPin(src));
    // Prepare read destination buffer. Use aigned vector for ease of verification with src above.
    auto dst = std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>(
        bytes_per_device / sizeof(uint32_t), 0);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
        *mesh_device_,
        MeshCoordinateRangeSet(coord_range),
        host_buffer,
        /*map_to_noc=*/true);
    ASSERT_TRUE(pinned_shared);
    for (auto coord : coord_range) {
        log_info(tt::LogTest, "Testing writing from pinned memory to shard at coord {}", coord);
        auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
        distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });
        mesh_device_->mesh_command_queue().enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);

        // Read back via hugepage
        std::fill(dst.begin(), dst.end(), 0);
        auto read_transfer = distributed::ShardDataTransfer{coord}
                                 .host_data(static_cast<void*>(dst.data()))
                                 .region(BufferRegion(0, bytes_per_device));
        mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);
        EXPECT_EQ(*src, dst);
        // Pinned memory should have been used, so locking may block.
        EXPECT_TRUE(pinned_shared->lock_may_block());
    }
}

TEST_F(MeshBufferTestSuite, EnqueueWriteShardsWithPinnedMemoryWaitsOnClose) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be equal to PCIE read alignment: " << hal.get_read_alignment(HalMemType::HOST)
        << std::endl;

    auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src->begin(), src->end(), 0);
    std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>> dst(
        bytes_per_device / sizeof(uint32_t), 0);

    distributed::MeshCoordinate coord(0, 0);
    {
        HostBuffer host_buffer(ttsl::Span<uint32_t>(src->data(), bytes_per_device / sizeof(uint32_t)), MemoryPin(src));
        auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
            *mesh_device_,
            MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord)),
            host_buffer,
            /*map_to_noc=*/true);
        ASSERT_TRUE(pinned_shared);

        auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
        distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });
        mesh_device_->mesh_command_queue().enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);

        EXPECT_TRUE(pinned_shared->lock_may_block());
    }

    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst.data()))
                             .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);
    EXPECT_EQ(*src, dst);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteShardsWithPinnedMemoryFullRangeLargePage) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    // Size must be larger than max_prefetch_cmd_size.
    uint32_t single_tile_size = 2 * 1024 * 1024;

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 12;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be equal to PCIE read alignment: " << hal.get_read_alignment(HalMemType::HOST)
        << std::endl;
    // Prepare write source buffer and pin the entire destination range for the target shard
    auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src->begin(), src->end(), 0);
    // Create HostBuffer on top of src
    HostBuffer host_buffer(ttsl::Span<uint32_t>(src->data(), bytes_per_device / sizeof(uint32_t)), MemoryPin(src));
    // Prepare read destination buffer. Use aigned vector for ease of verification with src above.
    auto dst = std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>(
        bytes_per_device / sizeof(uint32_t), 0);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
        *mesh_device_,
        MeshCoordinateRangeSet(coord_range),
        host_buffer,
        /*map_to_noc=*/true);
    ASSERT_TRUE(pinned_shared);
    for (auto coord : coord_range) {
        log_info(tt::LogTest, "Testing writing from pinned memory to shard at coord {}", coord);
        auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
        distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });

        mesh_device_->mesh_command_queue().enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);

        // Read back via hugepage
        std::fill(dst.begin(), dst.end(), 0);
        auto read_transfer = distributed::ShardDataTransfer{coord}
                                 .host_data(static_cast<void*>(dst.data()))
                                 .region(BufferRegion(0, bytes_per_device));
        mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);
        EXPECT_EQ(*src, dst);
        // Pinned memory should have been used, so locking may block.
        EXPECT_TRUE(pinned_shared->lock_may_block());
    }
}

// Regression test for remote pinned H2D relays. The extra CQ1 write adds remote traffic while CQ0 relays a large
// 16-byte-shifted pinned source, making prefetch_h crediting of partially written downstream pages observable.
TEST_F(MeshBufferTest1x2MultiCQ, EnqueueWriteShardsWithRemotePinnedMemoryAndAlignmentPrefix) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    constexpr uint32_t page_size_bytes = 4096;
    constexpr uint32_t pages_per_device = 128256;
    constexpr uint32_t bytes_per_device = pages_per_device * page_size_bytes;
    constexpr size_t aligned_byte_shift = 16;

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size_bytes, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto traffic_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be a multiple of PCIE read alignment: "
        << hal.get_read_alignment(HalMemType::HOST);
    ASSERT_EQ(aligned_byte_shift % hal.get_alignment(HalMemType::L1), 0u)
        << "Alignment prefix must preserve L1 alignment for pinned relay";

    using AlignedByteVector = std::vector<uint8_t, ttsl::aligned_allocator<uint8_t, device_read_align>>;
    const MeshCoordinate remote_coord(0, 1);
    std::vector<std::shared_ptr<AlignedByteVector>> source_storage;
    std::vector<HostBuffer> source_host_buffers;
    std::vector<std::shared_ptr<tt_metal::experimental::PinnedMemory>> pinned_sources;
    std::vector<distributed::ShardDataTransfer> write_transfers;
    std::optional<size_t> remote_source_index;
    const MeshCoordinateRange coord_range(mesh_device_->shape());
    const std::vector<MeshCoordinate> coords(coord_range.begin(), coord_range.end());
    source_storage.reserve(coords.size());
    source_host_buffers.reserve(coords.size());
    pinned_sources.reserve(coords.size());
    write_transfers.reserve(coords.size());

    for (size_t coord_idx = 0; coord_idx < coords.size(); ++coord_idx) {
        const auto& coord = coords[coord_idx];
        auto src = std::make_shared<AlignedByteVector>(bytes_per_device + aligned_byte_shift, 0);
        uint8_t* src_shifted = src->data() + aligned_byte_shift;
        for (uint32_t i = 0; i < bytes_per_device; ++i) {
            src_shifted[i] = static_cast<uint8_t>((i * 131 + (i / page_size_bytes) + 17 * coord_idx + 0x5a) & 0xff);
        }

        source_storage.push_back(src);
        source_host_buffers.emplace_back(ttsl::Span<uint8_t>(src_shifted, bytes_per_device), MemoryPin(src));
        pinned_sources.push_back(tt_metal::experimental::PinnedMemory::Create(
            *mesh_device_,
            MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord)),
            source_host_buffers.back(),
            /*map_to_noc=*/true));
        ASSERT_TRUE(pinned_sources.back());

        if (coord == remote_coord) {
            const auto target_device_id = mesh_buffer->get_device_buffer(remote_coord)->device()->id();
            auto noc_addr = pinned_sources.back()->get_noc_addr(target_device_id);
            ASSERT_TRUE(noc_addr.has_value());
            if (noc_addr->device_id == target_device_id) {
                GTEST_SKIP() << "Pinned source is local to target device; remote relay path is not exercised";
                return;
            }
            remote_source_index = coord_idx;
        }

        auto write_transfer = distributed::ShardDataTransfer{coord}
                                  .host_data(static_cast<void*>(src_shifted))
                                  .region(BufferRegion(0, bytes_per_device));
        tt_metal::experimental::ShardDataTransferSetPinnedMemory(write_transfer, pinned_sources.back());
        write_transfers.push_back(std::move(write_transfer));
    }
    ASSERT_TRUE(remote_source_index.has_value());

    auto traffic_src = std::make_shared<AlignedByteVector>(bytes_per_device, 0);
    for (uint32_t i = 0; i < bytes_per_device; ++i) {
        (*traffic_src)[i] = static_cast<uint8_t>((i * 29 + (i / page_size_bytes) + 0xa5) & 0xff);
    }
    auto traffic_write = distributed::ShardDataTransfer{remote_coord}
                             .host_data(static_cast<void*>(traffic_src->data()))
                             .region(BufferRegion(0, bytes_per_device));
    auto& traffic_cq = mesh_device_->mesh_command_queue(1);
    traffic_cq.enqueue_write_shards(traffic_buffer, {traffic_write}, /*blocking=*/false);

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, write_transfers, /*blocking=*/false);
    EXPECT_TRUE(pinned_sources.at(remote_source_index.value())->lock_may_block());

    AlignedByteVector dst(bytes_per_device, 0);
    auto read_transfer = distributed::ShardDataTransfer{remote_coord}
                             .host_data(static_cast<void*>(dst.data()))
                             .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);
    traffic_cq.finish();

    const uint8_t* src_shifted = source_storage.at(remote_source_index.value())->data() + aligned_byte_shift;
    auto mismatch = std::mismatch(src_shifted, src_shifted + bytes_per_device, dst.data());
    if (mismatch.first != src_shifted + bytes_per_device) {
        const size_t offset = static_cast<size_t>(mismatch.first - src_shifted);
        FAIL() << "Remote pinned H2D readback mismatch at byte offset " << offset
               << " expected=" << static_cast<uint32_t>(*mismatch.first)
               << " actual=" << static_cast<uint32_t>(*mismatch.second);
    }
}

TEST_F(MeshBufferTestSuite, EnqueueWriteShardsWithPinnedMemoryFullRangeUnaligned) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    // Use a replicated mesh buffer so per-device buffers are interleaved (not sharded)
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    // Make the buffer multiple tiles to exercise multi-page transfers
    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    for (const size_t aligned_byte_shift : {4, 16}) {
        log_info(
            tt::LogTest, "Testing writing from pinned memory to shard with aligned byte shift {}", aligned_byte_shift);

        ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
        auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        constexpr int device_read_align{64};
        ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
            << "Source vector alignment must be equal to PCIE read alignment: "
            << hal.get_read_alignment(HalMemType::HOST) << std::endl;
        // How many words to shift the source buffer by to get it to start an unaligned word
        ASSERT_EQ(aligned_byte_shift % sizeof(uint32_t), 0u);
        const size_t unaligned_word_shift = aligned_byte_shift / sizeof(uint32_t);
        const size_t num_words = (bytes_per_device / sizeof(uint32_t));
        // Prepare write source buffer and pin the entire destination range for the target shard
        auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
            num_words + unaligned_word_shift, 0);

        uint32_t* src_unaligned = src->data() + unaligned_word_shift;
        std::iota(src_unaligned, src_unaligned + num_words, 0);

        // Create a copy of the source vector to make it easy to verify with the destination vector.
        std::vector<uint32_t> src_vector(src_unaligned, src_unaligned + num_words);
        // Create HostBuffer on top of src
        HostBuffer host_buffer(ttsl::Span<uint32_t>(src_unaligned, num_words), MemoryPin(src));
        std::vector<uint32_t> dst(num_words, 0);

        distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
        auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
            *mesh_device_,
            MeshCoordinateRangeSet(coord_range),
            host_buffer,
            /*map_to_noc=*/true);
        ASSERT_TRUE(pinned_shared);
        for (auto coord : coord_range) {
            log_info(tt::LogTest, "Testing writing from pinned memory to shard at coord {}", coord);
            auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
            distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });
            mesh_device_->mesh_command_queue().enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);

            // Read back via hugepage
            std::fill(dst.begin(), dst.end(), 0);
            auto read_transfer = distributed::ShardDataTransfer{coord}
                                     .host_data(static_cast<void*>(dst.data()))
                                     .region(BufferRegion(0, bytes_per_device));
            mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);
            EXPECT_EQ(src_vector, dst);
            if (aligned_byte_shift % 16 == 0) {
                // Pinned memory should have been used, so locking may block.
                EXPECT_TRUE(pinned_shared->lock_may_block());
            }
        }
    }
}

TEST_F(MeshBufferTestSuite, EnqueueProgramAfterPinnedMemoryWriteRerunsCorrectly) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    constexpr CoreCoord logical_core = {0, 0};
    const CoreRangeSet single_core(CoreRange(logical_core, logical_core));
    const MeshCoordinateRange all_devices(mesh_device_->shape());
    const MeshCoordinate test_coord(0, 0);

    const uint32_t single_tile_size = ::tt::tile_size(DataFormat::Float16_b);
    const DeviceLocalBufferConfig tile_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    const ReplicatedBufferConfig tile_mesh_config{.size = single_tile_size};
    auto input_buffer = MeshBuffer::create(tile_mesh_config, tile_buffer_config, mesh_device_.get());
    auto output_buffer = MeshBuffer::create(tile_mesh_config, tile_buffer_config, mesh_device_.get());

    std::vector<uint16_t> input(single_tile_size / sizeof(uint16_t), 1);
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), input_buffer, input, /*blocking=*/true);

    Program program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        single_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, single_core, cb_src0_config);

    constexpr uint32_t semaphore_value = 1;
    const uint32_t scaling_sem_idx = CreateSemaphore(program, single_core, semaphore_value);
    constexpr uint32_t first_add_value = 3;
    constexpr uint32_t second_add_value = 7;
    SetRuntimeArgs(
        program,
        kernel,
        logical_core,
        {input_buffer->address(),
         output_buffer->address(),
         0, /* src_bank_id */
         0, /* dst_bank_id */
         first_add_value,
         constants::TILE_HEIGHT,
         constants::TILE_WIDTH,
         scaling_sem_idx,
         constants::TILE_HEIGHT + 1});

    MeshWorkload workload;
    workload.add_program(all_devices, std::move(program));

    auto read_and_check_output = [&](uint16_t expected_value) {
        std::vector<uint16_t> output;
        ReadShard(mesh_device_->mesh_command_queue(), output, output_buffer, test_coord);
        ASSERT_EQ(output.size(), input.size());
        for (uint16_t value : output) {
            EXPECT_EQ(value, expected_value);
        }
    };

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);
    read_and_check_output(input[0] + first_add_value);

    constexpr uint32_t pinned_write_tiles = 512;
    const uint32_t pinned_write_size = pinned_write_tiles * single_tile_size;
    const DeviceLocalBufferConfig pinned_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    const ReplicatedBufferConfig pinned_mesh_config{.size = pinned_write_size};
    auto pinned_write_buffer = MeshBuffer::create(pinned_mesh_config, pinned_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align = 64;
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be a multiple of PCIE read alignment: "
        << hal.get_read_alignment(HalMemType::HOST);
    auto pinned_src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        pinned_write_size / sizeof(uint32_t), 0);
    std::iota(pinned_src->begin(), pinned_src->end(), 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(pinned_src->data(), pinned_src->size()), MemoryPin(pinned_src));
    auto pinned_memory = tt_metal::experimental::PinnedMemory::Create(
        *mesh_device_, MeshCoordinateRangeSet(all_devices), host_buffer, /*map_to_noc=*/true);
    ASSERT_TRUE(pinned_memory);

    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
    distributed_host_buffer.emplace_shard(test_coord, [&host_buffer]() { return host_buffer; });
    mesh_device_->mesh_command_queue().enqueue_write(pinned_write_buffer, distributed_host_buffer, /*blocking=*/true);

    auto& program_after_first_run = workload.get_programs().at(all_devices);
    auto& rtas = GetRuntimeArgs(program_after_first_run, kernel);
    rtas[logical_core.x][logical_core.y].at(4) = second_add_value;

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);
    read_and_check_output(input[0] + second_add_value);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteDeviceLocalShardedBufferWithPinnedMemory) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();

    // Test configuration - use multiple pages per core and multiple cores to test coalescing
    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = {20, 20},
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = {1, 2048},  // 2048 bytes per page, L1-aligned
        .mem_config = TensorMemoryLayout::HEIGHT_SHARDED};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
        .bottom_up = false};

    uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    ReplicatedBufferConfig global_buffer_config{.size = buf_size};

    auto buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be a multiple of PCIE read alignment: "
        << hal.get_read_alignment(HalMemType::HOST);

    // How many words to shift the source buffer by to get it to start 16 bytes unaligned
    constexpr size_t unaligned_word_shift = 4;  // 16 bytes = 4 words
    const size_t num_words = (buf_size / sizeof(uint32_t));

    // Prepare write source buffer and pin it
    auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        num_words + unaligned_word_shift, 0);

    uint32_t* src_unaligned = src->data() + unaligned_word_shift;
    std::iota(src_unaligned, src_unaligned + num_words, 0);

    // Create a copy of the source vector to make it easy to verify with the destination vector
    std::vector<uint32_t> src_vector(src_unaligned, src_unaligned + num_words);

    // Create HostBuffer on top of unaligned src
    HostBuffer host_buffer(ttsl::Span<uint32_t>(src_unaligned, num_words), MemoryPin(src));

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
        *mesh_device_,
        MeshCoordinateRangeSet(coord_range),
        host_buffer,
        /*map_to_noc=*/true);
    ASSERT_TRUE(pinned_shared);

    for (auto coord : coord_range) {
        log_info(tt::LogTest, "Testing writing from pinned memory to sharded buffer at coord {}", coord);

        // Write using pinned memory
        auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
        std::function<HostBuffer()> produce_buffer = [&host_buffer]() { return host_buffer; };
        distributed_host_buffer.emplace_shard(coord, produce_buffer);
        mesh_device_->mesh_command_queue().enqueue_write(buf, distributed_host_buffer, /*blocking=*/false);
        EXPECT_TRUE(pinned_shared->lock_may_block());

        // Read back and verify
        std::vector<uint32_t> dst_vec = {};
        ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, coord);
        ASSERT_EQ(dst_vec.size(), src_vector.size());
        EXPECT_EQ(dst_vec, src_vector);
    }
}

// Regression coverage for packed pinned writes to sharded buffers with multiple host ranges per core. Each range is
// relayed through one contiguous scratch stream before the NoC writes fan out to L1, so the PCIe padding for later
// ranges must be computed relative to the current scratch stream offset. This uses a host pointer shifted by 16 bytes:
// it is still L1-aligned for the destination writes, but it is not PCIe-read-aligned. If each range is padded
// independently instead of stream-relative, data after the first range is written with the wrong byte offset.
TEST_F(MeshBufferTestSuite, EnqueueWriteDeviceLocalWidthShardedBufferWithPinnedMemoryAndMisalignedRanges) {
    if (!tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();
    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = {8, 8},
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = {1, 2048},
        .mem_config = TensorMemoryLayout::WIDTH_SHARDED};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
        .bottom_up = false};

    const uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    auto buf =
        MeshBuffer::create(ReplicatedBufferConfig{.size = buf_size}, per_device_buffer_config, mesh_device_.get());

    const auto mapping = buf->get_device_buffer(MeshCoordinate(0, 0))->get_buffer_page_mapping();
    const bool has_multi_range_core = std::any_of(
        mapping->core_page_mappings.begin(), mapping->core_page_mappings.end(), [](const auto& core_mappings) {
            return std::any_of(core_mappings.begin(), core_mappings.end(), [](const BufferCorePageMapping& mapping) {
                return mapping.host_ranges.size() > 1;
            });
        });
    ASSERT_TRUE(has_multi_range_core) << "Test must exercise packed pinned writes with multiple host ranges per core";

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    ASSERT_TRUE(device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0)
        << "Source vector alignment must be a multiple of PCIE read alignment: "
        << hal.get_read_alignment(HalMemType::HOST);

    constexpr size_t source_byte_shift = 16;
    static_assert(source_byte_shift % sizeof(uint32_t) == 0);
    ASSERT_EQ(source_byte_shift % hal.get_alignment(HalMemType::L1), 0u);
    ASSERT_NE(source_byte_shift % hal.get_read_alignment(HalMemType::HOST), 0u);
    const size_t source_word_shift = source_byte_shift / sizeof(uint32_t);
    const size_t num_words = buf_size / sizeof(uint32_t);

    auto src = std::make_shared<std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, device_read_align>>>(
        num_words + source_word_shift, 0);
    uint32_t* src_shifted = src->data() + source_word_shift;
    std::iota(src_shifted, src_shifted + num_words, 0);

    std::vector<uint32_t> expected(src_shifted, src_shifted + num_words);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(src_shifted, num_words), MemoryPin(src));

    const MeshCoordinate coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    auto pinned_shared = tt_metal::experimental::PinnedMemory::Create(
        *mesh_device_,
        MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord)),
        host_buffer,
        /*map_to_noc=*/true);
    ASSERT_TRUE(pinned_shared);

    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
    distributed_host_buffer.emplace_shard(coord, [&host_buffer]() { return host_buffer; });
    mesh_device_->mesh_command_queue().enqueue_write(buf, distributed_host_buffer, /*blocking=*/false);
    EXPECT_TRUE(pinned_shared->lock_may_block());

    std::vector<uint32_t> dst;
    ReadShard(mesh_device_->mesh_command_queue(), dst, buf, coord);
    ASSERT_EQ(dst.size(), expected.size());
    EXPECT_EQ(dst, expected);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteDeviceLocalShardedBufferWithCoreFilter) {
    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();
    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = {2, 2},
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = {1, 1024},
        .mem_config = TensorMemoryLayout::HEIGHT_SHARDED};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
        .bottom_up = false};

    uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    ReplicatedBufferConfig global_buffer_config{.size = buf_size};

    auto buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    distributed::MeshCoordinate coord(0, 0);

    constexpr uint32_t k_sentinel = 0x33333333u;
    constexpr uint32_t k_new = 0x44444444u;
    const uint32_t num_u32 = buf_size / sizeof(uint32_t);
    std::vector<uint32_t> sentinel_vec(num_u32, k_sentinel);
    WriteShard(mesh_device_->mesh_command_queue(), buf, sentinel_vec, coord, /*blocking=*/true);

    std::vector<uint32_t> newest(num_u32, k_new);
    CoreRangeSet filter(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    DistributedHostBuffer distributed_host_buffer = DistributedHostBuffer::create(mesh_device_->shape());
    distributed_host_buffer.emplace_shard(coord, [newest]() { return HostBuffer(newest); });
    tt::tt_metal::experimental::core_subset_write::enqueue_write(
        mesh_device_->mesh_command_queue(), *buf, distributed_host_buffer, /*blocking=*/true, filter);

    std::vector<uint32_t> dst_vec;
    ReadShard(mesh_device_->mesh_command_queue(), dst_vec, buf, coord);
    Buffer& shard = *buf->get_device_buffer(coord);
    auto expected =
        expected_shard_host_layout_after_filtered_write(shard, test_config.page_size(), k_sentinel, k_new, filter);
    ASSERT_EQ(dst_vec.size(), expected.size());
    EXPECT_EQ(dst_vec, expected);
}

TEST_F(MeshBufferTestSuite, EnqueueWriteWithNullFilterIsEquivalent) {
    CoreCoord core_grid_size = mesh_device_->compute_with_storage_grid_size();
    DeviceLocalShardedBufferTestConfig test_config{
        .num_pages_per_core = {2, 2},
        .num_cores = {core_grid_size.x, core_grid_size.y},
        .page_shape = {1, 1024},
        .mem_config = TensorMemoryLayout::HEIGHT_SHARDED};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = test_config.page_size(),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(test_config.shard_parameters(), test_config.mem_config),
        .bottom_up = false};

    uint32_t buf_size = test_config.num_pages() * test_config.page_size();
    ReplicatedBufferConfig global_buffer_config{.size = buf_size};

    auto buf_explicit = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto buf_default = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    distributed::MeshCoordinate coord(0, 0);

    const uint32_t num_u32 = buf_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(num_u32);
    std::iota(src_vec.begin(), src_vec.end(), 0u);

    DistributedHostBuffer dhb_a = DistributedHostBuffer::create(mesh_device_->shape());
    dhb_a.emplace_shard(coord, [src_vec]() { return HostBuffer(src_vec); });
    mesh_device_->mesh_command_queue().enqueue_write(buf_explicit, dhb_a, /*blocking=*/true);

    DistributedHostBuffer dhb_b = DistributedHostBuffer::create(mesh_device_->shape());
    dhb_b.emplace_shard(coord, [src_vec]() { return HostBuffer(src_vec); });
    mesh_device_->mesh_command_queue().enqueue_write(buf_default, dhb_b, /*blocking=*/true);

    std::vector<uint32_t> out_a;
    std::vector<uint32_t> out_b;
    ReadShard(mesh_device_->mesh_command_queue(), out_a, buf_explicit, coord);
    ReadShard(mesh_device_->mesh_command_queue(), out_b, buf_default, coord);
    EXPECT_EQ(out_a, out_b);
    EXPECT_EQ(out_a, src_vec);
}

// On a single host all coords are local, so this test exercises the SD sharded buffer
// write/read path but does not trigger the remote-coord crash directly. For that,
// run with TT_MESH_ID=0 TT_MESH_HOST_RANK=0 on T3K with the dual-host mesh descriptor.
class SDMeshBufferFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            GTEST_SKIP() << "Requires TT_METAL_SLOW_DISPATCH_MODE=1";
        }
        const auto system_shape = MetalContext::instance().get_system_mesh().shape();
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(system_shape), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{});
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(SDMeshBufferFixture, ShardedBufferWriteReadRoundtrip) {
    const uint32_t num_rows = mesh_device_->num_rows();
    const uint32_t num_cols = mesh_device_->num_cols();
    const uint32_t shard_w = 1024;

    const Shape2D global_buffer_shape{num_rows, shard_w * num_cols};
    const uint32_t global_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

    const DeviceLocalBufferConfig per_device_config{
        .page_size = shard_w * sizeof(uint32_t), .buffer_type = BufferType::DRAM, .bottom_up = false};

    const ShardedBufferConfig sharded_config{
        .global_size = global_size,
        .global_buffer_shape = global_buffer_shape,
        .shard_shape = {1, shard_w},
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_config, mesh_device_.get());

    std::vector<uint32_t> src(global_buffer_shape.height() * global_buffer_shape.width());
    std::iota(src.begin(), src.end(), 0);

    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src);

    std::vector<uint32_t> dst;
    EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst, mesh_buffer);

    EXPECT_EQ(dst, src);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
