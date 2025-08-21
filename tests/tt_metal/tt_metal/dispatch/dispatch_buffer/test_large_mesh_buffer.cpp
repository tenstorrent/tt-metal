// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/distributed.hpp>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "env_lib.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape2d.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/util.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <cstring>
#include "tt_metal/api/tt-metalium/math.hpp"
namespace tt::tt_metal::distributed::test {
namespace {

using MeshBufferTest2x4 = MeshDevice2x4Fixture;
using MeshBufferTestSuite = GenericMeshDeviceFixture;
using ElementType = uint64_t;
constexpr uint32_t ElementSize = sizeof(ElementType);

class LargeMeshBufferTestSuiteBase : public MeshBufferTestSuite {
protected:
    static constexpr uint64_t max_num_elements_ = (8ull << 30) / ElementSize;  // 8GB
    // A large datatype is used to reduce random shuffle time
    inline static std::vector<ElementType> src_vec_;

    LargeMeshBufferTestSuiteBase() {
        if (src_vec_.empty()) {
            log_info(tt::LogTest, "LargeMeshBufferTestSuiteBase ctor: initializing src_vec_");
            src_vec_.resize(max_num_elements_);
            std::iota(src_vec_.begin(), src_vec_.end(), 0);
            // std::shuffle(src_vec_.begin(), src_vec_.end(), std::mt19937(std::random_device{}()));
        }
    }
};

class ReplicatedMeshBufferTestSuite : public LargeMeshBufferTestSuiteBase,
                                      public testing::WithParamInterface<std::tuple<uint64_t, uint32_t>> {};

TEST_P(ReplicatedMeshBufferTestSuite, DRAMReadback) {
    // - REPLICATED layout for writing, SHARDED with ROW_MAJOR for reading
    // - DRAM, bottom up allocation
    auto [tensor_size, page_size] = GetParam();
    // auto num_devices = mesh_device_->num_devices();
    tensor_size = tensor_size * ElementSize;

    const DeviceLocalBufferConfig device_local_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    constexpr auto address = 32;

    // const auto dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    // uint32_t length_adjust = ((std::random_device{}() % page_size) & ~(dram_alignment - 1)) &
    // CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK;
    // tensor_size -= length_adjust;
    log_info(tt::LogTest, "page_size: {}, tensor_size/device (MB): {}", page_size, tensor_size / (1 << 20));

    const ReplicatedBufferConfig buffer_config{.size = tensor_size};

    // Create replicated buffer for writing with specific address 32
    auto mesh_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get(), address);
    // Verify buffer properties
    EXPECT_EQ(mesh_buffer->size(), tensor_size);
    EXPECT_EQ(mesh_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(mesh_buffer->device_local_size(), tensor_size);
    EXPECT_EQ(mesh_buffer->address(), address);
    EXPECT_TRUE(mesh_buffer->is_allocated());

    // Create test data - use uint16_t for easy verification
    auto num_elements = tensor_size / ElementSize;
    ASSERT_TRUE(num_elements <= max_num_elements_);
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(num_elements, 0);
        output_shards.push_back({coord, dst_vec[coord].data()});
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        EXPECT_EQ(dst.second, src_vec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeReplicatedReadback,
    ReplicatedMeshBufferTestSuite,
    ::testing::Combine(
        ::testing::Values(
            (2ull << 30) / ElementSize, (4ull << 30) / ElementSize, (8ull << 30) / ElementSize),  // tensor sizes
        ::testing::Values(1024, 2048, 16 << 10, 1 << 20)                                          // page sizes
        ));

class ShardedMeshBufferTestSuite : public LargeMeshBufferTestSuiteBase,
                                   public testing::WithParamInterface<std::tuple<Shape2D, uint32_t>> {};

TEST_P(ShardedMeshBufferTestSuite, DRAMReadback) {
    // shard_shape: shape on device (elements)
    // page_size: (bytes)
    auto [shard_shape, page_size] = GetParam();
    uint64_t shard_size = shard_shape.height() * shard_shape.width() * ElementSize;
    uint32_t num_pages = static_cast<uint32_t>(shard_size / page_size);
    ASSERT_TRUE(num_pages * page_size == shard_size);

    // Ensure buffer dimensions are divisible by tile dimensions
    ASSERT_TRUE(shard_shape.height() % constants::TILE_HEIGHT == 0);
    ASSERT_TRUE(shard_shape.width() % constants::TILE_WIDTH == 0);

    CoreCoord available_grid_size = mesh_device_->dram_grid_size();
    CoreCoord start(0, 0);
    CoreCoord end(7 /*available_grid_size.x - 1*/, available_grid_size.y - 1);
    CoreRange cores(start, end);
    CoreCoord core_grid_size = cores.grid_size();

    // Determine page shape in elements
    ASSERT_TRUE(page_size > constants::TILE_HEIGHT * constants::TILE_WIDTH / ElementSize);
    uint32_t page_height = page_size / constants::TILE_WIDTH / ElementSize;
    uint32_t page_width = constants::TILE_WIDTH;
    ASSERT_TRUE(page_height * page_width * ElementSize == page_size);
    Shape2D page_shape{page_height, page_width};

    // Define memory bank dimensions (elements) for core grid. A bank must fit whole pages.
    auto num_banks = core_grid_size.x * core_grid_size.y;
    uint32_t num_pages_per_bank = tt::div_up(num_pages, num_banks);
    // Find all factors of num_pages_per_bank
    std::vector<uint32_t> factors{1};
    auto temp = num_pages_per_bank;
    uint32_t f = 2;
    while (f < temp) {
        if (temp % f == 0) {
            factors.push_back(f);
            temp /= f;
        } else {
            f++;
        }
    }
    factors.push_back(temp);
    // Start with last factor vs product of rest
    int split_idx = factors.size() - 1;
    int32_t bank_height = 1, bank_width = num_pages_per_bank, temp_height, temp_width;
    int32_t min_diff = std::abs(bank_width - bank_height);
    while (split_idx > 0) {
        temp_height = std::accumulate(factors.begin(), factors.begin() + split_idx, 1, std::multiplies<uint32_t>());
        temp_width = std::accumulate(factors.begin() + split_idx, factors.end(), 1, std::multiplies<uint32_t>());
        auto temp_diff = std::abs(bank_height - bank_width);
        if (temp_diff > min_diff) {
            break;
        }
        bank_height = temp_height;
        bank_width = temp_width;
        min_diff = temp_diff;
        split_idx--;
    }
    ASSERT_TRUE(bank_height * bank_width == num_pages_per_bank);
    ASSERT_TRUE(bank_height * bank_width == num_pages_per_bank);
    // convert to elements
    bank_height *= page_height;
    bank_width *= page_width;
    Shape2D bank_shape{bank_height, bank_width};

    // Device shard shape in pages
    uint32_t shard_height_pages = shard_shape.height() / page_height;
    uint32_t shard_width_pages = shard_shape.width() / page_width;
    ASSERT_TRUE(shard_height_pages * page_height == shard_shape.height());
    ASSERT_TRUE(shard_width_pages * page_width == shard_shape.width());
    Shape2D shard_device_shape{shard_height_pages, shard_width_pages};

    constexpr auto tensor_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    auto shard_orientation = ShardOrientation::ROW_MAJOR;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer{CoreRangeSet(cores), bank_shape, shard_orientation, page_shape, shard_device_shape},
            tensor_layout),
        .bottom_up = true};

    // shard size in bytes
    log_info(
        tt::LogTest,
        "Core grid size:{}, on device shard size:{} MB, shape:{} pages, {} elements, shape of bank:{} elements, page "
        "shape:{} elements, page_size:{} KB",
        core_grid_size,
        shard_size >> 20,
        shard_device_shape,
        shard_shape,
        bank_shape,
        page_shape,
        page_size >> 10);

    uint32_t device_rows = mesh_device_->num_rows();
    uint32_t device_cols = mesh_device_->num_cols();
    uint32_t num_devices = device_rows * device_cols;
    // tensor (buffer loaded across devices) shape in elements
    Shape2D tensor_shape = {shard_shape.height() * device_rows, shard_shape.width() * device_cols};
    // tensor size in bytes
    uint64_t tensor_size = num_devices * shard_size;
    auto num_elements = tensor_size / num_devices / ElementSize;
    /*ShardedBufferConfig buf_config{
        .global_size = tensor_size,
        .global_buffer_shape = tensor_shape,
        .shard_shape = shard_shape,
        .shard_orientation = shard_orientation,
    };*/
    const ReplicatedBufferConfig buffer_config{.size = tensor_size};
    log_info(
        tt::LogTest,
        "Mesh buffer (tensor) shape:{} elements, tensor size:{} MB, num_elements:{}",
        tensor_shape,
        tensor_size >> 20,
        num_elements);

    // ASSERT_TRUE(buffer_config.compute_datum_size_bytes() == ElementSize);
    auto mesh_buffer = MeshBuffer::create(buffer_config, per_device_buffer_config, mesh_device_.get());

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    ASSERT_TRUE(num_elements <= max_num_elements_);
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);
    std::vector<MeshCommandQueue::ShardDataTransfer> input_shards = {};
    for (auto& coord : coord_range) {
        input_shards.push_back({coord, src_vec.data()});
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<MeshCommandQueue::ShardDataTransfer> output_shards = {};
    for (auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(tensor_size / num_devices / ElementSize, 0);
        output_shards.push_back({coord, dst_vec[coord].data()});
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        EXPECT_EQ(dst.second, src_vec);
        if (false and src_vec != dst.second) {
            std::cout << "Buffer mismatch detected! Printing mismatching values in hex (16 per row):" << std::endl;
            std::cout << "Index: [src] [result]" << std::endl;
            std::cout << "------------------------" << std::endl;

            size_t mismatch_count = 0;
            for (size_t i = 0; i < src_vec.size() && i < dst.second.size(); i++) {
                if (src_vec[i] != dst.second[i]) {
                    if (mismatch_count % 16 == 0) {
                        if (mismatch_count > 0) {
                            std::cout << std::endl;
                        }
                        std::cout << "Row " << (mismatch_count / 16) << ":" << std::endl;
                    }
                    std::cout << "[" << std::hex << std::setw(4) << std::setfill(' ') << i << "]: " << std::setw(9)
                              << std::setfill(' ') << src_vec[i] << "-" << dst.second[i] << ", ";
                    mismatch_count++;
                }
            }
            if (mismatch_count > 0) {
                std::cout << std::endl << "Total mismatches: " << std::dec << mismatch_count << std::endl;
            }
            std::cout << std::dec;  // Reset to decimal output
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeShardedReadback,
    ShardedMeshBufferTestSuite,
    ::testing::Combine(
        // shard_shape (elements), page_size (bytes)
        ::testing::Values(
            Shape2D((1 << 14), (1 << 14)),                // 2 GB with uint64_t
            Shape2D((1 << 14), (1 << 15)),                // 4 GB with uint64_t
            Shape2D((1 << 15), (1 << 15))),               // 8 GB with uint64_t
        ::testing::Values(1024, 4096, 16 << 10, 1 << 20)  // page size
        ));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
