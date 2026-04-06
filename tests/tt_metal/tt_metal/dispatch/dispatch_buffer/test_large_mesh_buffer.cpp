// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/distributed.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <vector>
#include <iomanip>
#include <sstream>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "env_lib.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape2d.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/math.hpp"
#include <enchantum/enchantum.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

// Helper function to detect and print mismatch ranges with formatted output
template <typename T>
void print_mismatch_ranges(
    const std::vector<T>& src_vec, const std::vector<T>& dst_vec, const std::string& context = "") {
    if (src_vec.size() != dst_vec.size()) {
        log_info(
            tt::LogTest,
            "{} Size mismatch: src_vec.size()={}, dst_vec.size()={}",
            context,
            src_vec.size(),
            dst_vec.size());
        return;
    }

    std::vector<std::pair<size_t, size_t>> mismatch_ranges;
    bool in_mismatch = false;
    size_t range_start = 0;

    // Find all mismatch ranges
    for (size_t i = 0; i < src_vec.size(); ++i) {
        if (src_vec[i] != dst_vec[i]) {
            if (!in_mismatch) {
                range_start = i;
                in_mismatch = true;
            }
        } else {
            if (in_mismatch) {
                mismatch_ranges.emplace_back(range_start, i - 1);
                in_mismatch = false;
            }
        }
    }
    // Handle case where mismatch extends to end
    if (in_mismatch) {
        mismatch_ranges.emplace_back(range_start, src_vec.size() - 1);
    }

    if (mismatch_ranges.empty()) {
        log_info(tt::LogTest, "{} Data matches perfectly!", context);
        return;
    }

    log_info(tt::LogTest, "{} Found {} mismatch range(s):", context, mismatch_ranges.size());

    for (const auto& [start, end] : mismatch_ranges) {
        log_info(tt::LogTest, "  Mismatch range: [{}, {}] (length: {})", start, end, end - start + 1);

        // Print first 128 mismatched values in this range
        size_t print_count = std::min(size_t(128), end - start + 1);
        log_info(tt::LogTest, "  First {} mismatched values:", print_count);

        std::stringstream ss;
        for (size_t i = 0; i < print_count; ++i) {
            if (i % 8 == 0) {
                if (i > 0) {
                    log_info(tt::LogTest, "{}", ss.str());
                    ss.str("");
                    ss.clear();
                }
                ss << "    ";
            }

            // Format as 8-character wide hex values, aligned columns
            ss << std::setw(8) << std::setfill('0') << std::hex << static_cast<uint64_t>(src_vec[start + i]) << ":"
               << std::setw(8) << std::setfill('0') << std::hex << static_cast<uint64_t>(dst_vec[start + i]);

            if ((i + 1) % 8 != 0 && i + 1 < print_count) {
                ss << "  ";
            }
        }

        if (!ss.str().empty()) {
            log_info(tt::LogTest, "{}", ss.str());
        }
    }
}

using MeshBufferTest2x4 = MeshDevice2x4Fixture;
using MeshBufferTestSuite = GenericMeshDeviceFixture;
using ElementType = uint64_t;
constexpr uint32_t ElementSize = sizeof(ElementType);

constexpr auto KB{1ul << 10};
constexpr auto MB{1ul << 20};
constexpr auto GB{1ul << 30};
class LargeMeshBufferTestSuiteBase : public MeshBufferTestSuite {
protected:
    static constexpr uint64_t max_num_elements_ = (12 * GB) / ElementSize;
    // A large datatype is used to reduce random shuffle time
    inline static std::vector<ElementType> src_vec_;

    LargeMeshBufferTestSuiteBase() {
        if (src_vec_.empty()) {
            log_info(tt::LogTest, "LargeMeshBufferTestSuiteBase ctor: initializing src_vec_");
            src_vec_.resize(max_num_elements_);
            std::iota(src_vec_.begin(), src_vec_.end(), 0);
            std::shuffle(src_vec_.begin(), src_vec_.end(), std::mt19937(std::random_device{}()));
        }
    }
};

bool validate_interleaved_test_inputs(size_t size, MeshDevice& mesh_device) {
    const size_t num_dram_channels = mesh_device.num_dram_channels();
    const size_t dram_size_per_channel = mesh_device.dram_size_per_channel();
    const DeviceAddr bank_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    bool result{true};
    auto allocation_size_per_bank = (size + num_dram_channels - 1) / num_dram_channels;
    result = result and allocation_size_per_bank <= dram_size_per_channel - bank_offset;
    if (not result) {
        log_info(
            tt::LogTest,
            "Skipping because requested size, size/bank ({:X}, {:X}) is greater than available "
            "size/bank, #banks ({:X}, {})",
            size,
            allocation_size_per_bank,
            dram_size_per_channel - bank_offset,
            num_dram_channels);
        return result;
    }

    return result;
}

class InterleavedMeshBufferTestSuite : public LargeMeshBufferTestSuiteBase,
                                       public testing::WithParamInterface<std::tuple<uint64_t, uint32_t>> {};

TEST_P(InterleavedMeshBufferTestSuite, NIGHTLY_DRAMReadback) {
    // - REPLICATED layout for writing, SHARDED with ROW_MAJOR for reading
    // - DRAM, bottom up allocation
    auto [tensor_size, page_size] = GetParam();
    if (!validate_interleaved_test_inputs(tensor_size, *mesh_device_)) {
        GTEST_SKIP();
    }

    const DeviceLocalBufferConfig device_local_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    log_info(tt::LogTest, "page_size: {}, tensor_size/device (MB): {}", page_size, tensor_size / (1 << 20));

    const ReplicatedBufferConfig buffer_config{.size = tensor_size};

    // Create replicated buffer for writing with specific address 32
    auto mesh_buffer = MeshBuffer::create(buffer_config, device_local_config, mesh_device_.get());
    // Verify buffer properties
    EXPECT_EQ(mesh_buffer->size(), tensor_size);
    EXPECT_EQ(mesh_buffer->global_layout(), MeshBufferLayout::REPLICATED);
    EXPECT_EQ(mesh_buffer->device_local_size(), tensor_size);
    EXPECT_TRUE(mesh_buffer->is_allocated());

    // Create test data - use uint16_t for easy verification
    auto num_elements = tensor_size / ElementSize;
    if (num_elements > max_num_elements_) {
        TT_THROW("Buffer size {} elements exceeds test vector {} elements", num_elements, max_num_elements_);
    }
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);

    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());
    auto mesh_size = coord_range.shape().mesh_size();
    std::vector<distributed::ShardDataTransfer> input_shards = {};
    input_shards.reserve(mesh_size);
    for (const auto& coord : coord_range) {
        input_shards.emplace_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<distributed::ShardDataTransfer> output_shards = {};
    output_shards.reserve(mesh_size);
    for (const auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(num_elements, 0);
        output_shards.emplace_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        std::string context =
            "Device coord (" + std::to_string(dst.first[0]) + "," + std::to_string(dst.first[1]) + ")";
        print_mismatch_ranges(src_vec, dst.second, context);
        EXPECT_EQ(dst.second, src_vec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeInterleavedReadback,
    InterleavedMeshBufferTestSuite,
    ::testing::Combine(
        ::testing::Values(2 * GB, 4 * GB, 8 * GB, 12 * GB),  // tensor sizes
        ::testing::Values(4 * KB, 16 * KB, 1 * MB)           // page sizes
        ));

bool validate_sharded_test_inputs(
    CoreCoord grid_size, Shape2D tensor_shape, MeshDevice& mesh_device, uint32_t element_sizeB) {
    CoreCoord available_grid_size = mesh_device.dram_grid_size();
    bool result{true};
    result = result and available_grid_size.x >= grid_size.x and available_grid_size.y >= grid_size.y;
    if (not result) {
        log_info(
            tt::LogTest,
            "Skipping test because required grid {} is incompatible with available grid {}",
            grid_size,
            available_grid_size);
    }

    // Determine if required allocation per bank can fit
    const size_t num_dram_channels = mesh_device.num_dram_channels();
    const size_t dram_size_per_channel = mesh_device.dram_size_per_channel();
    const DeviceAddr bank_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    uint32_t shard_height = tensor_shape.height() / grid_size.y;
    uint32_t shard_width = tensor_shape.width() / grid_size.x;
    size_t shard_sizeB = size_t(shard_height) * shard_width * element_sizeB;

    result = result and shard_sizeB <= dram_size_per_channel - bank_offset;
    if (not result) {
        log_info(
            tt::LogTest,
            "Skipping because required shard size/bank (tensor shape: {}, grid shape: {}, derived shard size: {:X}) is "
            "greater than available "
            "size/bank: {:X}, #banks: {})",
            tensor_shape,
            grid_size,
            shard_sizeB,
            dram_size_per_channel - bank_offset,
            num_dram_channels);
        return result;
    }

    return result;
}

class ShardedMeshBufferTestSuite
    : public LargeMeshBufferTestSuiteBase,
      public testing::WithParamInterface<std::tuple<std::pair<Shape2D, CoreCoord>, uint32_t>> {};

TEST_P(ShardedMeshBufferTestSuite, NIGHTLY_DRAMReadback) {
    // shard_shape: shape on device (elements)
    // page_size: (bytes)
    auto [tensor_and_grid, page_size] = GetParam();
    auto [device_tensor_shape, core_grid_size] = tensor_and_grid;
    uint64_t device_tensor_size = device_tensor_shape.height() * device_tensor_shape.width() * ElementSize;

    if (!validate_sharded_test_inputs(core_grid_size, device_tensor_shape, *mesh_device_, ElementSize)) {
        GTEST_SKIP();
    }

    uint32_t num_pages = static_cast<uint32_t>(device_tensor_size / page_size);
    if (uint64_t(num_pages) * page_size != device_tensor_size) {
        TT_THROW(
            "Check test parameters: shard shape:{}, page size:{} are incompatible (shard size:{}, #pages:{})",
            device_tensor_shape,
            page_size,
            device_tensor_size,
            num_pages);
    }

    CoreCoord start(0, 0);
    CoreCoord end(core_grid_size.x - 1, core_grid_size.y - 1);
    CoreRange cores(start, end);

    constexpr auto tile_height{constants::TILE_HEIGHT};
    constexpr auto tile_width{constants::TILE_WIDTH};
    constexpr auto tile_area{tile_height * tile_width};
    // Ensure buffer dimensions are divisible by tile dimensions
    ASSERT_TRUE(device_tensor_shape.height() % tile_height == 0);
    ASSERT_TRUE(device_tensor_shape.width() % tile_width == 0);

    auto test_helper_factor_area = [](uint32_t area, int32_t& height, int32_t& width) {
        // Find all factors of area
        std::vector<uint32_t> factors{1};
        auto temp = area;
        auto f = 2;
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
        height = 1, width = area;
        int32_t temp_height, temp_width;
        int32_t min_diff = std::abs(width - height);
        while (split_idx > 0) {
            temp_height = std::accumulate(factors.begin(), factors.begin() + split_idx, 1, std::multiplies<uint32_t>());
            temp_width = std::accumulate(factors.begin() + split_idx, factors.end(), 1, std::multiplies<uint32_t>());
            auto temp_diff = std::abs(height - width);
            if (temp_diff > min_diff) {
                break;
            }
            height = temp_height;
            width = temp_width;
            min_diff = temp_diff;
            split_idx--;
        }
    };

    // Determine page shape in elements
    uint32_t page_area = page_size / ElementSize;
    int32_t page_height;
    int32_t page_width;
    if (page_area >= tile_area) {
        // Calculate page dims in terms of number of tiles in a page
        if ((page_area / tile_area) * tile_area != page_area) {
            TT_THROW("page size elements {} is incompatible with tile size {}", page_area, tile_area);
        }
        test_helper_factor_area(page_area / tile_area, page_height, page_width);
        page_height *= tile_height;
        page_width *= tile_width;
    } else {
        // calculate page dims in terms of number of pages per tile
        if ((tile_area / page_area) * page_area != tile_area) {
            TT_THROW("page size elements {} is incompatible with tile size {}", page_area, tile_area);
        }
        int32_t tile_height_pages, tile_width_pages;
        test_helper_factor_area(tile_area / page_area, tile_height_pages, tile_width_pages);
        page_height = tile_height / tile_height_pages;
        page_width = tile_width / tile_width_pages;
    }
    if (page_height * page_width != page_area) {
        TT_THROW("Incorrectly derived page dims: {} {} {}", page_area, page_height, page_width);
    }
    Shape2D page_shape{page_height, page_width};
    // Device tensor shape in pages
    uint32_t device_tensor_height_pages = device_tensor_shape.height() / page_height;
    uint32_t device_tensor_width_pages = device_tensor_shape.width() / page_width;
    if (device_tensor_height_pages * page_height != device_tensor_shape.height()) {
        TT_THROW(
            "Shard height {} on device cannot fit integral # pages heightwise {}",
            device_tensor_shape.height(),
            page_height);
    }
    if (device_tensor_width_pages * page_width != device_tensor_shape.width()) {
        TT_THROW(
            "Shard width {} on device cannot fit integral # pages widthwise {}",
            device_tensor_shape.width(),
            page_width);
    }
    Shape2D device_tensor_shape_pages{device_tensor_height_pages, device_tensor_width_pages};

    // Derive shard dimensions (elements) for core grid. A shard is an on core memory block and must be in whole pages.
    uint32_t shard_height = device_tensor_shape.height() / core_grid_size.y;
    uint32_t shard_width = device_tensor_shape.width() / core_grid_size.x;
    Shape2D shard_shape{shard_height, shard_width};
    if (((shard_height / page_height) * page_height != shard_height) or
        ((shard_width / page_width) * page_width != shard_width)) {
        TT_THROW("Shard shape {} not an integral multiple of page shape {}", shard_shape, page_shape);
    }
    auto tensor_layout = TensorMemoryLayout::BLOCK_SHARDED;
    if (device_tensor_shape.height() == shard_shape.height() and device_tensor_shape.width() > shard_shape.width()) {
        tensor_layout = TensorMemoryLayout::WIDTH_SHARDED;
    } else if (
        device_tensor_shape.width() == shard_shape.width() and device_tensor_shape.height() > shard_shape.height()) {
        tensor_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    }
    auto shard_orientation = ShardOrientation::ROW_MAJOR;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer{CoreRangeSet(cores), shard_shape, shard_orientation, page_shape, device_tensor_shape_pages},
            tensor_layout),
        .bottom_up = true};

    auto num_elements = device_tensor_size / ElementSize;
    log_info(
        tt::LogTest,
        "Core grid size:{}, device tensor: size:{} MB, shape:{} pages, shape:{} elements, shard shape:{} elements, "
        "page shape:{} elements, page_size:{} KB, tensor layout:{}",
        core_grid_size,
        device_tensor_size >> 20,
        device_tensor_shape_pages,
        device_tensor_shape,
        shard_shape,
        page_shape,
        page_size >> 10,
        reflect::enum_name(tensor_layout));

    uint32_t device_rows = mesh_device_->num_rows();
    uint32_t device_cols = mesh_device_->num_cols();
    uint32_t num_devices = device_rows * device_cols;
    uint64_t tensor_size = num_devices * device_tensor_size;
    const ReplicatedBufferConfig buffer_config{.size = tensor_size};
    auto mesh_buffer = MeshBuffer::create(buffer_config, per_device_buffer_config, mesh_device_.get());
    distributed::MeshCoordinateRange coord_range(mesh_device_->shape());

    if (num_elements > max_num_elements_) {
        TT_THROW("Buffer size {} elements exceeds test vector {} elements", num_elements, max_num_elements_);
    }
    std::vector<ElementType> src_vec(src_vec_.begin(), src_vec_.begin() + num_elements);
    auto mesh_size = coord_range.shape().mesh_size();
    std::vector<distributed::ShardDataTransfer> input_shards = {};
    input_shards.reserve(mesh_size);
    for (const auto& coord : coord_range) {
        input_shards.emplace_back(distributed::ShardDataTransfer{coord}.host_data(src_vec.data()));
    }

    std::unordered_map<distributed::MeshCoordinate, std::vector<ElementType>> dst_vec = {};
    std::vector<distributed::ShardDataTransfer> output_shards = {};
    output_shards.reserve(mesh_size);
    for (const auto& coord : coord_range) {
        dst_vec[coord] = std::vector<ElementType>(device_tensor_size / ElementSize, 0);
        output_shards.emplace_back(distributed::ShardDataTransfer{coord}.host_data(dst_vec[coord].data()));
    }

    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, input_shards, false);
    mesh_device_->mesh_command_queue().enqueue_read_shards(output_shards, mesh_buffer, true);

    for (auto& dst : dst_vec) {
        std::string context =
            "Device coord (" + std::to_string(dst.first[0]) + "," + std::to_string(dst.first[1]) + ")";
        print_mismatch_ranges(src_vec, dst.second, context);
        EXPECT_EQ(dst.second, src_vec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LargeShardedReadback,
    ShardedMeshBufferTestSuite,
    ::testing::Combine(
        // shard_shape (elements), page_size (bytes)
        ::testing::Values(
            std::make_pair(Shape2D((1 << 14), (1 << 14)), CoreCoord(8, 1)),   // 2 GB with uint64_t
            std::make_pair(Shape2D((1 << 14), (1 << 15)), CoreCoord(8, 1)),   // 4 GB with uint64_t
            std::make_pair(Shape2D((1 << 14), (3 << 14)), CoreCoord(12, 1)),  // 6 GB with uint64_t
            std::make_pair(Shape2D((1 << 15), (1 << 15)), CoreCoord(8, 1))),  // 8 GB with uint64_t
        ::testing::Values(4096, 16 << 10, 1 << 20)                            // page size
        ));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
