// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <cstdlib>
#include <sys/types.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include "dispatch/system_memory_manager.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "impl/dispatch/command_queue_common.hpp"
#include "command_queue_fixture.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "dispatch_test_utils.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include "multi_command_queue_fixture.hpp"
#include <tt-metalium/shape2d.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>

#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {
class CommandQueue;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;

struct BufferStressTestConfig {
    // Used for normal write/read tests
    uint32_t seed;
    uint32_t num_pages_total;

    uint32_t page_size;
    uint32_t max_num_pages_per_buffer;

    // Used for wrap test
    uint32_t num_iterations;
    uint32_t num_unique_vectors;
};

class BufferStressTestConfigSharded {
public:
    uint32_t seed{};
    uint32_t num_iterations = 100;

    const std::array<uint32_t, 2> max_num_pages_per_core;
    const std::array<uint32_t, 2> max_num_cores;

    std::array<uint32_t, 2> num_pages_per_core{};
    std::array<uint32_t, 2> num_cores{};
    std::array<uint32_t, 2> page_shape = {32, 32};
    uint32_t element_size = 1;
    TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    BufferStressTestConfigSharded(std::array<uint32_t, 2> pages_per_core, std::array<uint32_t, 2> cores) :
        max_num_pages_per_core(pages_per_core),
        max_num_cores(cores),
        num_pages_per_core(pages_per_core),
        num_cores(cores) {}

    std::array<uint32_t, 2> tensor2d_shape_in_pages() {
        return {num_pages_per_core[0] * num_cores[0], num_pages_per_core[1] * num_cores[1]};
    }

    uint32_t num_pages() { return tensor2d_shape_in_pages()[0] * tensor2d_shape_in_pages()[1]; }

    std::array<uint32_t, 2> shard_shape() {
        return {num_pages_per_core[0] * page_shape[0], num_pages_per_core[1] * page_shape[1]};
    }

    CoreRangeSet shard_grid() {
        return CoreRangeSet(std::set<CoreRange>(
            {CoreRange(CoreCoord(0, 0), CoreCoord(this->num_cores[0] - 1, this->num_cores[1] - 1))}));
    }

    ShardSpecBuffer shard_parameters() {
        return ShardSpecBuffer(
            this->shard_grid(),
            this->shard_shape(),
            this->shard_orientation,
            this->page_shape,
            this->tensor2d_shape_in_pages());
    }

    uint32_t page_size() { return page_shape[0] * page_shape[1] * element_size; }
};

struct ShardedSubBufferStressTestConfig {
    uint32_t buffer_size = 0;
    uint32_t page_size = 0;
    uint32_t region_offset = 0;
    uint32_t region_size = 0;
    CoreRangeSet cores;
    Shape2D shard_shape;
    Shape2D page_shape;
    Shape2D tensor2d_shape_in_pages;
    TensorMemoryLayout layout;
    ShardOrientation orientation;
};

namespace local_test_functions {

vector<uint32_t> generate_arange_vector(uint32_t size_bytes) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "Error");
    vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    return src;
}

void clear_buffer(distributed::MeshCommandQueue& cq, const std::shared_ptr<distributed::MeshBuffer>& buffer) {
    TT_FATAL(buffer->size() % sizeof(uint32_t) == 0, "Error");
    vector<uint32_t> zeroes(buffer->size() / sizeof(uint32_t), 0);
    distributed::WriteShard(cq, buffer, zeroes, distributed::MeshCoordinate(0, 0));
}

vector<ShardedSubBufferStressTestConfig> generate_sharded_sub_buffer_test_configs(uint32_t max_buffer_size) {
    vector<ShardedSubBufferStressTestConfig> configs;

    uint32_t buffer_size = 0;
    while (buffer_size <= max_buffer_size) {
        uint32_t page_size = 4;
        while (page_size <= buffer_size) {
            uint32_t region_offset = 0;
            while (buffer_size % page_size == 0 && region_offset < buffer_size) {
                uint32_t region_size = page_size;
                while (region_offset + region_size <= buffer_size) {
                    CoreCoord start(0, 0);
                    for (uint32_t end_core_idx = 3; end_core_idx <= 4; end_core_idx++) {
                        CoreCoord end(end_core_idx, end_core_idx);
                        CoreRange cores(start, end);
                        const uint32_t num_pages = buffer_size / page_size;
                        const uint32_t num_shards = cores.size();
                        const uint32_t num_pages_per_shard = tt::div_up(num_pages, num_shards);
                        uint32_t page_shape_height_div_factor = 1;
                        while (page_shape_height_div_factor <= num_pages_per_shard) {
                            uint32_t page_shape_width_div_factor = 1;
                            while (page_shape_width_div_factor <= num_pages_per_shard) {
                                if (page_shape_width_div_factor * page_shape_height_div_factor == num_pages_per_shard) {
                                    uint32_t tensor2d_shape_in_pages_height = page_shape_height_div_factor;
                                    while (tensor2d_shape_in_pages_height <= num_pages) {
                                        uint32_t tensor2d_shape_in_pages_width = page_shape_width_div_factor;
                                        while (tensor2d_shape_in_pages_width <= num_pages) {
                                            if (tensor2d_shape_in_pages_height * tensor2d_shape_in_pages_width ==
                                                num_pages) {
                                                for (TensorMemoryLayout layout :
                                                     {TensorMemoryLayout::HEIGHT_SHARDED,
                                                      TensorMemoryLayout::BLOCK_SHARDED,
                                                      TensorMemoryLayout::WIDTH_SHARDED}) {
                                                    for (ShardOrientation orientation :
                                                         {ShardOrientation::COL_MAJOR, ShardOrientation::ROW_MAJOR}) {
                                                        ShardedSubBufferStressTestConfig config{
                                                            .buffer_size = buffer_size,
                                                            .page_size = page_size,
                                                            .region_offset = region_offset,
                                                            .region_size = region_size,
                                                            .cores = CoreRangeSet(cores),
                                                            .shard_shape =
                                                                {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                                                            .page_shape =
                                                                {tt::constants::TILE_HEIGHT /
                                                                     page_shape_height_div_factor,
                                                                 tt::constants::TILE_WIDTH /
                                                                     page_shape_width_div_factor},
                                                            .tensor2d_shape_in_pages =
                                                                {tensor2d_shape_in_pages_height,
                                                                 tensor2d_shape_in_pages_width},
                                                            .layout = layout,
                                                            .orientation = orientation};
                                                        configs.push_back(config);
                                                    }
                                                }
                                            }
                                            tensor2d_shape_in_pages_width += page_shape_width_div_factor;
                                        }
                                        tensor2d_shape_in_pages_height += page_shape_height_div_factor;
                                    }
                                }
                                page_shape_width_div_factor += 1;
                            }
                            page_shape_height_div_factor += 1;
                        }
                    }
                    region_size += 2 * page_size;
                }
                region_offset += page_size;
            }
            page_size += sizeof(uint32_t);
        }
        buffer_size += 2 * sizeof(uint32_t);
    }

    return configs;
}

// These are helper functions that are used for Slow Dispatch based IO. These are used in tests mixing Fast Dispatch IO
// with Slow Dispatch for validation.
void WriteToUnitMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const TestBufferConfig& config,
    const std::vector<uint32_t>& src,
    const std::shared_ptr<distributed::MeshBuffer>& buf,
    const std::optional<BufferShardingArgs>& sharding_args) {
    auto* device = mesh_device->get_devices()[0];
    std::shared_ptr<Buffer> slow_dispatch_buffer;
    if (sharding_args.has_value()) {
        slow_dispatch_buffer = Buffer::create(device, buf->address(), buf->size(), config.page_size, config.buftype, sharding_args.value());
    } else {
        slow_dispatch_buffer = Buffer::create(device, buf->address(), buf->size(), config.page_size, config.buftype);
    }
    detail::WriteToBuffer(*slow_dispatch_buffer, src);
}

void ReadFromUnitMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const TestBufferConfig& config,
    std::vector<uint32_t>& dst,
    const std::shared_ptr<distributed::MeshBuffer>& buf,
    const std::optional<BufferShardingArgs>& sharding_args) {
    auto* device = mesh_device->get_devices()[0];
    std::shared_ptr<Buffer> slow_dispatch_buffer;
    if (sharding_args.has_value()) {
        slow_dispatch_buffer = Buffer::create(device, buf->address(), buf->size(), config.page_size, config.buftype, sharding_args.value());
    } else {
        slow_dispatch_buffer = Buffer::create(device, buf->address(), buf->size(), config.page_size, config.buftype);
    }
    detail::ReadFromBuffer(*slow_dispatch_buffer, dst);
}

// These are helper functions used to write and read from a region of a MeshBuffer (sub-buffer)
void EnqueueWriteMeshSubBuffer(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const std::vector<uint32_t>& src,
    const BufferRegion& region,
    bool blocking) {
    auto shard_data_transfer = distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}
                                   .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                                   .region(region);

    cq.enqueue_write_shards(buffer, {shard_data_transfer}, blocking);
}

void EnqueueReadMeshSubBuffer(
    distributed::MeshCommandQueue& cq,
    std::vector<uint32_t>& dst,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const BufferRegion& region,
    bool blocking) {
    auto shard_data_transfer =
        distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}.host_data(dst.data()).region(region);

    cq.enqueue_read_shards({shard_data_transfer}, buffer, blocking);
}

template <bool cq_dispatch_only = false>
void test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const TestBufferConfig& config) {
    auto* device = mesh_device->get_devices()[0];
    // Clear out command queue
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    uint32_t cq_start =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    auto device_coord = distributed::MeshCoordinate(0, 0);
    std::vector<uint32_t> cq_zeros((cq_size - cq_start) / sizeof(uint32_t), 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
        cq_zeros.data(),
        (cq_size - cq_start),
        get_absolute_cq_offset(channel, 0, cq_size) + cq_start,
        mmio_device_id,
        channel);

    for (const bool cq_write : {true, false}) {
        for (const bool cq_read : {true, false}) {
            if constexpr (cq_dispatch_only) {
                if (not(cq_write and cq_read)) {
                    continue;
                }
            }
            if (not cq_write and not cq_read) {
                continue;
            }
            size_t buf_size = config.num_pages * config.page_size;

            std::shared_ptr<distributed::MeshBuffer> bufa;

            const distributed::ReplicatedBufferConfig buffer_config{.size = buf_size};

            if (config.sharding_args.has_value()) {
                distributed::DeviceLocalBufferConfig local_config{
                    .page_size = config.page_size,
                    .buffer_type = config.buftype,
                    .sharding_args = config.sharding_args.value(),
                    .bottom_up = false};
                bufa = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
            } else {
                distributed::DeviceLocalBufferConfig local_config{
                    .page_size = config.page_size,
                    .buffer_type = config.buftype,
                    .bottom_up = false,
                };
                bufa = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
            }

            vector<uint32_t> src = generate_arange_vector(bufa->size());

            if (cq_write) {
                distributed::WriteShard(cq, bufa, src, device_coord);
            } else {
                WriteToUnitMeshBuffer(mesh_device, config, src, bufa, config.sharding_args);
                if (config.buftype == BufferType::DRAM) {
                    tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device->id());
                } else {
                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
                }
            }

            vector<uint32_t> result;
            result.resize(buf_size / sizeof(uint32_t));

            if (cq_write and not cq_read) {
                distributed::Finish(cq);
            }

            if (cq_read) {
                distributed::ReadShard(cq, result, bufa, device_coord);
            } else {
                ReadFromUnitMeshBuffer(mesh_device, config, result, bufa, config.sharding_args);
            }

            EXPECT_EQ(src, result);
        }
    }
}

template <bool blocking>
bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const BufferStressTestConfig& config) {
    srand(config.seed);
    bool pass = true;
    uint32_t num_pages_left = config.num_pages_total;

    std::vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
    std::vector<std::vector<uint32_t>> srcs;
    std::vector<std::vector<uint32_t>> dsts;
    while (num_pages_left) {
        uint32_t num_pages = std::min((rand() % (config.max_num_pages_per_buffer)) + 1, num_pages_left);
        num_pages_left -= num_pages;

        uint32_t buf_size = num_pages * config.page_size;
        vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);

        for (unsigned int& val : src) {
            val = rand();
        }

        BufferType buftype = BufferType::DRAM;
        if ((rand() % 2) == 0) {
            buftype = BufferType::L1;
        }

        distributed::DeviceLocalBufferConfig local_config{
            .page_size = config.page_size, .buffer_type = buftype, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config {
            .size = buf_size
        };

        std::shared_ptr<distributed::MeshBuffer> buf;
        try {
            buf = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
        } catch (...) {
            Finish(cq);
            size_t i = 0;
            for (const auto& dst : dsts) {
                EXPECT_EQ(srcs[i++], dst);
            }
            srcs.clear();
            dsts.clear();
            buffers.clear();
            buf = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
        }
        auto coord = distributed::MeshCoordinate(0, 0);
        distributed::WriteShard(cq, buf, src, coord);
        vector<uint32_t> dst;
        if constexpr (blocking) {
            distributed::ReadShard(cq, dst, buf, coord);
            EXPECT_EQ(src, dst);
        } else {
            srcs.push_back(std::move(src));
            dsts.push_back(dst);
            buffers.push_back(std::move(buf));  // Ensures that buffer not destroyed when moved out of scope
            distributed::ReadShard(cq, dsts[dsts.size() - 1], buffers[buffers.size() - 1], coord);
        }
    }

    if constexpr (not blocking) {
        Finish(cq);
        size_t i = 0;
        for (const auto& dst : dsts) {
            EXPECT_EQ(srcs[i++], dst);
        }
    }
    return pass;
}

void stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    BufferStressTestConfigSharded& config,
    BufferType buftype,
    bool read_only) {
    srand(config.seed);

    auto device_coord = distributed::MeshCoordinate(0, 0);
    auto* device = mesh_device->get_devices()[0];

    for (const bool cq_write : {true, false}) {
        for (const bool cq_read : {true, false}) {
            // Temp until >64k writes enabled
            if (read_only and cq_write) {
                continue;
            }
            if (not cq_write and not cq_read) {
                continue;
            }
            // first keep num_pages_per_core consistent and increase num_cores
            for (uint32_t iteration_id = 0; iteration_id < config.num_iterations; iteration_id++) {
                auto shard_spec = config.shard_parameters();

                // explore a tensor_shape , keeping inner pages constant
                uint32_t num_pages = config.num_pages();

                uint32_t buf_size = num_pages * config.page_size();
                vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);

                for (uint32_t i = 0; i < src.size(); i++) {
                    src.at(i) = i;
                }

                distributed::DeviceLocalBufferConfig local_config{
                    .page_size = config.page_size(), .buffer_type = buftype, .bottom_up = false};
                const distributed::ReplicatedBufferConfig buffer_config{.size = buf_size};
                auto buf = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

                TestBufferConfig test_config = {
                    .num_pages = buf_size / config.page_size(), .page_size = config.page_size(), .buftype = buftype};

                vector<uint32_t> src2 = src;
                if (cq_write) {
                    distributed::WriteShard(cq, buf, src, device_coord, false);
                } else {
                    local_test_functions::WriteToUnitMeshBuffer(mesh_device, test_config, src, buf, std::nullopt);
                    if (buftype == BufferType::DRAM) {
                        tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device->id());
                    } else {
                        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
                    }
                }

                if (cq_write and not cq_read) {
                    Finish(cq);
                }

                vector<uint32_t> res;
                res.resize(buf_size / sizeof(uint32_t));
                if (cq_read) {
                    distributed::ReadShard(cq, res, buf, device_coord, true);
                } else {
                    local_test_functions::ReadFromUnitMeshBuffer(mesh_device, test_config, res, buf, std::nullopt);
                }
                EXPECT_EQ(src, res);
            }
        }
    }
}

std::pair<std::shared_ptr<distributed::MeshBuffer>, std::vector<uint32_t>> EnqueueWriteShard_prior_to_wrap(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const TestBufferConfig& config) {
    size_t buf_size = config.num_pages * config.page_size;

    distributed::DeviceLocalBufferConfig local_config{
        .page_size = config.page_size, .buffer_type = config.buftype, .bottom_up = false};

    const distributed::ReplicatedBufferConfig buffer_config{.size = buf_size};

    auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    std::vector<uint32_t> src =
        create_random_vector_of_bfloat16(buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    distributed::WriteShard(cq, buffer, src, distributed::MeshCoordinate(0, 0), false);
    return std::make_pair(std::move(buffer), src);
}

void test_EnqueueWrap_on_EnqueueReadBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const TestBufferConfig& config) {
    auto [buffer, src] = EnqueueWriteShard_prior_to_wrap(mesh_device, cq, config);
    vector<uint32_t> dst;
    distributed::ReadShard(cq, dst, buffer, distributed::MeshCoordinate(0, 0), true);

    EXPECT_EQ(src, dst);
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const BufferStressTestConfig& config) {
    srand(config.seed);

    vector<vector<uint32_t>> unique_vectors;
    for (uint32_t i = 0; i < config.num_unique_vectors; i++) {
        uint32_t num_pages = (rand() % (config.max_num_pages_per_buffer)) + 1;
        size_t buf_size = num_pages * config.page_size;
        unique_vectors.push_back(create_random_vector_of_bfloat16(
            buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));
    }

    vector<std::shared_ptr<distributed::MeshBuffer>> bufs;
    uint32_t start = 0;

    for (uint32_t i = 0; i < config.num_iterations; i++) {
        size_t buf_size = unique_vectors[i % unique_vectors.size()].size() * sizeof(uint32_t);
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = config.page_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buf_size};
        try {
            bufs.push_back(distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get()));
        } catch (const std::exception& e) {
            log_info(tt::LogTest, "Deallocating on iteration {}", i);
            bufs.clear();
            start = i;
            bufs = {distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get())};
        }
        distributed::WriteShard(
            cq,
            bufs[bufs.size() - 1],
            unique_vectors[i % unique_vectors.size()],
            distributed::MeshCoordinate(0, 0),
            false);
    }

    log_info(tt::LogTest, "Comparing {} buffers", bufs.size());
    bool pass = true;
    vector<uint32_t> dst;
    uint32_t idx = start;
    for (const auto& buffer : bufs) {
        distributed::ReadShard(cq, dst, buffer, distributed::MeshCoordinate(0, 0), true);
        pass &= dst == unique_vectors[idx % unique_vectors.size()];
        idx++;
    }

    return pass;
}

bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    vector<std::reference_wrapper<distributed::MeshCommandQueue>>& cqs,
    const TestBufferConfig& config) {
    bool pass = true;
    for (const bool use_void_star_api : {true, false}) {
        size_t buf_size = config.num_pages * config.page_size;
        std::vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
        std::vector<std::vector<uint32_t>> srcs;
        for (uint i = 0; i < cqs.size(); i++) {
            distributed::DeviceLocalBufferConfig dram_config{
                .page_size = config.page_size,
                .buffer_type = config.buftype,
                .bottom_up = false};
            const distributed::ReplicatedBufferConfig buffer_config{
                .size = buf_size};

            buffers.push_back(distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get()));
            srcs.push_back(generate_arange_vector(buffers[i]->size()));
            distributed::WriteShard(cqs[i], buffers[i], srcs[i], distributed::MeshCoordinate(0, 0), false);
        }

        for (uint i = 0; i < cqs.size(); i++) {
            std::vector<uint32_t> result;
            if (use_void_star_api) {
                result.resize(buf_size / sizeof(uint32_t));
                distributed::ReadShard(cqs[i], result, buffers[i], distributed::MeshCoordinate(0, 0), true);
            } else {
                distributed::ReadShard(cqs[i], result, buffers[i], distributed::MeshCoordinate(0, 0), true);
            }
            bool local_pass = (srcs[i] == result);
            pass &= local_pass;
        }
    }

    return pass;
}

}  // end namespace local_test_functions

namespace basic_tests {
namespace dram_tests {

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileToDramBank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileToAllDramBanks) {
    for (const auto& mesh_device : devices_) {
        TestBufferConfig config = {
            .num_pages = uint32_t(mesh_device->allocator()->get_num_banks(BufferType::DRAM)),
            .page_size = 2048,
            .buftype = BufferType::DRAM};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr uint32_t num_round_robins = 2;
    for (const auto& mesh_device : devices_) {
        TestBufferConfig config = {
            .num_pages = num_round_robins * (mesh_device->allocator()->get_num_banks(BufferType::DRAM)),
            .page_size = 2048,
            .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, Sending131072Pages) {
    for (const auto& mesh_device : devices_) {
        TestBufferConfig config = {.num_pages = 131072, .page_size = 128, .buftype = BufferType::DRAM};
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestPageLargerThanAndUnalignedToTransferPage) {
    constexpr uint32_t num_round_robins = 2;
    for (const auto& mesh_device : devices_) {
        TestBufferConfig config = {
            .num_pages = num_round_robins * (mesh_device->allocator()->get_num_banks(BufferType::DRAM)),
            .page_size = DispatchSettings::TRANSFER_PAGE_SIZE + 32,
            .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestSinglePageLargerThanMaxPrefetchCommandSize) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 1, .page_size = max_prefetch_command_size + 2048, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSize) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 1024, .page_size = max_prefetch_command_size + 2048, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestSinglePageLargerThanMaxPrefetchCommandSizeShardedBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 2048;
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        CoreCoord start(0, 0);
        CoreCoord end(0, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {1, 1});
        TestBufferConfig config = {
            .num_pages = 1,
            .page_size = page_size,
            .buftype = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED)};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSizeShardedBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 2048;
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->get_devices()[0]->id());
        CoreCoord start(0, 0);
        CoreCoord end(3, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT / 4, tt::constants::TILE_WIDTH / 2},
            {4, 8});
        TestBufferConfig config = {
            .num_pages = 32,
            .page_size = page_size,
            .buftype = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED)};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeShardedBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 4;
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        CoreCoord start(0, 0);
        CoreCoord end(3, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::COL_MAJOR,
            {tt::constants::TILE_HEIGHT / 4, tt::constants::TILE_WIDTH / 2},
            {4, 8});
        TestBufferConfig config = {
            .num_pages = 32,
            .page_size = page_size,
            .buftype = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED)};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSizeSubBuffer) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());

        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        const uint32_t page_size = max_prefetch_command_size + 2048;
        const uint32_t buffer_size = 40 * page_size;
        const uint32_t region_size = 5 * page_size;
        const uint32_t region_offset = 30 * page_size;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        const BufferRegion region(region_offset, region_size);
        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        std::vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestSingleUnalignedPageLargerThanMaxPrefetchCommandSize) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        uint32_t unaligned_page_size = max_prefetch_command_size + 4;
        TestBufferConfig config = {.num_pages = 1, .page_size = unaligned_page_size, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSize) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 1024, .page_size = max_prefetch_command_size + 4, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeSubBuffer) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());

        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        const uint32_t page_size = max_prefetch_command_size + 4;
        const uint32_t buffer_size = 40 * page_size;
        const uint32_t region_size = 5 * page_size;
        const uint32_t region_offset = 30 * page_size;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        const BufferRegion region(region_offset, region_size);
        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSizeShardedSubBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 2048;
    const uint32_t buffer_size = 20 * page_size;
    const uint32_t region_size = 5 * page_size;
    const uint32_t region_offset = 9 * page_size;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        CoreCoord start(0, 0);
        CoreCoord end(4, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT / 2, tt::constants::TILE_WIDTH / 2},
            {5, 4});

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED),
            .bottom_up = false
            };

        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        const BufferRegion region(region_offset, region_size);
        auto src = local_test_functions::generate_arange_vector(region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);

        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeShardedSubBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 4;
    const uint32_t buffer_size = 20 * page_size;
    const uint32_t region_size = 5 * page_size;
    const uint32_t region_offset = 9 * page_size;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        CoreCoord start(0, 0);
        CoreCoord end(4, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::COL_MAJOR,
            {tt::constants::TILE_HEIGHT / 4, tt::constants::TILE_WIDTH},
            {4, 5});

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED),
            .bottom_up = false};

        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        const BufferRegion region(region_offset, region_size);
        auto src = local_test_functions::generate_arange_vector(region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);

        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestNon32BAlignedPageSizeForDram) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    for (const auto& mesh_device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestNon32BAlignedPageSizeForDram2) {
    // From stable diffusion read buffer
    TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    for (const auto& mesh_device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

// Requires enqueue write buffer
TEST_F(UnitMeshCQSingleCardBufferFixture, TestWrapHostHugepageOnEnqueueReadBuffer) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        uint32_t page_size = 2048;
        uint32_t command_issue_region_size = device->sysmem_manager().get_issue_queue_size(0);
        uint32_t cq_start = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
            CommandQueueHostAddrType::UNRESERVED);

        uint32_t max_command_size = command_issue_region_size - cq_start;
        uint32_t buffer = 14240;
        uint32_t buffer_size = max_command_size - buffer;
        uint32_t num_pages = buffer_size / page_size;

        TestBufferConfig buf_config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWrap_on_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), buf_config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        uint32_t page_size = 2048;
        uint32_t command_queue_size = device->sysmem_manager().get_cq_size();
        uint32_t num_pages = command_queue_size / page_size;

        TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

// Test that command queue wraps when buffer available space in completion region is less than a page
TEST_F(UnitMeshCQSingleCardBufferFixture, TestWrapCompletionQOnInsufficientSpace) {
    uint32_t large_page_size = 8192;  // page size for first and third read
    uint32_t small_page_size = 2048;  // page size for second read

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        uint32_t command_completion_region_size = device->sysmem_manager().get_completion_queue_size(0);

        uint32_t first_buffer_size =
            tt::round_up(static_cast<uint32_t>(command_completion_region_size * 0.95), large_page_size);

        uint32_t space_after_first_buffer = command_completion_region_size - first_buffer_size;
        // leave only small_page_size * 2 B of space in the completion queue
        uint32_t num_pages_second_buffer = (space_after_first_buffer / small_page_size) - 2;

        distributed::DeviceLocalBufferConfig dram_config1{
            .page_size = large_page_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config1{.size = first_buffer_size};

        auto buff_1 = distributed::MeshBuffer::create(buffer_config1, dram_config1, mesh_device.get());

        auto src_1 = local_test_functions::generate_arange_vector(buff_1->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buff_1, src_1, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result_1(first_buffer_size / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_1, buff_1, distributed::MeshCoordinate(0, 0), true);
        EXPECT_EQ(src_1, result_1);

        distributed::DeviceLocalBufferConfig dram_config2{
            .page_size = small_page_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config2{.size = num_pages_second_buffer * small_page_size};

        auto buff_2 = distributed::MeshBuffer::create(buffer_config2, dram_config2, mesh_device.get());
        auto src_2 = local_test_functions::generate_arange_vector(buff_2->size());

        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buff_2, src_2, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result_2(num_pages_second_buffer * small_page_size / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_2, buff_2, distributed::MeshCoordinate(0, 0), true);
        EXPECT_EQ(src_2, result_2);

        distributed::DeviceLocalBufferConfig dram_config3{
            .page_size = large_page_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config3{.size = 32 * large_page_size};

        auto buff_3 = distributed::MeshBuffer::create(buffer_config3, dram_config3, mesh_device.get());
        auto src_3 = local_test_functions::generate_arange_vector(buff_3->size());

        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buff_3, src_3, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result_3(32 * large_page_size / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_3, buff_3, distributed::MeshCoordinate(0, 0), true);
        EXPECT_EQ(src_3, result_3);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteShardedSubBuffer) {
    const uint32_t page_size = 256;
    const uint32_t buffer_size = 64 * page_size;
    const BufferRegion region(256, 512);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        CoreCoord start(0, 0);
        CoreCoord end(5, 0);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {8, 8});

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED),
            .bottom_up = false};

        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        vector<uint32_t> src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);

        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteSubBuffer) {
    const uint32_t page_size = 256;
    const uint32_t buffer_size = 64 * page_size;
    const BufferRegion region(256, 512);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        vector<uint32_t> src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);

        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteSubBufferLargeOffset) {
    const uint32_t page_size = 4;
    const uint32_t buffer_size = (0xFFFF + 50000) * 2 * page_size;
    const BufferRegion region(((2 * 0xFFFF) + 25000) * page_size, 32);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        vector<uint32_t> src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadBufferWriteSubBuffer) {
    const uint32_t page_size = 128;
    const uint32_t buffer_size = 100 * page_size;
    const uint32_t buffer_region_offset = 50 * page_size;
    const uint32_t buffer_region_size = 128;
    const BufferRegion region(buffer_region_offset, buffer_region_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        auto src = local_test_functions::generate_arange_vector(buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> read_buf_result(buffer_region_size / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), read_buf_result, buffer, distributed::MeshCoordinate(0, 0), true);
        vector<uint32_t> result;
        for (uint32_t i = buffer_region_offset / sizeof(uint32_t);
             i < (buffer_region_offset + buffer_region_size) / sizeof(uint32_t);
             i++) {
            result.push_back(read_buf_result[i]);
        }
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadSubBufferWriteBuffer) {
    const uint32_t page_size = 128;
    const uint32_t buffer_size = 100 * page_size;
    const uint32_t buffer_region_offset = 50 * page_size;
    const uint32_t buffer_region_size = 128;
    const BufferRegion region(buffer_region_offset, buffer_region_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(buffer_size);
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buffer, src, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result(buffer_region_size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        vector<uint32_t> expected_result;
        for (uint32_t i = buffer_region_offset / sizeof(uint32_t);
             i < (buffer_region_offset + buffer_region_size) / sizeof(uint32_t);
             i++) {
            expected_result.push_back(src[i]);
        }
        EXPECT_EQ(expected_result, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadSubBufferInvalidRegion) {
    const uint32_t page_size = 4;
    const uint32_t buffer_size = 100 * page_size;
    const uint32_t buffer_region_offset = 25 * page_size;
    const uint32_t buffer_region_size = buffer_size;
    const BufferRegion region(buffer_region_offset, buffer_region_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        vector<uint32_t> result(buffer_region_size / sizeof(uint32_t));
        EXPECT_ANY_THROW(local_test_functions::EnqueueReadMeshSubBuffer(
            mesh_device->mesh_command_queue(), result, buffer, region, true));
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestWriteSubBufferInvalidRegion) {
    const uint32_t page_size = 4;
    const uint32_t buffer_size = 100 * page_size;
    const uint32_t buffer_region_offset = 25 * page_size;
    const uint32_t buffer_region_size = buffer_size;
    const BufferRegion region(buffer_region_offset, buffer_region_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(buffer_region_size);
        EXPECT_ANY_THROW(local_test_functions::EnqueueWriteMeshSubBuffer(
            mesh_device->mesh_command_queue(), buffer, src, region, true));
    }
}

// Test that command queue wraps when buffer read needs to be split into multiple enqueue_read_buffer commands and
// available space in completion region is less than a page
TEST_F(UnitMeshCQSingleCardBufferFixture, TestWrapCompletionQOnInsufficientSpace2) {
    // Using default 75-25 issue and completion queue split
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        uint32_t command_completion_region_size = device->sysmem_manager().get_completion_queue_size(0);

        uint32_t num_pages_buff_1 = 9;
        uint32_t page_size_buff_1 = 2048;
        distributed::DeviceLocalBufferConfig dram_config1{
            .page_size = page_size_buff_1, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config1{.size = num_pages_buff_1 * page_size_buff_1};
        auto buff_1 = distributed::MeshBuffer::create(buffer_config1, dram_config1, mesh_device.get());
        uint32_t space_after_buff_1 = command_completion_region_size - buff_1->size();

        uint32_t page_size = 8192;
        uint32_t desired_remaining_space_before_wrap = 6144;
        uint32_t avail_space_for_wrapping_buffer = space_after_buff_1 - desired_remaining_space_before_wrap;
        uint32_t num_pages_for_wrapping_buffer = (avail_space_for_wrapping_buffer / page_size) + 4;

        auto src_1 = local_test_functions::generate_arange_vector(buff_1->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buff_1, src_1, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result_1(buff_1->size() / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_1, buff_1, distributed::MeshCoordinate(0, 0), true);
        EXPECT_EQ(src_1, result_1);

        distributed::DeviceLocalBufferConfig dram_config2{
            .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config2{.size = num_pages_for_wrapping_buffer * page_size};

        auto wrap_buff = distributed::MeshBuffer::create(buffer_config2, dram_config2, mesh_device.get());
        auto src_2 = local_test_functions::generate_arange_vector(wrap_buff->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), wrap_buff, src_2, distributed::MeshCoordinate(0, 0), false);
        vector<uint32_t> result_2(wrap_buff->size() / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_2, wrap_buff, distributed::MeshCoordinate(0, 0), true);
        EXPECT_EQ(src_2, result_2);
    }
}

// TODO: add test for wrapping with non aligned page sizes

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileToDramBank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileToAllDramBanks) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        TestBufferConfig config = {
            .num_pages = uint32_t(mesh_device->allocator()->get_num_banks(BufferType::DRAM)),
            .page_size = 2048,
            .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr uint32_t num_round_robins = 2;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        TestBufferConfig config = {
            .num_pages = num_round_robins * (mesh_device->allocator()->get_num_banks(BufferType::DRAM)),
            .page_size = 2048,
            .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, Sending131072Pages) {
    // Was a failing case where we used to accidentally program cb num pages to be total
    // pages instead of cb num pages.
    TestBufferConfig config = {.num_pages = 131072, .page_size = 128, .buftype = BufferType::DRAM};
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, TestNon32BAlignedPageSizeForDram) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, TestNon32BAlignedPageSizeForDram2) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        // From stable diffusion read buffer
        TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        uint32_t page_size = 2048;
        uint32_t command_queue_size = device->sysmem_manager().get_cq_size();
        uint32_t num_pages = command_queue_size / page_size;

        TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileToDramBank0) {
    auto mesh_device = this->device_;
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileToAllDramBanks) {
    auto mesh_device = this->device_;
    auto* device = mesh_device->get_devices()[0];
    TestBufferConfig config = {
        .num_pages = uint32_t(device->allocator()->get_num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    auto mesh_device = this->device_;
    auto* device = mesh_device->get_devices()[0];
    constexpr uint32_t num_round_robins = 2;
    TestBufferConfig config = {
        .num_pages = num_round_robins * (device->allocator()->get_num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, Sending131072Pages) {
    auto mesh_device = this->device_;
    // Was a failing case where we used to accidentally program cb num pages to be total
    // pages instead of cb num pages.
    TestBufferConfig config = {.num_pages = 131072, .page_size = 128, .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, TestNon32BAlignedPageSizeForDram) {
    auto mesh_device = this->device_;
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, TestNon32BAlignedPageSizeForDram2) {
    auto mesh_device = this->device_;
    // From stable diffusion read buffer
    TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    auto mesh_device = this->device_;
    auto* device = mesh_device->get_devices()[0];
    uint32_t page_size = 2048;
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    uint32_t command_queue_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_size(device->id(), channel);
    uint32_t num_pages = command_queue_size / page_size;

    TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshCQMultiDeviceBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSize) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 50, .page_size = max_prefetch_command_size + 4, .buftype = BufferType::DRAM};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

}  // end namespace dram_tests

namespace l1_tests {

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteShardedSubBufferForL1) {
    const uint32_t max_buffer_size = 152;
    const std::vector<ShardedSubBufferStressTestConfig>& configs =
        local_test_functions::generate_sharded_sub_buffer_test_configs(max_buffer_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running on Device {}", device->id());
        for (const ShardedSubBufferStressTestConfig& config : configs) {
            log_debug(
                tt::LogTest,
                "Device: {} buffer_size: {} page_size: {} region_offset: {} region_size: {} shard_shape: [{}, {}] "
                "page_shape: [{}, {}] tensor2d_shape_in_pages: [{}, {}] layout: {} orientation: {} cores: {}",
                device->id(),
                config.buffer_size,
                config.page_size,
                config.region_offset,
                config.region_size,
                config.shard_shape.height(),
                config.shard_shape.width(),
                config.page_shape.height(),
                config.page_shape.width(),
                config.tensor2d_shape_in_pages.height(),
                config.tensor2d_shape_in_pages.width(),
                enchantum::to_string(config.layout).data(),
                enchantum::to_string(config.orientation).data(),
                config.cores.str());

            ShardSpecBuffer shard_spec = ShardSpecBuffer(
                config.cores,
                {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                config.orientation,
                config.page_shape,
                config.tensor2d_shape_in_pages);

            distributed::DeviceLocalBufferConfig l1_config{
                .page_size = config.page_size, .buffer_type = BufferType::L1, .bottom_up = false};
            const distributed::ReplicatedBufferConfig buffer_config{.size = config.buffer_size};
            auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

            local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

            const BufferRegion region(config.region_offset, config.region_size);
            vector<uint32_t> src = local_test_functions::generate_arange_vector(config.region_size);
            local_test_functions::EnqueueWriteMeshSubBuffer(
                mesh_device->mesh_command_queue(), buffer, src, region, false);
            vector<uint32_t> result(region.size / sizeof(uint32_t));
            local_test_functions::EnqueueReadMeshSubBuffer(
                mesh_device->mesh_command_queue(), result, buffer, region, true);
            EXPECT_EQ(src, result);
        }
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleNonOverlappingWritesShardedSubBufferForL1) {
    const uint32_t page_size = 64;
    const uint32_t buffer_size = 16 * page_size;
    const uint32_t buffer_region_size = 4 * page_size;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running on Device {}", device->id());
        CoreCoord start_coord = {0, 0};
        CoreCoord end_coord = {5, 5};
        CoreRange cores(start_coord, end_coord);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {16, 1});

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED),
            .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        vector<uint32_t> ones(buffer_region_size / sizeof(uint32_t), 1);
        BufferRegion region(0, buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, ones, region, false);

        vector<uint32_t> twos(buffer_region_size / sizeof(uint32_t), 2);
        region = BufferRegion(buffer_region_size, buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, twos, region, false);

        vector<uint32_t> threes(buffer_region_size / sizeof(uint32_t), 3);
        region = BufferRegion(buffer_region_size * 2, buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(
            mesh_device->mesh_command_queue(), buffer, threes, region, false);

        vector<uint32_t> fours(buffer_region_size / sizeof(uint32_t), 4);
        region = BufferRegion(buffer_region_size * 3, buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(
            mesh_device->mesh_command_queue(), buffer, fours, region, false);

        vector<uint32_t> expected;
        expected.reserve(buffer_size / sizeof(uint32_t));

        expected.insert(expected.end(), ones.begin(), ones.end());
        expected.insert(expected.end(), twos.begin(), twos.end());
        expected.insert(expected.end(), threes.begin(), threes.end());
        expected.insert(expected.end(), fours.begin(), fours.end());

        vector<uint32_t> result(buffer_size / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result, buffer, distributed::MeshCoordinate(0, 0), true);

        EXPECT_EQ(expected, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSizeForL1) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 30, .page_size = max_prefetch_command_size + 2048, .buftype = BufferType::L1};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultiplePagesLargerThanMaxPrefetchCommandSizeForL1ShardedBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 2048;
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        CoreCoord start(0, 0);
        CoreCoord end(4, 4);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::COL_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {5, 5});
        TestBufferConfig config = {
            .num_pages = 25,
            .page_size = page_size,
            .buftype = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED)};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestSingleUnalignedPageLargerThanMaxPrefetchCommandSizeForL1) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 1, .page_size = max_prefetch_command_size + 4, .buftype = BufferType::L1};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeForL1) {
    for (const auto& mesh_device : devices_) {
        const uint32_t max_prefetch_command_size =
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 30, .page_size = max_prefetch_command_size + 4, .buftype = BufferType::L1};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(
    UnitMeshCQSingleCardBufferFixture, TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeForL1ShardedBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 4;
    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->id());
        CoreCoord start(0, 0);
        CoreCoord end(3, 3);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH / 2},
            {8, 4});
        TestBufferConfig config = {
            .num_pages = 32,
            .page_size = page_size,
            .buftype = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED)};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(
    UnitMeshCQSingleCardBufferFixture,
    TestMultipleUnalignedPagesLargerThanMaxPrefetchCommandSizeForL1ShardedSubBuffer) {
    const uint32_t page_size = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() + 4;
    const uint32_t buffer_size = 32 * page_size;
    const uint32_t region_offset = 16 * page_size;
    const uint32_t region_size = 16 * page_size;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        CoreCoord start(0, 0);
        CoreCoord end(3, 3);
        CoreRange cores(start, end);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT / 2, tt::constants::TILE_WIDTH},
            {16, 2});

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);

        const BufferRegion region(region_offset, region_size);
        auto src = local_test_functions::generate_arange_vector(region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);

        vector<uint32_t> result(region_size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);

        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestMultipleNonOverlappingReadsShardedSubBufferForL1) {
    const uint32_t page_size = 64;
    const uint32_t buffer_size = 16 * page_size;
    const uint32_t buffer_region_size = buffer_size / 4;
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running on Device {}", device->id());
        CoreCoord start_coord = {0, 0};
        CoreCoord end_coord = {5, 5};
        CoreRange cores(start_coord, end_coord);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {8, 2});

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED),
            .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

        vector<uint32_t> expected = local_test_functions::generate_arange_vector(buffer_size);
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), buffer, expected, distributed::MeshCoordinate(0, 0), true);

        vector<uint32_t> first_read(buffer_region_size / sizeof(uint32_t));
        BufferRegion region(0, buffer_region_size);
        local_test_functions::EnqueueReadMeshSubBuffer(
            mesh_device->mesh_command_queue(), first_read, buffer, region, false);

        vector<uint32_t> second_read(buffer_region_size / sizeof(uint32_t));
        region = BufferRegion(buffer_region_size, buffer_region_size);
        local_test_functions::EnqueueReadMeshSubBuffer(
            mesh_device->mesh_command_queue(), second_read, buffer, region, false);

        vector<uint32_t> third_read(buffer_region_size / sizeof(uint32_t));
        region = BufferRegion(buffer_region_size * 2, buffer_region_size);
        local_test_functions::EnqueueReadMeshSubBuffer(
            mesh_device->mesh_command_queue(), third_read, buffer, region, false);

        vector<uint32_t> fourth_read(buffer_region_size / sizeof(uint32_t));
        region = BufferRegion(buffer_region_size * 3, buffer_region_size);
        local_test_functions::EnqueueReadMeshSubBuffer(
            mesh_device->mesh_command_queue(), fourth_read, buffer, region, false);

        mesh_device->mesh_command_queue().finish();

        vector<uint32_t> result;
        result.reserve(buffer_size / sizeof(uint32_t));

        result.insert(result.end(), first_read.begin(), first_read.end());
        result.insert(result.end(), second_read.begin(), second_read.end());
        result.insert(result.end(), third_read.begin(), third_read.end());
        result.insert(result.end(), fourth_read.begin(), fourth_read.end());

        EXPECT_EQ(expected, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteShardedSubBufferMultiplePagesPerShardForL1) {
    const uint32_t page_size = 64;
    const uint32_t buffer_size = page_size * 16;
    const uint32_t buffer_region_offset = page_size * 3;
    const uint32_t buffer_region_size = page_size * 7;
    vector<uint32_t> src = local_test_functions::generate_arange_vector(buffer_region_size);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running on Device {}", device->id());
        CoreCoord start_coord = {0, 0};
        CoreCoord end_coord = {3, 3};
        CoreRange cores(start_coord, end_coord);
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            CoreRangeSet(cores),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::COL_MAJOR,
            {tt::constants::TILE_HEIGHT / 4, tt::constants::TILE_WIDTH / 2},
            {8, 2});

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::BLOCK_SHARDED),
            .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
        local_test_functions::clear_buffer(mesh_device->mesh_command_queue(), buffer);
        const BufferRegion region(buffer_region_offset, buffer_region_size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> result(buffer_region_size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteSubBufferForL1) {
    const uint32_t page_size = 256;
    const uint32_t buffer_size = 128 * page_size;
    const BufferRegion region(2 * page_size, 2048);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestReadWriteSubBufferLargeOffsetForL1) {
    const uint32_t page_size = 256;
    const uint32_t buffer_size = 512 * page_size;
    const BufferRegion region(400 * page_size, 2048);
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());

        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
        auto src = local_test_functions::generate_arange_vector(region.size);
        local_test_functions::EnqueueWriteMeshSubBuffer(mesh_device->mesh_command_queue(), buffer, src, region, false);
        vector<uint32_t> result(region.size / sizeof(uint32_t));
        local_test_functions::EnqueueReadMeshSubBuffer(mesh_device->mesh_command_queue(), result, buffer, region, true);
        EXPECT_EQ(src, result);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileToL1Bank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
    for (const auto& mesh_device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileToAllL1Banks) {
    for (const auto& mesh_device : devices_) {
        auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();
        TestBufferConfig config = {
            .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    for (const auto& mesh_device : devices_) {
        auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();
        TestBufferConfig config = {
            .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestNon32BAlignedPageSizeForL1) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        if (device->is_mmio_capable()) {
            continue;
        }
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, TestBackToBackNon32BAlignedPageSize) {
    constexpr BufferType buff_type = BufferType::L1;

    for (const auto& mesh_device : devices_) {
        distributed::DeviceLocalBufferConfig l1_config1{.page_size = 100, .buffer_type = buff_type, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config1{.size = 125000};
        auto bufa = distributed::MeshBuffer::create(buffer_config1, l1_config1, mesh_device.get());
        auto src_a = local_test_functions::generate_arange_vector(bufa->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), bufa, src_a, distributed::MeshCoordinate(0, 0), false);

        distributed::DeviceLocalBufferConfig l1_config2{.page_size = 152, .buffer_type = buff_type, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config2{.size = 152000};
        auto bufb = distributed::MeshBuffer::create(buffer_config2, l1_config2, mesh_device.get());
        auto src_b = local_test_functions::generate_arange_vector(bufb->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), bufb, src_b, distributed::MeshCoordinate(0, 0), false);

        vector<uint32_t> result_a(bufa->size() / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_a, bufa, distributed::MeshCoordinate(0, 0), true);

        vector<uint32_t> result_b(bufb->size() / sizeof(uint32_t));
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), result_b, bufb, distributed::MeshCoordinate(0, 0), true);

        EXPECT_EQ(src_a, result_a);
        EXPECT_EQ(src_b, result_b);
    }
}

// This case was failing for FD v1.3 design
TEST_F(UnitMeshCQSingleCardBufferFixture, TestLargeBuffer4096BPageSize) {
    for (const auto& mesh_device : devices_) {
        TestBufferConfig config = {.num_pages = 512, .page_size = 4096, .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
            mesh_device, mesh_device->mesh_command_queue(), config);
    }
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileToL1Bank0) {
    auto mesh_device = this->device_;
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileToAllL1Banks) {
    auto mesh_device = this->device_;
    auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    auto mesh_device = this->device_;
    auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQSingleDeviceBufferFixture, TestNon32BAlignedPageSizeForL1) {
    auto mesh_device = this->device_;
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
    EXPECT_TRUE(
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileToL1Bank0) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileToAllL1Banks) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();
        TestBufferConfig config = {
            .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        auto compute_with_storage_grid = mesh_device->compute_with_storage_grid_size();

        TestBufferConfig config = {
            .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

TEST_F(UnitMeshMultiCQMultiDeviceBufferFixture, TestNon32BAlignedPageSizeForL1) {
    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running On Device {}", device->id());
        TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

        distributed::MeshCommandQueue& a = mesh_device->mesh_command_queue(0);
        distributed::MeshCommandQueue& b = mesh_device->mesh_command_queue(1);
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {a, b};
        EXPECT_TRUE(
            local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(mesh_device, cqs, config));
    }
}

}  // end namespace l1_tests

TEST_F(UnitMeshCQSingleCardBufferFixture, TestNonblockingReads) {
    constexpr BufferType buff_type = BufferType::L1;

    for (const auto& mesh_device : devices_) {
        distributed::DeviceLocalBufferConfig l1_config1{
            .page_size = 2048, .buffer_type = buff_type, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config1{.size = 2048};
        auto bufa = distributed::MeshBuffer::create(buffer_config1, l1_config1, mesh_device.get());
        auto src_a = local_test_functions::generate_arange_vector(bufa->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), bufa, src_a, distributed::MeshCoordinate(0, 0), false);

        distributed::DeviceLocalBufferConfig l1_config2{
            .page_size = 2048, .buffer_type = buff_type, .bottom_up = false};
        const distributed::ReplicatedBufferConfig buffer_config2{.size = 2048};
        auto bufb = distributed::MeshBuffer::create(buffer_config2, l1_config2, mesh_device.get());

        auto src_b = local_test_functions::generate_arange_vector(bufb->size());
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), bufb, src_b, distributed::MeshCoordinate(0, 0), false);

        vector<uint32_t> result_a(bufa->size() / sizeof(uint32_t));
        distributed::ReadShard(mesh_device->mesh_command_queue(), result_a, bufa, distributed::MeshCoordinate(0, 0));

        vector<uint32_t> result_b(bufb->size() / sizeof(uint32_t));
        distributed::ReadShard(mesh_device->mesh_command_queue(), result_b, bufb, distributed::MeshCoordinate(0, 0));
        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_EQ(src_a, result_a);
        EXPECT_EQ(src_b, result_b);
    }
}

}  // end namespace basic_tests

namespace stress_tests {

// TODO: Add stress test that vary page size

TEST_F(UnitMeshCQSingleCardBufferFixture, WritesToRandomBufferTypeAndThenReadsBlocking) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};

    for (const auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running on Device {}", mesh_device->id());
        EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            mesh_device, mesh_device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, WritesToRandomBufferTypeAndThenReadsNonblocking) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};

    for (const auto& mesh_device : devices_) {
        if (not mesh_device->get_devices()[0]->is_mmio_capable()) {
            continue;
        }
        EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            mesh_device, mesh_device->mesh_command_queue(), config));
    }
}

// TODO: Split this into separate tests
TEST_F(UnitMeshCQSingleCardBufferFixture, ShardedBufferL1ReadWrites) {
    std::map<std::string, std::vector<std::array<uint32_t, 2>>> test_params;

    for (const auto& mesh_device : devices_) {
        // This test hangs on Blackhole A0 when using static VCs through static TLBs and there are large number of
        // reads/writes issued
        //  workaround is to use dynamic VC (implemented in UMD)
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
            test_params = {
                {"cores",
                 {{1, 1},
                  {static_cast<uint32_t>(mesh_device->compute_with_storage_grid_size().x),
                   static_cast<uint32_t>(mesh_device->compute_with_storage_grid_size().y)}}},
                {"num_pages", {{3, 65}}},
                {"page_shape", {{32, 32}}}};
        } else {
            test_params = {
                {"cores",
                 {{1, 1},
                  {5, 1},
                  {1, 5},
                  {5, 3},
                  {3, 5},
                  {5, 5},
                  {static_cast<uint32_t>(mesh_device->compute_with_storage_grid_size().x),
                   static_cast<uint32_t>(mesh_device->compute_with_storage_grid_size().y)}}},
                {"num_pages", {{1, 1}, {2, 1}, {1, 2}, {2, 2}, {7, 11}, {3, 65}, {67, 4}, {3, 137}}},
                {"page_shape", {{32, 32}, {1, 4}, {1, 120}, {1, 1024}, {1, 2048}}}};
        }
        for (const std::array<uint32_t, 2> cores : test_params.at("cores")) {
            for (const std::array<uint32_t, 2> num_pages : test_params.at("num_pages")) {
                for (const std::array<uint32_t, 2> page_shape : test_params.at("page_shape")) {
                    for (const TensorMemoryLayout shard_strategy :
                         {TensorMemoryLayout::HEIGHT_SHARDED,
                          TensorMemoryLayout::WIDTH_SHARDED,
                          TensorMemoryLayout::BLOCK_SHARDED}) {
                        for (const uint32_t num_iterations : {
                                 1,
                             }) {
                            BufferStressTestConfigSharded config(num_pages, cores);
                            config.seed = 0;
                            config.num_iterations = num_iterations;
                            config.mem_config = shard_strategy;
                            config.page_shape = page_shape;
                            log_info(
                                tt::LogTest,
                                "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: {}, "
                                "num_iterations: {}",
                                mesh_device->id(),
                                cores[0],
                                cores[1],
                                num_pages[0],
                                num_pages[1],
                                page_shape[0],
                                page_shape[1],
                                enchantum::to_string(shard_strategy).data(),
                                num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                mesh_device, mesh_device->mesh_command_queue(), config, BufferType::L1, false);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, ShardedBufferDRAMReadWrites) {
    for (const auto& mesh_device : devices_) {
        for (const std::array<uint32_t, 2> cores :
             {std::array<uint32_t, 2>{1, 1},
              std::array<uint32_t, 2>{5, 1},
              std::array<uint32_t, 2>{
                  static_cast<uint32_t>(mesh_device->dram_grid_size().x),
                  static_cast<uint32_t>(mesh_device->dram_grid_size().y)}}) {
            for (const std::array<uint32_t, 2> num_pages : {
                     std::array<uint32_t, 2>{1, 1},
                     std::array<uint32_t, 2>{2, 1},
                     std::array<uint32_t, 2>{1, 2},
                     std::array<uint32_t, 2>{2, 2},
                     std::array<uint32_t, 2>{7, 11},
                     std::array<uint32_t, 2>{3, 65},
                     std::array<uint32_t, 2>{67, 4},
                     std::array<uint32_t, 2>{3, 137},
                 }) {
                for (const std::array<uint32_t, 2> page_shape : {
                         std::array<uint32_t, 2>{32, 32},
                         std::array<uint32_t, 2>{1, 4},
                         std::array<uint32_t, 2>{1, 120},
                         std::array<uint32_t, 2>{1, 1024},
                         std::array<uint32_t, 2>{1, 2048},
                     }) {
                    for (const TensorMemoryLayout shard_strategy :
                         {TensorMemoryLayout::HEIGHT_SHARDED,
                          TensorMemoryLayout::WIDTH_SHARDED,
                          TensorMemoryLayout::BLOCK_SHARDED}) {
                        for (const uint32_t num_iterations : {
                                 1,
                             }) {
                            BufferStressTestConfigSharded config(num_pages, cores);
                            config.seed = 0;
                            config.num_iterations = num_iterations;
                            config.mem_config = shard_strategy;
                            config.page_shape = page_shape;
                            log_info(
                                tt::LogTest,
                                "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: "
                                "{}, num_iterations: {}",
                                mesh_device->id(),
                                cores[0],
                                cores[1],
                                num_pages[0],
                                num_pages[1],
                                page_shape[0],
                                page_shape[1],
                                enchantum::to_string(shard_strategy).data(),
                                num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                mesh_device, mesh_device->mesh_command_queue(), config, BufferType::DRAM, false);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, ShardedBufferLargeL1ReadWrites) {
    for (const auto& mesh_device : devices_) {
        for (const std::array<uint32_t, 2> cores : {std::array<uint32_t, 2>{1, 1}, std::array<uint32_t, 2>{2, 3}}) {
            for (const std::array<uint32_t, 2> num_pages : {
                     std::array<uint32_t, 2>{1, 1},
                     std::array<uint32_t, 2>{1, 2},
                     std::array<uint32_t, 2>{2, 3},
                 }) {
                for (const std::array<uint32_t, 2> page_shape : {
                         std::array<uint32_t, 2>{1, 65536},
                         std::array<uint32_t, 2>{1, 65540},
                         std::array<uint32_t, 2>{1, 65568},
                         std::array<uint32_t, 2>{1, 65520},
                         std::array<uint32_t, 2>{1, 132896},
                         std::array<uint32_t, 2>{256, 256},
                         std::array<uint32_t, 2>{336, 272},
                     }) {
                    for (const TensorMemoryLayout shard_strategy :
                         {TensorMemoryLayout::HEIGHT_SHARDED,
                          TensorMemoryLayout::WIDTH_SHARDED,
                          TensorMemoryLayout::BLOCK_SHARDED}) {
                        for (const uint32_t num_iterations : {
                                 1,
                             }) {
                            BufferStressTestConfigSharded config(num_pages, cores);
                            config.seed = 0;
                            config.num_iterations = num_iterations;
                            config.mem_config = shard_strategy;
                            config.page_shape = page_shape;
                            log_info(
                                tt::LogTest,
                                "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: {}, "
                                "num_iterations: {}",
                                mesh_device->id(),
                                cores[0],
                                cores[1],
                                num_pages[0],
                                num_pages[1],
                                page_shape[0],
                                page_shape[1],
                                enchantum::to_string(shard_strategy).data(),
                                num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                mesh_device, mesh_device->mesh_command_queue(), config, BufferType::L1, true);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, ShardedBufferLargeDRAMReadWrites) {
    for (const auto& mesh_device : devices_) {
        for (const std::array<uint32_t, 2> cores : {std::array<uint32_t, 2>{1, 1}, std::array<uint32_t, 2>{6, 1}}) {
            for (const std::array<uint32_t, 2> num_pages : {
                     std::array<uint32_t, 2>{1, 1},
                     std::array<uint32_t, 2>{1, 2},
                     std::array<uint32_t, 2>{2, 3},
                 }) {
                for (const std::array<uint32_t, 2> page_shape : {
                         std::array<uint32_t, 2>{1, 65536},
                         std::array<uint32_t, 2>{1, 65540},
                         std::array<uint32_t, 2>{1, 65568},
                         std::array<uint32_t, 2>{1, 65520},
                         std::array<uint32_t, 2>{1, 132896},
                         std::array<uint32_t, 2>{256, 256},
                         std::array<uint32_t, 2>{336, 272},
                     }) {
                    for (const TensorMemoryLayout shard_strategy :
                         {TensorMemoryLayout::HEIGHT_SHARDED,
                          TensorMemoryLayout::WIDTH_SHARDED,
                          TensorMemoryLayout::BLOCK_SHARDED}) {
                        for (const uint32_t num_iterations : {
                                 1,
                             }) {
                            BufferStressTestConfigSharded config(num_pages, cores);
                            config.seed = 0;
                            config.num_iterations = num_iterations;
                            config.mem_config = shard_strategy;
                            config.page_shape = page_shape;
                            log_info(
                                tt::LogTest,
                                "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: "
                                "{}, num_iterations: {}",
                                mesh_device->id(),
                                cores[0],
                                cores[1],
                                num_pages[0],
                                num_pages[1],
                                page_shape[0],
                                page_shape[1],
                                enchantum::to_string(shard_strategy).data(),
                                num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                mesh_device, mesh_device->mesh_command_queue(), config, BufferType::DRAM, true);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(UnitMeshCQSingleCardBufferFixture, StressWrapTest) {
    if (this->arch_ == tt::ARCH::WORMHOLE_B0) {
        log_info(tt::LogTest, "cannot run this test on WH B0");
        GTEST_SKIP();
        return;  // skip for WH B0
    }

    BufferStressTestConfig config = {
        .page_size = 4096, .max_num_pages_per_buffer = 2000, .num_iterations = 10000, .num_unique_vectors = 20};
    for (const auto& mesh_device : devices_) {
        EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
            mesh_device, mesh_device->mesh_command_queue(), config));
    }
}

}  // end namespace stress_tests

}  // namespace tt::tt_metal
