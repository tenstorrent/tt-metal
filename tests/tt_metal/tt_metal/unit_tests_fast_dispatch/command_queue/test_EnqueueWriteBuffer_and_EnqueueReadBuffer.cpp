// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt::tt_metal;

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
    uint32_t seed;
    uint32_t num_iterations = 100;

    const std::array<uint32_t, 2> max_num_pages_per_core;
    const std::array<uint32_t, 2> max_num_cores;

    std::array<uint32_t, 2> num_pages_per_core;
    std::array<uint32_t, 2> num_cores;
    std::array<uint32_t, 2> page_shape = {32, 32};
    uint32_t element_size = 1;
    TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;
    bool halo = false;

    BufferStressTestConfigSharded(std::array<uint32_t, 2> pages_per_core, std::array<uint32_t, 2> cores) :
        max_num_pages_per_core(pages_per_core), max_num_cores(cores) {
        this->num_pages_per_core = pages_per_core;
        this->num_cores = cores;
    }

    std::array<uint32_t, 2> tensor2d_shape() {
        return {num_pages_per_core[0] * num_cores[0], num_pages_per_core[1] * num_cores[1]};
    }

    uint32_t num_pages() { return tensor2d_shape()[0] * tensor2d_shape()[1]; }

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
            this->halo,
            this->page_shape,
            this->tensor2d_shape());
    }

    uint32_t page_size() { return page_shape[0] * page_shape[1] * element_size; }
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

template <bool cq_dispatch_only = false>
void test_EnqueueWriteBuffer_and_EnqueueReadBuffer(Device *device, CommandQueue &cq, const TestBufferConfig &config) {
    // Clear out command queue
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t cq_start = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);

    std::vector<uint32_t> cq_zeros((cq_size - cq_start) / sizeof(uint32_t), 0);

    tt::Cluster::instance().write_sysmem(cq_zeros.data(), (cq_size - cq_start), get_absolute_cq_offset(channel, 0, cq_size) + cq_start, mmio_device_id, channel);

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
            Buffer bufa(device, buf_size, config.page_size, config.buftype);

            vector<uint32_t> src = generate_arange_vector(bufa.size());

            if (cq_write) {
                EnqueueWriteBuffer(cq, bufa, src.data(), false);
            } else {
                ::detail::WriteToBuffer(bufa, src);
                if (config.buftype == BufferType::DRAM) {
                    tt::Cluster::instance().dram_barrier(device->id());
                } else {
                    tt::Cluster::instance().l1_barrier(device->id());
                }
            }

            vector<uint32_t> result;
            result.resize(buf_size / sizeof(uint32_t));

            if (cq_write and not cq_read) {
                Finish(cq);
            }

            if (cq_read) {
                EnqueueReadBuffer(cq, bufa, result.data(), true);
            } else {
                ::detail::ReadFromBuffer(bufa, result);
            }

            EXPECT_EQ(src, result);
        }
    }
}

template <bool blocking>
bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    Device *device, CommandQueue &cq, const BufferStressTestConfig &config) {
    srand(config.seed);
    bool pass = true;
    uint32_t num_pages_left = config.num_pages_total;

    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<std::vector<uint32_t>> srcs;
    std::vector<std::vector<uint32_t>> dsts;
    while (num_pages_left) {
        uint32_t num_pages = std::min(rand() % (config.max_num_pages_per_buffer) + 1, num_pages_left);
        num_pages_left -= num_pages;

        uint32_t buf_size = num_pages * config.page_size;
        vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);

        for (uint32_t i = 0; i < src.size(); i++) {
            src[i] = rand();
        }

        BufferType buftype = BufferType::DRAM;
        if ((rand() % 2) == 0) {
            buftype = BufferType::L1;
        }

        std::unique_ptr<Buffer> buf;
        try {
            buf = std::make_unique<Buffer>(device, buf_size, config.page_size, buftype);
        } catch (...) {
            Finish(cq);
            size_t i = 0;
            for (const auto &dst : dsts) {
                EXPECT_EQ(srcs[i++], dst);
            }
            srcs.clear();
            dsts.clear();
            buffers.clear();
            buf = std::make_unique<Buffer>(device, buf_size, config.page_size, buftype);
        }
        EnqueueWriteBuffer(cq, *buf, src, false);
        vector<uint32_t> dst;
        if constexpr (blocking) {
            EnqueueReadBuffer(cq, *buf, dst, true);
            EXPECT_EQ(src, dst);
        } else {
            srcs.push_back(std::move(src));
            dsts.push_back(dst);
            buffers.push_back(std::move(buf));  // Ensures that buffer not destroyed when moved out of scope
            EnqueueReadBuffer(cq, *buffers[buffers.size() - 1], dsts[dsts.size() - 1], false);
        }
    }

    if constexpr (not blocking) {
        Finish(cq);
        size_t i = 0;
        for (const auto &dst : dsts) {
            EXPECT_EQ(srcs[i++], dst);
        }
    }
    return pass;
}

void stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
    Device *device, CommandQueue &cq, BufferStressTestConfigSharded &config, BufferType buftype, bool read_only) {
    srand(config.seed);

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

                uint32_t page_size = config.page_size();
                for (uint32_t i = 0; i < src.size(); i++) {
                    src.at(i) = i;
                }

                Buffer buf(device, buf_size, config.page_size(), buftype, config.mem_config, shard_spec);
                vector<uint32_t> src2 = src;
                if (cq_write) {
                    EnqueueWriteBuffer(cq, buf, src2.data(), false);
                } else {
                    ::detail::WriteToBuffer(buf, src);
                    if (buftype == BufferType::DRAM) {
                        tt::Cluster::instance().dram_barrier(device->id());
                    } else {
                        tt::Cluster::instance().l1_barrier(device->id());
                    }
                }

                if (cq_write and not cq_read) {
                    Finish(cq);
                }

                vector<uint32_t> res;
                res.resize(buf_size / sizeof(uint32_t));
                if (cq_read) {
                    EnqueueReadBuffer(cq, buf, res.data(), true);
                } else {
                    ::detail::ReadFromBuffer(buf, res);
                }
                EXPECT_EQ(src, res);
            }
        }
    }
}

void test_EnqueueWrap_on_EnqueueReadBuffer(Device *device, CommandQueue &cq, const TestBufferConfig &config) {
    auto [buffer, src] = EnqueueWriteBuffer_prior_to_wrap(device, cq, config);
    vector<uint32_t> dst;
    EnqueueReadBuffer(cq, buffer, dst, true);

    EXPECT_EQ(src, dst);
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
    Device *device, CommandQueue &cq, const BufferStressTestConfig &config) {
    srand(config.seed);

    vector<vector<uint32_t>> unique_vectors;
    for (uint32_t i = 0; i < config.num_unique_vectors; i++) {
        uint32_t num_pages = rand() % (config.max_num_pages_per_buffer) + 1;
        size_t buf_size = num_pages * config.page_size;
        unique_vectors.push_back(create_random_vector_of_bfloat16(
            buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));
    }

    vector<std::shared_ptr<Buffer>> bufs;
    uint32_t start = 0;

    for (uint32_t i = 0; i < config.num_iterations; i++) {
        size_t buf_size = unique_vectors[i % unique_vectors.size()].size() * sizeof(uint32_t);
        tt::tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = buf_size,
            .page_size = config.page_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM};
        try {
            bufs.push_back(CreateBuffer(dram_config));
        } catch (const std::exception &e) {
            tt::log_info("Deallocating on iteration {}", i);
            bufs.clear();
            start = i;
            bufs = {CreateBuffer(dram_config)};
        }
        EnqueueWriteBuffer(cq, bufs[bufs.size() - 1], unique_vectors[i % unique_vectors.size()], false);
    }

    tt::log_info("Comparing {} buffers", bufs.size());
    bool pass = true;
    vector<uint32_t> dst;
    uint32_t idx = start;
    for (auto buffer : bufs) {
        EnqueueReadBuffer(cq, buffer, dst, true);
        pass &= dst == unique_vectors[idx % unique_vectors.size()];
        idx++;
    }

    return pass;
}

}  // end namespace local_test_functions

namespace basic_tests {
namespace dram_tests {

TEST_F(CommandQueueSingleCardFixture, WriteOneTileToDramBank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    for (Device *device : devices_) {
        tt::log_info("Running On Device {}", device->id());
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, WriteOneTileToAllDramBanks) {
    for (Device *device : devices_) {
        TestBufferConfig config = {
            .num_pages = uint32_t(device->num_banks(BufferType::DRAM)), .page_size = 2048, .buftype = BufferType::DRAM};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr uint32_t num_round_robins = 2;
    for (Device *device : devices_) {
        TestBufferConfig config = {
            .num_pages = num_round_robins * (device->num_banks(BufferType::DRAM)),
            .page_size = 2048,
            .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, Sending131072Pages) {
    for (Device *device : devices_) {
        TestBufferConfig config = {.num_pages = 131072, .page_size = 128, .buftype = BufferType::DRAM};
        tt::log_info("Running On Device {}", device->id());
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestPageLargerThanAndUnalignedToTransferPage) {
    constexpr uint32_t num_round_robins = 2;
    for (Device *device : devices_) {
        TestBufferConfig config = {
            .num_pages = num_round_robins * (device->num_banks(BufferType::DRAM)),
            .page_size = dispatch_constants::TRANSFER_PAGE_SIZE + 32,
            .buftype = BufferType::DRAM
        };
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestPageLargerThanMaxPrefetchCommandSize) {
    constexpr uint32_t num_round_robins = 1;
    for (Device *device : devices_) {
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
        TestBufferConfig config = {
            .num_pages = 1,
            .page_size = max_prefetch_command_size + 2048,
            .buftype = BufferType::DRAM
        };
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestUnalignedPageLargerThanMaxPrefetchCommandSize) {
    constexpr uint32_t num_round_robins = 1;
    for (Device *device : devices_) {
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
        uint32_t unaligned_page_size = max_prefetch_command_size + 4;
        TestBufferConfig config = {
            .num_pages = 1,
            .page_size = unaligned_page_size,
            .buftype = BufferType::DRAM
        };
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestNon32BAlignedPageSizeForDram) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    for (Device *device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestNon32BAlignedPageSizeForDram2) {
    // From stable diffusion read buffer
    TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    for (Device *device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueFixture, TestPageSizeTooLarge) {
    // Should throw a host error due to the page size not fitting in the consumer CB
    TestBufferConfig config = {.num_pages = 1024, .page_size = 250880 * 2, .buftype = BufferType::DRAM};

    EXPECT_ANY_THROW((local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
        this->device_, this->device_->command_queue(), config)));
}

// Requires enqueue write buffer
TEST_F(CommandQueueSingleCardFixture, TestWrapHostHugepageOnEnqueueReadBuffer) {
    for (Device *device : this->devices_) {
        tt::log_info("Running On Device {}", device->id());
        uint32_t page_size = 2048;
        uint32_t command_issue_region_size = device->sysmem_manager().get_issue_queue_size(0);
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        uint32_t cq_start = dispatch_constants::get(dispatch_core_type).get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);

        uint32_t max_command_size = command_issue_region_size - cq_start;
        uint32_t buffer = 14240;
        uint32_t buffer_size = max_command_size - buffer;
        uint32_t num_pages = buffer_size / page_size;

        TestBufferConfig buf_config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};
        local_test_functions::test_EnqueueWrap_on_EnqueueReadBuffer(device, device->command_queue(), buf_config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    for (Device *device : this->devices_) {
        tt::log_info("Running On Device {}", device->id());
        uint32_t page_size = 2048;
        uint32_t command_queue_size = device->sysmem_manager().get_cq_size();
        uint32_t num_pages = command_queue_size / page_size;

        TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            device, device->command_queue(), config);
    }
}

// Test that command queue wraps when buffer available space in completion region is less than a page
TEST_F(CommandQueueSingleCardFixture, TestWrapCompletionQOnInsufficientSpace) {
    uint32_t large_page_size = 8192;  // page size for first and third read
    uint32_t small_page_size = 2048;  // page size for second read

    for (Device *device : devices_) {
        tt::log_info("Running On Device {}", device->id());
        uint32_t command_completion_region_size = device->sysmem_manager().get_completion_queue_size(0);

        uint32_t first_buffer_size = tt::round_up(command_completion_region_size * 0.95, large_page_size);

        uint32_t space_after_first_buffer = command_completion_region_size - first_buffer_size;
        // leave only small_page_size * 2 B of space in the completion queue
        uint32_t num_pages_second_buffer = (space_after_first_buffer / small_page_size) - 2;

        Buffer buff_1(device, first_buffer_size, large_page_size, BufferType::DRAM);
        auto src_1 = local_test_functions::generate_arange_vector(buff_1.size());
        EnqueueWriteBuffer(device->command_queue(), buff_1, src_1, false);
        vector<uint32_t> result_1;
        EnqueueReadBuffer(device->command_queue(), buff_1, result_1, true);
        EXPECT_EQ(src_1, result_1);

        Buffer buff_2(device, num_pages_second_buffer * small_page_size, small_page_size, BufferType::DRAM);
        auto src_2 = local_test_functions::generate_arange_vector(buff_2.size());
        EnqueueWriteBuffer(device->command_queue(), buff_2, src_2, false);
        vector<uint32_t> result_2;
        EnqueueReadBuffer(device->command_queue(), buff_2, result_2, true);
        EXPECT_EQ(src_2, result_2);

        Buffer buff_3(device, 32 * large_page_size, large_page_size, BufferType::DRAM);
        auto src_3 = local_test_functions::generate_arange_vector(buff_3.size());
        EnqueueWriteBuffer(device->command_queue(), buff_3, src_3, false);
        vector<uint32_t> result_3;
        EnqueueReadBuffer(device->command_queue(), buff_3, result_3, true);
        EXPECT_EQ(src_3, result_3);
    }
}

// Test that command queue wraps when buffer read needs to be split into multiple enqueue_read_buffer commands and
// available space in completion region is less than a page
TEST_F(CommandQueueSingleCardFixture, TestWrapCompletionQOnInsufficientSpace2) {
    // Using default 75-25 issue and completion queue split
    for (Device *device : devices_) {
        tt::log_info("Running On Device {}", device->id());
        uint32_t command_completion_region_size = device->sysmem_manager().get_completion_queue_size(0);

        uint32_t num_pages_buff_1 = 9;
        uint32_t page_size_buff_1 = 2048;
        Buffer buff_1(device, num_pages_buff_1 * page_size_buff_1, page_size_buff_1, BufferType::DRAM);
        uint32_t space_after_buff_1 = command_completion_region_size - buff_1.size();

        uint32_t page_size = 8192;
        uint32_t desired_remaining_space_before_wrap = 6144;
        uint32_t avail_space_for_wrapping_buffer = space_after_buff_1 - desired_remaining_space_before_wrap;
        uint32_t num_pages_for_wrapping_buffer = (avail_space_for_wrapping_buffer / page_size) + 4;

        auto src_1 = local_test_functions::generate_arange_vector(buff_1.size());
        EnqueueWriteBuffer(device->command_queue(), buff_1, src_1, false);
        vector<uint32_t> result_1;
        EnqueueReadBuffer(device->command_queue(), buff_1, result_1, true);
        EXPECT_EQ(src_1, result_1);

        Buffer wrap_buff(device, num_pages_for_wrapping_buffer * page_size, page_size, BufferType::DRAM);
        auto src_2 = local_test_functions::generate_arange_vector(wrap_buff.size());
        EnqueueWriteBuffer(device->command_queue(), wrap_buff, src_2, false);
        vector<uint32_t> result_2;
        EnqueueReadBuffer(device->command_queue(), wrap_buff, result_2, true);
        EXPECT_EQ(src_2, result_2);
    }
}

// TODO: add test for wrapping with non aligned page sizes

}  // end namespace dram_tests

namespace l1_tests {

TEST_F(CommandQueueSingleCardFixture, WriteOneTileToL1Bank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
    for (Device *device : devices_) {
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, WriteOneTileToAllL1Banks) {
    for (Device *device : devices_) {
        auto compute_with_storage_grid = device->compute_with_storage_grid_size();
        TestBufferConfig config = {
            .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    for (Device *device : devices_) {
        auto compute_with_storage_grid = device->compute_with_storage_grid_size();
        TestBufferConfig config = {
            .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
            .page_size = 2048,
            .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestNon32BAlignedPageSizeForL1) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    for (Device *device : devices_) {
        if (device->is_mmio_capable()) {
            continue;
        }
        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestBackToBackNon32BAlignedPageSize) {
    constexpr BufferType buff_type = BufferType::L1;

    for (Device *device : devices_) {
        Buffer bufa(device, 125000, 100, buff_type);
        auto src_a = local_test_functions::generate_arange_vector(bufa.size());
        EnqueueWriteBuffer(device->command_queue(), bufa, src_a, false);

        Buffer bufb(device, 152000, 152, buff_type);
        auto src_b = local_test_functions::generate_arange_vector(bufb.size());
        EnqueueWriteBuffer(device->command_queue(), bufb, src_b, false);

        vector<uint32_t> result_a;
        EnqueueReadBuffer(device->command_queue(), bufa, result_a, true);

        vector<uint32_t> result_b;
        EnqueueReadBuffer(device->command_queue(), bufb, result_b, true);

        EXPECT_EQ(src_a, result_a);
        EXPECT_EQ(src_b, result_b);
    }
}

// This case was failing for FD v1.3 design
TEST_F(CommandQueueSingleCardFixture, TestLargeBuffer4096BPageSize) {
    constexpr BufferType buff_type = BufferType::L1;

    for (Device *device : devices_) {
        TestBufferConfig config = {.num_pages = 512, .page_size = 4096, .buftype = BufferType::L1};

        local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(device, device->command_queue(), config);
    }
}

}  // end namespace l1_tests

TEST_F(CommandQueueSingleCardFixture, TestNonblockingReads) {
    constexpr BufferType buff_type = BufferType::L1;

    for (auto device : devices_) {
        Buffer bufa(device, 2048, 2048, buff_type);
        auto src_a = local_test_functions::generate_arange_vector(bufa.size());
        EnqueueWriteBuffer(device->command_queue(), bufa, src_a, false);

        Buffer bufb(device, 2048, 2048, buff_type);
        auto src_b = local_test_functions::generate_arange_vector(bufb.size());
        EnqueueWriteBuffer(device->command_queue(), bufb, src_b, false);

        vector<uint32_t> result_a;
        EnqueueReadBuffer(device->command_queue(), bufa, result_a, false);

        vector<uint32_t> result_b;
        EnqueueReadBuffer(device->command_queue(), bufb, result_b, false);
        Finish(device->command_queue());

        EXPECT_EQ(src_a, result_a);
        EXPECT_EQ(src_b, result_b);
    }
}

}  // end namespace basic_tests

namespace stress_tests {

// TODO: Add stress test that vary page size

TEST_F(CommandQueueSingleCardFixture, WritesToRandomBufferTypeAndThenReadsBlocking) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};

    for (Device *device : devices_) {
        tt::log_info("Running on Device {}", device->id());
        EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, WritesToRandomBufferTypeAndThenReadsNonblocking) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};

    for (Device *device : devices_) {
        if (not device->is_mmio_capable())
            continue;
        EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer<true>(
            device, device->command_queue(), config));
    }
}

// TODO: Split this into separate tests
TEST_F(CommandQueueSingleCardFixture, ShardedBufferL1ReadWrites) {
    std::map<std::string, std::vector<std::array<uint32_t, 2>>> test_params;

    for (Device *device : devices_) {
        // This test hangs on Blackhole A0 when using static VCs through static TLBs and there are large number of reads/writes issued
        //  workaround is to use dynamic VC (implemented in UMD)
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            test_params = {
                {"cores",
                 {{1, 1},
                  {static_cast<uint32_t>(device->compute_with_storage_grid_size().x),
                   static_cast<uint32_t>(device->compute_with_storage_grid_size().y)}}},
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
                  {static_cast<uint32_t>(device->compute_with_storage_grid_size().x),
                   static_cast<uint32_t>(device->compute_with_storage_grid_size().y)}}},
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
                            tt::log_info(tt::LogTest, "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: {}, num_iterations: {}", device->id(), cores[0],cores[1], num_pages[0],num_pages[1], page_shape[0],page_shape[1], magic_enum::enum_name(shard_strategy).data(), num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                device, device->command_queue(), config, BufferType::L1, false);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, ShardedBufferDRAMReadWrites) {
    for (Device *device : devices_) {
        for (const std::array<uint32_t, 2> cores :
             {std::array<uint32_t, 2>{1, 1},
              std::array<uint32_t, 2>{5, 1},
              std::array<uint32_t, 2>{static_cast<uint32_t>(device->dram_grid_size().x), static_cast<uint32_t>(device->dram_grid_size().y)}}) {
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
                            tt::log_info(
                                tt::LogTest,
                                    "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: "
                                    "{}, num_iterations: {}",
                                    device->id(),
                                    cores[0],
                                    cores[1],
                                    num_pages[0],
                                    num_pages[1],
                                    page_shape[0],
                                    page_shape[1],
                                    magic_enum::enum_name(shard_strategy).data(),
                                    num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                device, device->command_queue(), config, BufferType::DRAM, false);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, ShardedBufferLargeL1ReadWrites) {
    for (Device *device : devices_) {
        for (const std::array<uint32_t, 2> cores :
             {std::array<uint32_t, 2>{1, 1},
              std::array<uint32_t, 2>{2, 3}}) {
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
                            tt::log_info(tt::LogTest, "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: {}, num_iterations: {}", device->id(), cores[0],cores[1], num_pages[0],num_pages[1], page_shape[0],page_shape[1], magic_enum::enum_name(shard_strategy).data(), num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                device, device->command_queue(), config, BufferType::L1, true);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, ShardedBufferLargeDRAMReadWrites) {
    for (Device *device : devices_) {
        for (const std::array<uint32_t, 2> cores :
             {std::array<uint32_t, 2>{1, 1},
              std::array<uint32_t, 2>{6, 1}}) {
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
                            tt::log_info(
                                tt::LogTest,
                                    "Device: {} cores: [{},{}] num_pages: [{},{}] page_shape: [{},{}], shard_strategy: "
                                    "{}, num_iterations: {}",
                                    device->id(),
                                    cores[0],
                                    cores[1],
                                    num_pages[0],
                                    num_pages[1],
                                    page_shape[0],
                                    page_shape[1],
                                    magic_enum::enum_name(shard_strategy).data(),
                                    num_iterations);
                            local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
                                device, device->command_queue(), config, BufferType::DRAM, true);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CommandQueueFixture, StressWrapTest) {
    const char *arch = getenv("ARCH_NAME");
    if (strcasecmp(arch, "wormhole_b0") == 0) {
        tt::log_info("cannot run this test on WH B0");
        GTEST_SKIP();
        return;  // skip for WH B0
    }

    BufferStressTestConfig config = {
        .page_size = 4096, .max_num_pages_per_buffer = 2000, .num_iterations = 10000, .num_unique_vectors = 20};
    EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
        this->device_, this->device_->command_queue(), config));
}

}  // end namespace stress_tests
