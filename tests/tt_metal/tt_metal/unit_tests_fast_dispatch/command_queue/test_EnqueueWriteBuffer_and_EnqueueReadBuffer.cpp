// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/detail/tt_metal.hpp"

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

class BufferStressTestConfigSharded{
    public:
        uint32_t seed;
        uint32_t num_iterations = 100;

        const std::array<uint32_t,2> max_num_pages_per_core;
        const std::array<uint32_t,2> max_num_cores;

        std::array<uint32_t,2> num_pages_per_core;
        std::array<uint32_t,2> num_cores;
        std::array<uint32_t, 2> page_shape = {32,32};
        uint32_t element_size = 1;
        TensorMemoryLayout mem_config = TensorMemoryLayout::HEIGHT_SHARDED;
        ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;
        bool halo = false;

        BufferStressTestConfigSharded(std::array<uint32_t,2> pages_per_core,
                        std::array<uint32_t, 2> cores):
                        max_num_pages_per_core(pages_per_core), max_num_cores(cores)
                        {
                            this->num_pages_per_core = pages_per_core;
                            this->num_cores = cores;
                        }

        std::array<uint32_t, 2> tensor2d_shape(){
            return {num_pages_per_core[0]*num_cores[0],
                    num_pages_per_core[1]*num_cores[1]};
        }

        uint32_t num_pages(){
            return tensor2d_shape()[0] * tensor2d_shape()[1];
        }

        std::array<uint32_t, 2> shard_shape(){
            return {num_pages_per_core[0] * page_shape[0], num_pages_per_core[1] * page_shape[1]};
        }

        CoreRangeSet shard_grid(){
            return CoreRangeSet(std::set<CoreRange>(
            {
                CoreRange(CoreCoord(0, 0),
                CoreCoord(this->num_cores[0] -1, this->num_cores[1] - 1))
            }));

        }

        ShardSpecBuffer shard_parameters(){
            return ShardSpecBuffer(
                        this->shard_grid(),
                        this->shard_shape(),
                        this->shard_orientation,
                        this->halo,
                        this->page_shape,
                        this->tensor2d_shape()
                        );
        }

        uint32_t page_size(){
            return page_shape[0] * page_shape[1] * element_size;
        }
};

namespace local_test_functions {

vector<uint32_t> generate_arange_vector(uint32_t size_bytes) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0);
    vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    return src;
}

bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer(Device* device, CommandQueue& cq, const TestBufferConfig& config) {
    bool pass = true;
    for (const bool use_void_star_api: {true, false}) {

        size_t buf_size = config.num_pages * config.page_size;
        Buffer bufa(device, buf_size, config.page_size, config.buftype);

        vector<uint32_t> src = generate_arange_vector(bufa.size());

        if (use_void_star_api) {
            EnqueueWriteBuffer(cq, bufa, src.data(), false);
        } else {
            EnqueueWriteBuffer(cq, bufa, src, false);
        }
        vector<uint32_t> result;
        if (use_void_star_api) {
            result.resize(buf_size / sizeof(uint32_t));
            EnqueueReadBuffer(cq, bufa, result.data(), true);
        } else {
            EnqueueReadBuffer(cq, bufa, result, true);
        }
        pass &= (src == result);
    }

    return pass;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    Device* device, CommandQueue& cq, const BufferStressTestConfig& config) {
    srand(config.seed);
    bool pass = true;
    uint32_t num_pages_left = config.num_pages_total;
    while (num_pages_left) {
        uint32_t num_pages = std::min(rand() % (config.max_num_pages_per_buffer) + 1, num_pages_left);
        num_pages_left -= num_pages;

        uint32_t buf_size = num_pages * config.page_size;
        vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);

        for (uint32_t i = 0; i < src.size(); i++) {
            src.at(i) = i;
        }

        BufferType buftype = BufferType::DRAM;
        if ((rand() % 2) == 0) {
            buftype = BufferType::L1;
        }

        Buffer buf(device, buf_size, config.page_size, buftype);
        EnqueueWriteBuffer(cq, buf, src, false);

        vector<uint32_t> res;
        EnqueueReadBuffer(cq, buf, res, true);
        pass &= src == res;
    }
    return pass;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(
    Device* device, CommandQueue& cq, BufferStressTestConfigSharded config) {
    srand(config.seed);
    bool pass = true;

    // first keep num_pages_per_core consistent and increase num_cores
    for(uint32_t iteration_id = 0; iteration_id < config.num_iterations; iteration_id++){
        uint32_t num_cores_outer = rand() % (config.max_num_cores[1]) + 1;

        config.num_cores[1] = num_cores_outer;
        auto shard_spec = config.shard_parameters();

        // explore a tensor_shape , keeping inner pages constant
        uint32_t num_pages = config.num_pages();

        uint32_t buf_size = num_pages * config.page_size();
        vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);


        uint32_t page_size = config.page_size();
        for (uint32_t i = 0; i < src.size(); i++) {
            src.at(i) = i;
        }

        BufferType buftype = BufferType::L1;

        Buffer buf(device, buf_size, config.page_size(), buftype, config.mem_config, shard_spec);
        EnqueueWriteBuffer(cq, buf, src, false);

        vector<uint32_t> res;
        EnqueueReadBuffer(cq, buf, res, true);
        pass &= src == res;
    }
    return pass;
}


bool test_EnqueueWrap_on_EnqueueReadBuffer(Device* device, CommandQueue& cq, const TestBufferConfig& config) {
    auto [buffer, src] = EnqueueWriteBuffer_prior_to_wrap(device, cq, config);
    vector<uint32_t> dst;
    EnqueueReadBuffer(cq, buffer, dst, true);

    return src == dst;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
    Device* device, CommandQueue& cq, const BufferStressTestConfig& config) {

    srand(config.seed);

    vector<vector<uint32_t>> unique_vectors;
    for (uint32_t i = 0; i < config.num_unique_vectors; i++) {
        uint32_t num_pages = rand() % (config.max_num_pages_per_buffer) + 1;
        size_t buf_size = num_pages * config.page_size;
        unique_vectors.push_back(create_random_vector_of_bfloat16(
            buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));
    }

    vector<Buffer> bufs;
    uint32_t start = 0;


    for (uint32_t i = 0; i < config.num_iterations; i++) {
        size_t buf_size = unique_vectors[i % unique_vectors.size()].size() * sizeof(uint32_t);
        tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device= device,
                    .size = buf_size,
                    .page_size = config.page_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        try {
            bufs.push_back(CreateBuffer(dram_config));
        } catch (const std::exception& e) {
            tt::log_info("Deallocating on iteration {}", i);
            start = i;
            bufs = {CreateBuffer(dram_config)};
        }

        EnqueueWriteBuffer(cq, bufs[bufs.size() - 1], unique_vectors[i % unique_vectors.size()], false);
    }

    tt::log_info("Comparing {} buffers", bufs.size());
    bool pass = true;
    vector<uint32_t> dst;
    uint32_t idx = start;
    for (Buffer& buffer : bufs) {
        EnqueueReadBuffer(cq, buffer, dst, true);
        pass &= dst == unique_vectors[idx % unique_vectors.size()];
        idx++;
    }

    return pass;
}

}  // end namespace local_test_functions

namespace basic_tests {
namespace dram_tests {

TEST_F(CommandQueueFixture, WriteOneTileToDramBank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, WriteOneTileToAllDramBanks) {
    TestBufferConfig config = {
        .num_pages = uint32_t(this->device_->num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr uint32_t num_round_robins = 2;
    TestBufferConfig config = {
        .num_pages = num_round_robins * (this->device_->num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, Sending131072Pages) {
    // Was a failing case where we used to accidentally program cb num pages to be total
    // pages instead of cb num pages.
    TestBufferConfig config = {
        .num_pages = 131072,
        .page_size = 128,
        .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestNon32BAlignedPageSizeForDram) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestNon32BAlignedPageSizeForDram2) {
    // From stable diffusion read buffer
    TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestPageSizeTooLarge) {
    if (this->arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP(); // This test hanging on wormhole b0
    }
    // Should throw a host error due to the page size not fitting in the consumer CB
    TestBufferConfig config = {.num_pages = 1024, .page_size = 250880 * 2, .buftype = BufferType::DRAM};

    EXPECT_ANY_THROW(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestWrapHostHugepageOnEnqueueReadBuffer) {
    uint32_t page_size = 2048;
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    uint32_t command_queue_size = tt::Cluster::instance().get_host_channel_size(this->device_->id(), channel);
    uint32_t command_issue_region_size = command_queue_size * 0.75;

    uint32_t max_command_size = command_issue_region_size - CQ_START;
    uint32_t buffer = 14240;
    uint32_t buffer_size = max_command_size - (buffer + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    uint32_t num_pages = buffer_size / page_size;

    TestBufferConfig buf_config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};
    CommandQueue a(this->device_, 0);
    EXPECT_TRUE(local_test_functions::test_EnqueueWrap_on_EnqueueReadBuffer(this->device_, a, buf_config));
}

TEST_F(CommandQueueFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    uint32_t page_size = 2048;
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    uint32_t command_queue_size = tt::Cluster::instance().get_host_channel_size(this->device_->id(), channel);
    uint32_t num_pages = command_queue_size / page_size;

    TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

// Test that command queue wraps when buffer available space in completion region is less than a page
TEST_F(CommandQueueFixture, TestWrapCompletionQOnInsufficientSpace) {
    uint32_t large_page_size = 8192; // page size for first and third read
    uint32_t small_page_size = 2048; // page size for second read

    // Using default 75-25 issue and completion queue split
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    uint32_t command_queue_size = tt::Cluster::instance().get_host_channel_size(this->device_->id(), channel);
    uint32_t command_completion_region_size = command_queue_size * 0.25;

    uint32_t first_buffer_size = tt::round_up(command_completion_region_size * 0.95, large_page_size);

    uint32_t space_after_first_buffer = command_completion_region_size - first_buffer_size;
    // leave only small_page_size * 2 B of space in the completion queue
    uint32_t num_pages_second_buffer = (space_after_first_buffer / small_page_size) - 2;

    Buffer buff_1(this->device_, first_buffer_size, large_page_size, BufferType::DRAM);
    auto src_1 = local_test_functions::generate_arange_vector(buff_1.size());
    EnqueueWriteBuffer(*this->cmd_queue_, buff_1, src_1, false);
    vector<uint32_t> result_1;
    EnqueueReadBuffer(*this->cmd_queue_, buff_1, result_1, true);
    EXPECT_EQ(src_1, result_1);

    Buffer buff_2(this->device_, num_pages_second_buffer * small_page_size, small_page_size, BufferType::DRAM);
    auto src_2 = local_test_functions::generate_arange_vector(buff_2.size());
    EnqueueWriteBuffer(*this->cmd_queue_, buff_2, src_2, false);
    vector<uint32_t> result_2;
    EnqueueReadBuffer(*this->cmd_queue_, buff_2, result_2, true);
    EXPECT_EQ(src_2, result_2);

    Buffer buff_3(this->device_, 32 * large_page_size, large_page_size, BufferType::DRAM);
    auto src_3 = local_test_functions::generate_arange_vector(buff_3.size());
    EnqueueWriteBuffer(*this->cmd_queue_, buff_3, src_3, false);
    vector<uint32_t> result_3;
    EnqueueReadBuffer(*this->cmd_queue_, buff_3, result_3, true);
    EXPECT_EQ(src_3, result_3);
}

// Test that command queue wraps when buffer read needs to be split into multiple enqueue_read_buffer commands and available space in completion region is less than a page
TEST_F(CommandQueueFixture, TestWrapCompletionQOnInsufficientSpace2) {
    // Using default 75-25 issue and completion queue split
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    uint32_t command_queue_size = tt::Cluster::instance().get_host_channel_size(this->device_->id(), channel);
    uint32_t command_completion_region_size = command_queue_size * 0.25;

    uint32_t num_pages_buff_1 = 9;
    uint32_t page_size_buff_1 = 2048;
    Buffer buff_1(this->device_, num_pages_buff_1 * page_size_buff_1, page_size_buff_1, BufferType::DRAM);
    uint32_t space_after_buff_1 = command_completion_region_size - buff_1.size();

    uint32_t page_size = 8192;
    uint32_t desired_remaining_space_before_wrap = 6144;
    uint32_t avail_space_for_wrapping_buffer = space_after_buff_1 - desired_remaining_space_before_wrap;
    uint32_t num_pages_for_wrapping_buffer = (avail_space_for_wrapping_buffer / page_size) + 4;

    auto src_1 = local_test_functions::generate_arange_vector(buff_1.size());
    EnqueueWriteBuffer(*this->cmd_queue_, buff_1, src_1, false);
    vector<uint32_t> result_1;
    EnqueueReadBuffer(*this->cmd_queue_, buff_1, result_1, true);
    EXPECT_EQ(src_1, result_1);

    Buffer wrap_buff(this->device_, num_pages_for_wrapping_buffer * page_size, page_size, BufferType::DRAM);
    auto src_2 = local_test_functions::generate_arange_vector(wrap_buff.size());
    EnqueueWriteBuffer(*this->cmd_queue_, wrap_buff, src_2, false);
    vector<uint32_t> result_2;
    EnqueueReadBuffer(*this->cmd_queue_, wrap_buff, result_2, true);
    EXPECT_EQ(src_2, result_2);
}


}  // end namespace dram_tests

namespace l1_tests {

TEST_F(CommandQueueFixture, WriteOneTileToL1Bank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, WriteOneTileToAllL1Banks) {
    auto compute_with_storage_grid = this->device_->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    auto compute_with_storage_grid = this->device_->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestNon32BAlignedPageSizeForL1) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, *this->cmd_queue_, config));
}

TEST_F(CommandQueueFixture, TestBackToBackNon32BAlignedPageSize) {
    constexpr BufferType buff_type = BufferType::L1;

    Buffer bufa(device_, 125000, 100, buff_type);
    auto src_a = local_test_functions::generate_arange_vector(bufa.size());
    EnqueueWriteBuffer(*this->cmd_queue_, bufa, src_a, false);

    Buffer bufb(device_, 152000, 152, buff_type);
    auto src_b = local_test_functions::generate_arange_vector(bufb.size());
    EnqueueWriteBuffer(*this->cmd_queue_, bufb, src_b, false);

    vector<uint32_t> result_a;
    EnqueueReadBuffer(*this->cmd_queue_, bufa, result_a, true);

    vector<uint32_t> result_b;
    EnqueueReadBuffer(*this->cmd_queue_, bufb, result_b, true);

    EXPECT_EQ(src_a, result_a);
    EXPECT_EQ(src_b, result_b);
}

}  // end namespace l1_tests
}  // end namespace basic_tests

namespace stress_tests {

TEST_F(CommandQueueFixture, WritesToRandomBufferTypeAndThenReads) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};
    EXPECT_TRUE(
        local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device_, tt::tt_metal::detail::GetCommandQueue(this->device_), config));
}


TEST_F(CommandQueueFixture, ShardedBufferReadWrites) {
    BufferStressTestConfigSharded config({2,2}, {4,2});
    config.seed = 0;
    config.num_iterations = 100;

    EXPECT_TRUE(
        local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_sharded(this->device_, tt::tt_metal::detail::GetCommandQueue(this->device_), config));
}

TEST_F(CommandQueueFixture, StressWrapTest) {
    const char* arch = getenv("ARCH_NAME");
    if ( strcasecmp(arch,"wormhole_b0") == 0 ) {
      tt::log_info("cannot run this test on WH B0");
      GTEST_SKIP();
      return; //skip for WH B0
    }

    BufferStressTestConfig config = {
        .page_size = 4096, .max_num_pages_per_buffer = 2000, .num_iterations = 10000, .num_unique_vectors = 20};
    EXPECT_TRUE(
        local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(this->device_, tt::tt_metal::detail::GetCommandQueue(this->device_), config));
}

}  // end namespace stress_tests
