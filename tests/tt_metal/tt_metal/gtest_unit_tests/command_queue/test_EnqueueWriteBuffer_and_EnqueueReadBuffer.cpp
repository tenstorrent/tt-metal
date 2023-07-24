#include <memory>

#include "../basic_harness.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

using namespace tt::tt_metal;

struct BufferStressTestConfig {
    // Used for normal write/read tests
    u32 seed;
    u32 num_pages_total;

    u32 page_size;
    u32 max_num_pages_per_buffer;

    // Used for wrap test
    u32 num_iterations;
    u32 num_unique_vectors;
};

namespace local_test_functions {

vector<u32> generate_arange_vector(u32 size_bytes) {
    TT_ASSERT(size_bytes % sizeof(u32) == 0);
    vector<u32> src(size_bytes / sizeof(u32), 0);

    for (u32 i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    return src;
}

bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer(Device* device, CommandQueue& cq, const BufferConfig& config) {
    size_t buf_size = config.num_pages * config.page_size;
    Buffer bufa(device, buf_size, config.page_size, config.buftype);

    vector<u32> src = generate_arange_vector(bufa.size());

    EnqueueWriteBuffer(cq, bufa, src, false);

    vector<u32> result;
    EnqueueReadBuffer(cq, bufa, result, true);

    return src == result;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    Device* device, CommandQueue& cq, const BufferStressTestConfig& config) {
    srand(config.seed);
    bool pass = true;
    u32 num_pages_left = config.num_pages_total;
    while (num_pages_left) {
        u32 num_pages = std::min(rand() % (config.max_num_pages_per_buffer) + 1, num_pages_left);
        num_pages_left -= num_pages;

        u32 buf_size = num_pages * config.page_size;
        vector<u32> src(buf_size / sizeof(u32), 0);

        for (u32 i = 0; i < src.size(); i++) {
            src.at(i) = i;
        }

        BufferType buftype = BufferType::DRAM;
        if ((rand() % 2) == 0) {
            buftype = BufferType::L1;
        }

        Buffer buf(device, buf_size, config.page_size, buftype);
        EnqueueWriteBuffer(cq, buf, src, false);

        vector<u32> res;
        EnqueueReadBuffer(cq, buf, res, true);
        pass &= src == res;
    }
    return pass;
}

bool test_EnqueueWrap_on_EnqueueReadBuffer(Device* device, CommandQueue& cq, const BufferConfig& config) {
    auto [buffer, src] = EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    vector<u32> dst;
    EnqueueReadBuffer(cq, buffer, dst, true);

    return src == dst;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(
    Device* device, CommandQueue& cq, const BufferStressTestConfig& config) {

    srand(config.seed);

    vector<vector<u32>> unique_vectors;
    for (u32 i = 0; i < config.num_unique_vectors; i++) {
        u32 num_pages = rand() % (config.max_num_pages_per_buffer) + 1;
        size_t buf_size = num_pages * config.page_size;
        unique_vectors.push_back(create_random_vector_of_bfloat16(
            buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));
    }

    vector<Buffer> bufs;
    u32 start = 0;
    for (u32 i = 0; i < config.num_iterations; i++) {
        size_t buf_size = unique_vectors[i % unique_vectors.size()].size() * sizeof(u32);
        try {
            bufs.push_back(Buffer(device, buf_size, config.page_size, BufferType::DRAM));
        } catch (const std::exception& e) {
            tt::log_info("Deallocating on iteration {}", i);
            start = i;
            bufs = {Buffer(device, buf_size, config.page_size, BufferType::DRAM)};
        }

        EnqueueWriteBuffer(cq, bufs[bufs.size() - 1], unique_vectors[i % unique_vectors.size()], false);
    }

    tt::log_info("Comparing {} buffers", bufs.size());
    bool pass = true;
    vector<u32> dst;
    u32 idx = start;
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

TEST_F(CommandQueueHarness, WriteOneTileToDramBank0) {
    BufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllDramBanks) {
    BufferConfig config = {
        .num_pages = u32(this->device->cluster()->get_soc_desc(this->pcie_id).get_num_dram_channels()),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr u32 num_round_robins = 2;
    BufferConfig config = {
        .num_pages = num_round_robins * (this->device->cluster()->get_soc_desc(this->pcie_id).get_num_dram_channels()),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, FusedWriteDramBuffersInWhichRemainderBurstSizeDoesNotFitInLocalL1) {
    BufferConfig config = {.num_pages = 4096, .page_size = 22016, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestNon32BAlignedPageSizeForDram) {
    BufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestNon32BAlignedPageSizeForDram2) {
    // From stable diffusion read buffer
    BufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestWrapHostHugepageOnEnqueueReadBuffer) {
    BufferConfig buf_config = {.num_pages = 524270, .page_size = 2048, .buftype = BufferType::DRAM};

    EXPECT_TRUE(local_test_functions::test_EnqueueWrap_on_EnqueueReadBuffer(this->device, *this->cq, buf_config));
}

}  // end namespace dram_tests

namespace l1_tests {

TEST_F(CommandQueueHarness, WriteOneTileToL1Bank0) {
    BufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllL1Banks) {
    BufferConfig config = {
        .num_pages = u32(this->device->cluster()->get_soc_desc(this->pcie_id).compute_and_storage_cores.size()),
        .page_size = 2048,
        .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    BufferConfig config = {
        .num_pages = 2 * u32(this->device->cluster()->get_soc_desc(this->pcie_id).compute_and_storage_cores.size()),
        .page_size = 2048,
        .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestNon32BAlignedPageSizeForL1) {
    BufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, TestBackToBackNon32BAlignedPageSize) {
    constexpr BufferType buff_type = BufferType::L1;

    Buffer bufa(device, 125000, 100, buff_type);
    auto src_a = local_test_functions::generate_arange_vector(bufa.size());
    EnqueueWriteBuffer(*this->cq, bufa, src_a, false);

    Buffer bufb(device, 152000, 152, buff_type);
    auto src_b = local_test_functions::generate_arange_vector(bufb.size());
    EnqueueWriteBuffer(*this->cq, bufb, src_b, false);

    vector<u32> result_a;
    EnqueueReadBuffer(*this->cq, bufa, result_a, true);

    vector<u32> result_b;
    EnqueueReadBuffer(*this->cq, bufb, result_b, true);

    EXPECT_EQ(src_a, result_a);
    EXPECT_EQ(src_b, result_b);
}

}  // end namespace l1_tests
}  // end namespace basic_tests

namespace stress_tests {

TEST_F(CommandQueueHarness, WritesToRandomBufferTypeAndThenReads) {
    BufferStressTestConfig config = {
        .seed = 0, .num_pages_total = 50000, .page_size = 2048, .max_num_pages_per_buffer = 16};
    EXPECT_TRUE(
        local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, StressWrapTest) {
    BufferStressTestConfig config = {
        .page_size = 4096, .max_num_pages_per_buffer = 2000, .num_iterations = 10000, .num_unique_vectors = 20};
    EXPECT_TRUE(
        local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer_wrap(this->device, *this->cq, config));
}

}  // end namespace stress_tests
