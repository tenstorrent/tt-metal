#include <memory>

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "../basic_harness.hpp"


struct BufferConfig {
    u32 num_pages;
    u32 page_size;
    BufferType buftype;
    u32 bank_start;
};

struct BufferStressTestConfig {
    u32 seed;
    u32 num_pages_total;
    u32 page_size;
    u32 max_num_pages_per_buffer;
};

namespace local_test_functions {
bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer(
    Device* device, CommandQueue& cq, const BufferConfig& config) {
    size_t buf_size = config.num_pages * config.page_size;
    Buffer bufa(device, buf_size, config.bank_start, config.page_size, config.buftype);
    vector<u32> src(buf_size / sizeof(u32), 0);

    for (u32 i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }

    EnqueueWriteBuffer(cq, bufa, src, false);

    vector<u32> result;
    EnqueueReadBuffer(cq, bufa, result, true);

    return src == result;
}

bool stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(Device* device, CommandQueue& cq, const BufferStressTestConfig& config) {
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

        Buffer buf(device, buf_size, 0, config.page_size, buftype);
        EnqueueWriteBuffer(cq, buf, src, false);

        vector<u32> res;
        EnqueueReadBuffer(cq, buf, res, true);
        pass &= src == res;
    }
    return pass;
}

}  // end namespace local_test_functions

namespace basic_tests {
namespace dram_tests {

TEST_F(CommandQueueHarness, WriteOneTileToDramBank0) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM, .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllDramBanks) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferConfig config = {
        .num_pages = u32(this->device->cluster()->get_soc_desc(this->pcie_id).get_num_dram_channels()),
        .page_size = 2048,
        .buftype = BufferType::DRAM,
        .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    constexpr u32 num_round_robins = 2;
    BufferConfig config = {
        .num_pages = num_round_robins * (this->device->cluster()->get_soc_desc(this->pcie_id).get_num_dram_channels()),
        .page_size = 2048,
        .buftype = BufferType::DRAM,
        .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}
}  // end namespace dram_tests

namespace l1_tests {

TEST_F(CommandQueueHarness, WriteOneTileToL1Bank0) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1, .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllL1Banks) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferConfig config = {
        .num_pages = u32(this->device->cluster()->get_soc_desc(this->pcie_id).compute_and_storage_cores.size()),
        .page_size = 2048,
        .buftype = BufferType::L1,
        .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

TEST_F(CommandQueueHarness, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferConfig config = {
        .num_pages = 2 * u32(this->device->cluster()->get_soc_desc(this->pcie_id).compute_and_storage_cores.size()),
        .page_size = 2048,
        .buftype = BufferType::L1,
        .bank_start = 0};

    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

}  // end namespace l1_tests
}  // end namespace basic_tests

namespace stress_tests {

TEST_F(CommandQueueHarness, WritesToRandomBufferTypeAndThenReads) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    BufferStressTestConfig config = {
        .seed = 0,
        .num_pages_total = 50000,
        .page_size = 2048,
        .max_num_pages_per_buffer = 16
    };
    EXPECT_TRUE(local_test_functions::stress_test_EnqueueWriteBuffer_and_EnqueueReadBuffer(this->device, *this->cq, config));
}

} // end namespace stress_tests
