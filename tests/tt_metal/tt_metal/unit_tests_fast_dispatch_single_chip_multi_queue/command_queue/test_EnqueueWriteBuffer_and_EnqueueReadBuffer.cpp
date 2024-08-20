// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt::tt_metal;


namespace local_test_functions {

bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(Device* device, vector<std::reference_wrapper<CommandQueue>>& cqs, const TestBufferConfig& config) {
    bool pass = true;
    for (const bool use_void_star_api: {true, false}) {

        size_t buf_size = config.num_pages * config.page_size;
        std::vector<std::unique_ptr<Buffer>> buffers;
        std::vector<std::vector<uint32_t>> srcs;
        for (uint i = 0; i < cqs.size(); i++) {
            buffers.push_back(std::make_unique<Buffer>(device, buf_size, config.page_size, config.buftype));
            srcs.push_back(generate_arange_vector(buffers[i]->size()));
            if (use_void_star_api) {
                EnqueueWriteBuffer(cqs[i], *buffers[i], srcs[i].data(), false);
            } else {
                EnqueueWriteBuffer(cqs[i], *buffers[i], srcs[i], false);
            }
        }

        for (uint i = 0; i < cqs.size(); i++) {
            std::vector<uint32_t> result;
            if (use_void_star_api) {
                result.resize(buf_size / sizeof(uint32_t));
                EnqueueReadBuffer(cqs[i], *buffers[i], result.data(), true);
            } else {
                EnqueueReadBuffer(cqs[i], *buffers[i], result, true);
            }
            bool local_pass = (srcs[i] == result);
            pass &= local_pass;
        }
    }

    return pass;
}
}


namespace basic_tests {
namespace dram_tests {

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileToDramBank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileToAllDramBanks) {
    TestBufferConfig config = {
        .num_pages = uint32_t(this->device_->num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileAcrossAllDramBanksTwiceRoundRobin) {
    constexpr uint32_t num_round_robins = 2;
    TestBufferConfig config = {
        .num_pages = num_round_robins * (this->device_->num_banks(BufferType::DRAM)),
        .page_size = 2048,
        .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, Sending131072Pages) {
    // Was a failing case where we used to accidentally program cb num pages to be total
    // pages instead of cb num pages.
    TestBufferConfig config = {
        .num_pages = 131072,
        .page_size = 128,
        .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestNon32BAlignedPageSizeForDram) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestNon32BAlignedPageSizeForDram2) {
    // From stable diffusion read buffer
    TestBufferConfig config = {.num_pages = 8 * 1024, .page_size = 80, .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestPageSizeTooLarge) {
    if (this->arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP(); // This test hanging on wormhole b0
    }
    // Should throw a host error due to the page size not fitting in the consumer CB
    TestBufferConfig config = {.num_pages = 1024, .page_size = 250880 * 2, .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_ANY_THROW(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestIssueMultipleReadWriteCommandsForOneBuffer) {
    uint32_t page_size = 2048;
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    uint32_t command_queue_size = tt::Cluster::instance().get_host_channel_size(this->device_->id(), channel);
    uint32_t num_pages = command_queue_size / page_size;

    TestBufferConfig config = {.num_pages = num_pages, .page_size = page_size, .buftype = BufferType::DRAM};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}


}  // end namespace dram_tests

namespace l1_tests {

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileToL1Bank0) {
    TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::L1};
    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileToAllL1Banks) {
    auto compute_with_storage_grid = this->device_->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, WriteOneTileToAllL1BanksTwiceRoundRobin) {
    auto compute_with_storage_grid = this->device_->compute_with_storage_grid_size();
    TestBufferConfig config = {
        .num_pages = 2 * uint32_t(compute_with_storage_grid.x * compute_with_storage_grid.y),
        .page_size = 2048,
        .buftype = BufferType::L1};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestNon32BAlignedPageSizeForL1) {
    TestBufferConfig config = {.num_pages = 1250, .page_size = 200, .buftype = BufferType::L1};

    CommandQueue& a = this->device_->command_queue(0);
    CommandQueue& b = this->device_->command_queue(1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    EXPECT_TRUE(local_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer_multi_queue(this->device_, cqs, config));
}

}  // end namespace l1_tests
}  // end namespace basic_tests
