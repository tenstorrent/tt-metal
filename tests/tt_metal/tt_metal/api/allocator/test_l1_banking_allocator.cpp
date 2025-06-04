// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/host_api.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"

namespace unit_tests::test_l1_banking_allocator {

uint64_t get_alloc_limit(const tt::tt_metal::IDevice* device) {
    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto dispatch_core_config = tt::tt_metal::get_dispatch_core_config();
    auto storage_core_bank_size =
        tt::get_storage_core_bank_size(device->id(), device->num_hw_cqs(), dispatch_core_config);
    const uint32_t allocator_alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    const uint32_t interleaved_l1_bank_size = storage_core_bank_size.has_value()
                                                  ? storage_core_bank_size.value()
                                                  : (soc_desc.worker_l1_size - l1_unreserved_base);
    uint32_t storage_core_unreserved_base =
        ((tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
              tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::MAILBOX) +
          allocator_alignment - 1) /
         allocator_alignment) *
        allocator_alignment;
    uint64_t alloc_limit = interleaved_l1_bank_size - storage_core_unreserved_base;
    return alloc_limit;
}

}  // namespace unit_tests::test_l1_banking_allocator

namespace tt::tt_metal {

TEST_F(DeviceSingleCardBufferFixture, TestL1BuffersAllocatedTopDown) {
    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    size_t total_size_bytes = 0;

    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->device_);

    std::vector<std::shared_ptr<Buffer>> buffers;
    int alloc_size_idx = 0;
    uint32_t total_buffer_size = 0;
    while (total_size_bytes < alloc_limit) {
        uint32_t buffer_size = alloc_sizes.at(alloc_size_idx);
        alloc_size_idx = (alloc_size_idx + 1) % alloc_sizes.size();
        if (total_buffer_size + buffer_size >= alloc_limit) {
            break;
        }
        auto buffer =
            tt::tt_metal::Buffer::create(this->device_, buffer_size, buffer_size, tt::tt_metal::BufferType::L1);
        buffers.emplace_back(std::move(buffer));
        total_buffer_size += buffer_size;
        EXPECT_EQ(buffers.back()->address(), this->device_->l1_size_per_core() - total_buffer_size);
    }
    buffers.clear();
}

TEST_F(DeviceSingleCardBufferFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->device_);
    tt::tt_metal::InterleavedBufferConfig l1_config{
        .device = this->device_,
        .size = alloc_limit + 64,
        .page_size = alloc_limit + 64,
        .buffer_type = tt::tt_metal::BufferType::L1};

    EXPECT_ANY_THROW(auto buffer = tt::tt_metal::CreateBuffer(l1_config););
}

}  // namespace tt::tt_metal
