// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "basic_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

namespace unit_tests::test_l1_banking_allocator {

uint64_t get_alloc_limit(const tt::tt_metal::Device *device) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device->id());
    uint32_t l1_unreserved_base = device->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    auto storage_core_bank_size = tt::get_storage_core_bank_size(device->id(), device->num_hw_cqs(), dispatch_core_type);
    const uint32_t allocator_alignment = device->get_allocator_alignment();
    const uint32_t interleaved_l1_bank_size = storage_core_bank_size.has_value() ? storage_core_bank_size.value() : (soc_desc.worker_l1_size - l1_unreserved_base);
    uint32_t storage_core_unreserved_base = ((MEM_MAILBOX_BASE + allocator_alignment - 1) / allocator_alignment) * allocator_alignment;
    uint64_t alloc_limit = interleaved_l1_bank_size - storage_core_unreserved_base;
    return alloc_limit;
}

}   // namespace unit_tests::test_l1_banking_allocator

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersAllocatedTopDown) {
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0, 1, 0);

    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    size_t total_size_bytes = 0;

    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(device);

    std::vector<std::shared_ptr<Buffer>> buffers;
    int alloc_size_idx = 0;
    uint32_t total_buffer_size = 0;
    while (total_size_bytes < alloc_limit) {
        uint32_t buffer_size = alloc_sizes.at(alloc_size_idx);
        alloc_size_idx = (alloc_size_idx + 1) % alloc_sizes.size();
        if (total_buffer_size + buffer_size >= alloc_limit) {
            break;
        }
        auto buffer = tt::tt_metal::Buffer::create(device, buffer_size, buffer_size, tt::tt_metal::BufferType::L1);
        buffers.emplace_back(std::move(buffer));
        total_buffer_size += buffer_size;
        EXPECT_EQ(buffers.back()->address(), device->l1_size_per_core() - total_buffer_size);
    }
    buffers.clear();

    tt::tt_metal::CloseDevice(device);
}

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0, 1, 0);
    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(device);

    tt::tt_metal::InterleavedBufferConfig l1_config{
                    .device=device,
                    .size = alloc_limit + 64,
                    .page_size = alloc_limit + 64,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };

    EXPECT_ANY_THROW(
        auto buffer = tt::tt_metal::CreateBuffer(l1_config);
    );

    tt::tt_metal::CloseDevice(device);
}
