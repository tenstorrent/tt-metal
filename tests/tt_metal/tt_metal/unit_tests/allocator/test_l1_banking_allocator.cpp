// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "basic_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersAllocatedTopDown) {
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0);

    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    size_t total_size_bytes = 0;

    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device->id());
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    const uint32_t interleaved_l1_bank_size = tt::get_l1_bank_size(device->id(), device->num_hw_cqs(), dispatch_core_type);
    uint64_t alloc_limit = interleaved_l1_bank_size - STORAGE_ONLY_UNRESERVED_BASE;

    std::vector<std::unique_ptr<Buffer>> buffers;
    int alloc_size_idx = 0;
    uint32_t total_buffer_size = 0;
    while (total_size_bytes < alloc_limit) {
        uint32_t buffer_size = alloc_sizes.at(alloc_size_idx);
        alloc_size_idx = (alloc_size_idx + 1) % alloc_sizes.size();
        if (total_buffer_size + buffer_size >= alloc_limit) {
            break;
        }
        std::unique_ptr<tt::tt_metal::Buffer> buffer = std::make_unique<tt::tt_metal::Buffer>(device, buffer_size, buffer_size, tt::tt_metal::BufferType::L1);
        buffers.emplace_back(std::move(buffer));
        total_buffer_size += buffer_size;
        EXPECT_EQ(buffers.back()->address(), device->l1_size_per_core() - total_buffer_size);
    }
    buffers.clear();

    tt::tt_metal::CloseDevice(device);
}

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0);

    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device->id());
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    const uint32_t interleaved_l1_bank_size = tt::get_l1_bank_size(device->id(), device->num_hw_cqs(), dispatch_core_type);
    uint64_t alloc_limit = interleaved_l1_bank_size - STORAGE_ONLY_UNRESERVED_BASE;

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
