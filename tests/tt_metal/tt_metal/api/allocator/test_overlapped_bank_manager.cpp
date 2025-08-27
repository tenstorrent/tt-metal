// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace unit_tests::test_overlapped_bank_manager {}  // namespace unit_tests::test_overlapped_bank_manager

namespace tt::tt_metal {

TEST_F(DeviceSingleCardBufferFixture, TestOverlappedBankManager) {
    // This test sets up a BankManager for DRAM and L1, mimicking Allocator initialization.

    // Get device and soc descriptor
    IDevice* device = this->device_;
    const metal_SocDescriptor& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());

    // DRAM setup
    {
        // Gather all DRAM bank descriptors (address offsets per channel, as in Allocator)
        const size_t num_channels = soc_desc.get_num_dram_views();
        std::vector<int64_t> dram_bank_descriptors(num_channels);
        for (size_t channel = 0; channel < num_channels; ++channel) {
            dram_bank_descriptors.at(channel) =
                static_cast<int64_t>(soc_desc.get_address_offset(static_cast<int>(channel)));
        }
        // Use per-channel DRAM size adjusted by allocator bases and alignment
        const uint64_t dram_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        const uint32_t dram_alignment = device->allocator()->get_alignment(BufferType::DRAM);
        const uint64_t dram_trace_region_size = device->allocator()->get_config().trace_region_size;
        const uint64_t dram_size =
            static_cast<uint64_t>(device->dram_size_per_channel()) - dram_unreserved_base - dram_trace_region_size;

        // Set up BankManager for DRAM
        BankManager dram_bank_manager(
            BufferType::DRAM,
            dram_bank_descriptors,
            dram_size,
            dram_alignment,
            /*alloc_offset=*/dram_unreserved_base,
            /*disable_interleaved=*/false,
            /*num_states=*/1);

        // Allocate a buffer and check
        uint32_t alloc_size = 64 * 1024;
        auto addr = dram_bank_manager.allocate_buffer(
            alloc_size,
            alloc_size,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            0);
        EXPECT_GT(addr, 0u);
        dram_bank_manager.deallocate_buffer(addr, 0);
    }
}

}  // namespace tt::tt_metal
