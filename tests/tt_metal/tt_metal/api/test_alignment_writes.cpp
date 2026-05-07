// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, IllegalWriteOutOfBounds) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto* device = this->devices_.at(id)->get_devices()[0];

        // Intentionally try to write to an address that exceeds the L1 limit
        // Query the actual L1 size and write beyond it to ensure this is truly illegal
        uint32_t l1_size = this->devices_.at(id)->l1_size_per_core();
        uint32_t illegal_addr = l1_size + 0x100000;  // L1 size + 1MB beyond
        std::vector<uint32_t> data = {1, 2, 3, 4};
        CoreCoord logical_core = {0, 0};

        detail::WriteToDeviceL1(device, logical_core, illegal_addr, data);
    }
}

TEST_F(MeshDeviceFixture, L1_Alignment_SanityCheck) {
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    
    // 0x1001 is NOT 4-byte aligned. 
    // This should trigger your new abort() in emulated_program_runner.cpp
    uint32_t misaligned_addr = 0x10001; 
    std::vector<uint32_t> data = {1};

    EXPECT_DEATH(
        detail::WriteToDeviceL1(device, logical_core, misaligned_addr, data),
        ".*L1 Alignment.*"
    );
}

TEST_F(MeshDeviceFixture, DRAM_Alignment_SanityCheck_WH) {
    auto* device = this->devices_.at(0)->get_devices()[0];
    
    // 0x10 is 16-aligned, but NOT 32-aligned (required for Wormhole DRAM)
    uint32_t misaligned_dram = 0x200000 + 1;
    std::vector<uint32_t> data = {1};

    // We use a low-level UMD call to trigger the resolver
    int dram_channel = 0;
    EXPECT_DEATH(
        detail::WriteToDeviceDRAMChannel(device, dram_channel, misaligned_dram, data),
        ".*DRAM Alignment.*"
    );
}

}  // namespace tt::tt_metal
