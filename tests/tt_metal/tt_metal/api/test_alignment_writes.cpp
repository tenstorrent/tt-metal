// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.L1_Alignment_SanityCheck:MeshDeviceFixture.DRAM_Alignment_SanityCheck_WH"

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
