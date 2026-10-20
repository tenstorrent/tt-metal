// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

struct HostWriteCase {
    std::string name;
    HalMemType mem_type;
    uint32_t size_bytes;
};

// On Blackhole, Cluster::write_core sends Strict writes of 2 KiB or more and Relaxed writes of 64 KiB or more
// through PCIe DMA, and everything else, including sizes that are not a multiple of 4, through MMIO. L1 writes
// default to Strict and DRAM writes to Relaxed, so these sizes land on both sides of each threshold.
const std::vector<HostWriteCase> kHostWriteCases = {
    {"L1Mmio1KiB", HalMemType::L1, 1024},
    {"L1Dma2KiB", HalMemType::L1, 2048},
    {"L1Mmio2KiBPlus2", HalMemType::L1, 2050},
    {"L1Dma256KiB", HalMemType::L1, 256 * 1024},
    {"DramMmio32KiB", HalMemType::DRAM, 32 * 1024},
    {"DramDma64KiB", HalMemType::DRAM, 64 * 1024},
    {"DramDma4MiB", HalMemType::DRAM, 4 * 1024 * 1024},
};

}  // namespace

class BlackholeHostWriteDmaTest : public BlackholeSingleCardFixture,
                                  public ::testing::WithParamInterface<HostWriteCase> {};

// A host write reads back intact whichever transport it is routed through.
TEST_P(BlackholeHostWriteDmaTest, WriteReadsBack) {
    auto& mesh_device = *this->devices_[0];
    const HostWriteCase& test_case = GetParam();
    const uint32_t address = mesh_device.allocator()->get_base_allocator_addr(test_case.mem_type);

    std::vector<uint8_t> pattern(test_case.size_bytes);
    for (uint32_t i = 0; i < test_case.size_bytes; ++i) {
        pattern[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    std::vector<uint8_t> readback(test_case.size_bytes, 0);

    if (test_case.mem_type == HalMemType::L1) {
        slow_dispatch::WriteToL1(mesh_device, CoreCoord{0, 0}, address, std::span<const uint8_t>(pattern));
        slow_dispatch::ReadFromL1(mesh_device, CoreCoord{0, 0}, address, std::span<uint8_t>(readback));
    } else {
        slow_dispatch::WriteToDRAMChannel(mesh_device, 0, address, std::span<const uint8_t>(pattern));
        slow_dispatch::ReadFromDRAMChannel(mesh_device, 0, address, std::span<uint8_t>(readback));
    }

    EXPECT_EQ(readback, pattern);
}

INSTANTIATE_TEST_SUITE_P(
    HostWrites,
    BlackholeHostWriteDmaTest,
    ::testing::ValuesIn(kHostWriteCases),
    [](const ::testing::TestParamInfo<HostWriteCase>& info) { return info.param.name; });

}  // namespace tt::tt_metal
