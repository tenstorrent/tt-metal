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
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

// How the case writes its data: an L1 write with the default Strict ordering, an L1 write that asks for Relaxed
// ordering, or a DRAM channel write, which is Relaxed.
enum class HostWriteKind { L1Strict, L1Relaxed, Dram };

struct HostWriteCase {
    std::string name;
    HostWriteKind kind;
    uint32_t size_bytes;
    uint32_t address_offset = 0;
};

namespace {

// Cluster::write_core sends Relaxed writes of 64 KiB or more through PCIe DMA. Strict writes stay on MMIO on Blackhole
// and use DMA from 32 B on Wormhole, and anything whose size or address is not a multiple of 4 stays on MMIO. These
// cases land on both sides of each of those rules.
const std::vector<HostWriteCase> kHostWriteCases = {
    {"L1Strict1KiB", HostWriteKind::L1Strict, 1024},
    {"L1Strict256KiB", HostWriteKind::L1Strict, 256 * 1024},
    {"L1StrictUnalignedSize", HostWriteKind::L1Strict, 2050},
    {"L1RelaxedMmio32KiB", HostWriteKind::L1Relaxed, 32 * 1024},
    {"L1RelaxedDma64KiB", HostWriteKind::L1Relaxed, 64 * 1024},
    {"L1RelaxedMmio64KiBAtUnalignedAddress", HostWriteKind::L1Relaxed, 64 * 1024, 2},
    {"DramMmio32KiB", HostWriteKind::Dram, 32 * 1024},
    {"DramDma64KiB", HostWriteKind::Dram, 64 * 1024},
    {"DramDma4MiB", HostWriteKind::Dram, 4 * 1024 * 1024},
    {"DramMmio64KiBAtUnalignedAddress", HostWriteKind::Dram, 64 * 1024, 2},
};

}  // namespace

class HostWriteDmaTest : public UnitMeshFixture, public ::testing::WithParamInterface<HostWriteCase> {};

// A host write reads back intact whichever transport it is routed through.
TEST_P(HostWriteDmaTest, WriteReadsBack) {
    auto& mesh_device = *this->devices_[0];
    const HostWriteCase& test_case = GetParam();
    const HalMemType mem_type = test_case.kind == HostWriteKind::Dram ? HalMemType::DRAM : HalMemType::L1;
    const uint32_t address = mesh_device.allocator()->get_base_allocator_addr(mem_type) + test_case.address_offset;

    std::vector<uint8_t> pattern(test_case.size_bytes);
    for (uint32_t i = 0; i < test_case.size_bytes; ++i) {
        pattern[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    std::vector<uint8_t> readback(test_case.size_bytes, 0);

    switch (test_case.kind) {
        case HostWriteKind::L1Strict:
            slow_dispatch::WriteToL1(mesh_device, CoreCoord{0, 0}, address, std::span<const uint8_t>(pattern));
            break;
        case HostWriteKind::L1Relaxed: {
            const ChipId chip_id = mesh_device.get_device_ids()[0];
            const CoreCoord worker_core = mesh_device.worker_core_from_logical_core(CoreCoord{0, 0});
            MetalContext::instance().get_cluster().write_core(
                pattern.data(),
                pattern.size(),
                tt_cxy_pair(chip_id, worker_core),
                address,
                tt::umd::IoOrdering::Relaxed);
            break;
        }
        case HostWriteKind::Dram:
            slow_dispatch::WriteToDRAMChannel(mesh_device, 0, address, std::span<const uint8_t>(pattern));
            break;
    }

    if (mem_type == HalMemType::L1) {
        slow_dispatch::ReadFromL1(mesh_device, CoreCoord{0, 0}, address, std::span<uint8_t>(readback));
    } else {
        slow_dispatch::ReadFromDRAMChannel(mesh_device, 0, address, std::span<uint8_t>(readback));
    }

    EXPECT_EQ(readback, pattern);
}

INSTANTIATE_TEST_SUITE_P(
    HostWrites,
    HostWriteDmaTest,
    ::testing::ValuesIn(kHostWriteCases),
    [](const ::testing::TestParamInfo<HostWriteCase>& info) { return info.param.name; });

}  // namespace tt::tt_metal
