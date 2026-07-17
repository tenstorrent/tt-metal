// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Host_Alignment_*"

// Coverage for the host-side L1/DRAM alignment sanitizer (host_sanitizers.hpp
// check_host_l1_alignment / check_host_dram_alignment). These checks fire only
// when alignment > 1 && address % alignment != 0, where `alignment` is the
// transfer's REAL requirement from Cluster::get_alignment_requirements(device,
// size). On emule (memory-backed I/O, no DMA engine) that requirement resolves
// to 1, so the check is a deliberate no-op: the host->device poke path accepts
// any byte address (it is what WriteToBuffer relies on to issue row-major page
// remainders at unaligned offsets).
//
// A death test is therefore impossible on the emule path — there is no way to
// make the check fire. What we CAN guard is the no-op contract: an intentionally
// UNALIGNED host poke must complete without aborting. This is a regression guard
// against re-hardcoding a fixed alignment (the historical bug was a hardcoded
// %32 / %4 that false-positived legitimate unaligned writes). If someone
// reintroduces a fixed alignment > 1, these round-trips SIGABRT and fail.

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

// Unaligned host->L1 write/read at an odd (1-byte-aligned) address inside an
// allocated L1 buffer must NOT abort on emule, and must round-trip correctly.
TEST_F(MeshDeviceFixture, Host_Alignment_L1_Unaligned_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    // Allocate an L1 buffer so the poke targets a valid, non-reserved address.
    constexpr uint32_t buf_size = 256;
    auto buf = Buffer::create(device, buf_size, buf_size, BufferType::L1);

    // Deliberately unaligned: buffer base + 1 byte. On a DMA-backed build with a
    // real alignment > 1 this would be rejected; on emule it must be accepted.
    uint32_t unaligned_addr = static_cast<uint32_t>(buf->address()) + 1;
    std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF};

    // Must NOT abort.
    detail::WriteToDeviceL1(
        device, logical_core, unaligned_addr, ttsl::Span<const uint8_t>(payload.data(), payload.size()));

    std::vector<uint8_t> readback(payload.size(), 0);
    detail::ReadFromDeviceL1(
        device, logical_core, unaligned_addr, ttsl::Span<uint8_t>(readback.data(), readback.size()));

    EXPECT_EQ(readback, payload);

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// DRAM counterpart: an unaligned host->DRAM channel write/read must likewise be
// a no-op for the alignment check on emule and round-trip correctly.
TEST_F(MeshDeviceFixture, Host_Alignment_DRAM_Unaligned_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    // Allocate a DRAM buffer to obtain a valid, non-reserved DRAM address.
    constexpr uint32_t buf_size = 256;
    auto buf = Buffer::create(device, buf_size, buf_size, BufferType::DRAM);

    uint32_t unaligned_addr = static_cast<uint32_t>(buf->address()) + 1;
    std::vector<uint8_t> payload = {0x01, 0x23, 0x45, 0x67};

    // Channel 0 backs this DRAM allocation on the wormhole_N150 mock. Must NOT abort.
    detail::WriteToDeviceDRAMChannel(
        device, /*dram_channel=*/0, unaligned_addr, ttsl::Span<const uint8_t>(payload.data(), payload.size()));

    std::vector<uint8_t> readback(payload.size(), 0);
    detail::ReadFromDeviceDRAMChannel(
        device, /*dram_channel=*/0, unaligned_addr, ttsl::Span<uint8_t>(readback.data(), readback.size()));

    EXPECT_EQ(readback, payload);

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
