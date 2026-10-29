// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <stdexcept>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

class SRAMArenaAllocationTest : public test_helpers::MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0> {};

TEST_F(SRAMArenaAllocationTest, CPU_LockstepRejectsSizeThatOverflows32Bits) {
    // Oversized lockstep reservations must fail even when their 32-bit total wraps to zero.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ASSERT_EQ(fixture_device.allocator()->get_num_banks(BufferType::L1), 72u);
    constexpr uint32_t per_bank_size = 512 * 1024 * 1024;
    EXPECT_THAT(
        [&] { SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{.per_bank_size = per_bank_size}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Out of Memory")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
