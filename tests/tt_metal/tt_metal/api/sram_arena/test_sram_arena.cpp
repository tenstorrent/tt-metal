// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

TEST(SRAMArenaTypeTest, CPU_CannotAdoptImplementation) {
    // Arena ownership cannot be transferred by adopting an implementation through the public API.
    EXPECT_FALSE((std::is_constructible_v<SRAMArena, SRAMArenaImpl&&>));
}

TEST_F(SRAMArenaAllocationTest, CPU_MovedFromMutableAccessFails) {
    // Mutable implementation access rejects a moved-from arena.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { (void)arena.impl(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(SRAMArenaAllocationTest, CPU_MovedFromConstAccessFails) {
    // Const implementation access rejects a moved-from arena.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { (void)std::as_const(arena).impl(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(SRAMArenaAllocationTest, CPU_AllocatedArenaHasValue) {
    // A new arena owns its reservation.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    EXPECT_FALSE(arena.is_valueless_after_move());
}

TEST_F(SRAMArenaAllocationTest, CPU_MoveConstructionTransfersValue) {
    // Move construction transfers the reservation from the source to the new owner.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto source = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto owner = std::move(source);
    EXPECT_TRUE(source.is_valueless_after_move());
    EXPECT_FALSE(owner.is_valueless_after_move());
}

TEST_F(SRAMArenaAllocationTest, CPU_MoveAssignmentTransfersValue) {
    // Move assignment transfers the reservation from the source to the target.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto source = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto target = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    target = std::move(source);
    EXPECT_TRUE(source.is_valueless_after_move());
    EXPECT_FALSE(target.is_valueless_after_move());
}

TEST_F(SRAMArenaAllocationTest, CPU_AssignmentRestoresMovedFromArena) {
    // A moved-from arena can receive a new reservation through assignment.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto owner = std::move(arena);
    arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    EXPECT_FALSE(arena.is_valueless_after_move());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
