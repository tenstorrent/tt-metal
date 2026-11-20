// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "impl/dispatch/aux_region_allocator.hpp"
#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal {

namespace {
constexpr uint32_t kDeviceBase = 0x1000'0000;
}  // namespace

TEST(AuxRegionAllocatorTest, CPU_EmptyAllocatorReturnsNull) {
    AuxRegionAllocator allocator;
    auto [host_ptr, device_addr] = allocator.allocate(64);
    EXPECT_EQ(host_ptr, nullptr);
    EXPECT_EQ(device_addr, 0u);
    EXPECT_EQ(allocator.used(), 0u);
}

TEST(AuxRegionAllocatorTest, CPU_BumpsHostAndDeviceAddressesTogether) {
    std::vector<char> backing(1024, 0x5a);
    AuxRegionAllocator allocator(
        backing.data(), kDeviceBase, static_cast<uint32_t>(backing.size()), /*has_d2h_fallback_reserve=*/false);

    auto [host_0, dev_0] = allocator.allocate(100);  // rounded up to 128
    auto [host_1, dev_1] = allocator.allocate(4);    // rounded up to 64
    auto [host_2, dev_2] = allocator.allocate(64);

    EXPECT_EQ(host_0, backing.data());
    EXPECT_EQ(dev_0, kDeviceBase);
    EXPECT_EQ(host_1, backing.data() + 128);
    EXPECT_EQ(dev_1, kDeviceBase + 128);
    EXPECT_EQ(host_2, backing.data() + 192);
    EXPECT_EQ(dev_2, kDeviceBase + 192);
    EXPECT_EQ(allocator.used(), 256u);
    EXPECT_EQ(allocator.size(), 1024u);
}

TEST(AuxRegionAllocatorTest, CPU_ZeroesAllocatedBytesOnly) {
    std::vector<char> backing(256, 0x5a);
    AuxRegionAllocator allocator(
        backing.data(), kDeviceBase, static_cast<uint32_t>(backing.size()), /*has_d2h_fallback_reserve=*/false);

    allocator.allocate(1);  // zeroes the whole 64 B slot

    for (size_t i = 0; i < 64; i++) {
        EXPECT_EQ(backing[i], 0) << "byte " << i;
    }
    for (size_t i = 64; i < backing.size(); i++) {
        EXPECT_EQ(backing[i], 0x5a) << "byte " << i;
    }
}

TEST(AuxRegionAllocatorTest, CPU_FillsExactlyThenFails) {
    std::vector<char> backing(256);
    AuxRegionAllocator allocator(
        backing.data(), kDeviceBase, static_cast<uint32_t>(backing.size()), /*has_d2h_fallback_reserve=*/false);

    allocator.allocate(192);
    auto [host_ptr, device_addr] = allocator.allocate(64);
    EXPECT_EQ(host_ptr, backing.data() + 192);
    EXPECT_EQ(device_addr, kDeviceBase + 192);
    EXPECT_EQ(allocator.used(), 256u);

    EXPECT_THROW(allocator.allocate(1), std::runtime_error);
    EXPECT_EQ(allocator.used(), 256u);
}

TEST(AuxRegionAllocatorTest, CPU_MoveCarriesD2hFallbackReserve) {
    std::vector<char> backing(256);
    AuxRegionAllocator allocator(backing.data(), kDeviceBase, static_cast<uint32_t>(backing.size()), true);
    EXPECT_TRUE(allocator.has_d2h_fallback_reserve());
    EXPECT_FALSE(AuxRegionAllocator().has_d2h_fallback_reserve());

    AuxRegionAllocator moved(std::move(allocator));
    EXPECT_TRUE(moved.has_d2h_fallback_reserve());
    EXPECT_FALSE(allocator.has_d2h_fallback_reserve());  // NOLINT(bugprone-use-after-move)

    allocator = std::move(moved);
    EXPECT_TRUE(allocator.has_d2h_fallback_reserve());
    EXPECT_FALSE(moved.has_d2h_fallback_reserve());  // NOLINT(bugprone-use-after-move)
}

TEST(AuxRegionAllocatorTest, CPU_ReservedBytesPerCq) {
    constexpr uint32_t page = DispatchSettings::TRANSFER_PAGE_SIZE;
    constexpr uint32_t fallback = DispatchSettings::HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES;

    EXPECT_EQ(AuxRegionAllocator::reserved_bytes_per_cq(1, false), 2 * page);
    EXPECT_EQ(AuxRegionAllocator::reserved_bytes_per_cq(2, false), 2 * page);
    EXPECT_EQ(AuxRegionAllocator::reserved_bytes_per_cq(1, true), 2 * page + fallback);
    EXPECT_EQ(AuxRegionAllocator::reserved_bytes_per_cq(2, true), 2 * page + fallback / 2);
    // With 3 CQs the fallback share is rounded up to a whole page, so together they still cover the reserve.
    const uint32_t per_cq_3 = AuxRegionAllocator::reserved_bytes_per_cq(3, true);
    EXPECT_EQ(per_cq_3 % page, 0u);
    EXPECT_GE(3 * (per_cq_3 - 2 * page), fallback);
}

}  // namespace tt::tt_metal
