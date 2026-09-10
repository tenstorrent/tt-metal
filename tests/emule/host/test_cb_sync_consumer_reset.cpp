// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "tt_emule/cb_sync_state.hpp"
#include "tt_emule/device.hpp"

namespace {

constexpr uint32_t kCbId = 7;
constexpr uint32_t kPageSize = 1024;
constexpr uint32_t kNumPages = 4;

void register_two_consumers(tt_emule::CBSyncState& cb) {
    static int first_consumer;
    static int second_consumer;
    tt_emule::cb_register_consumer(cb, &first_consumer);
    tt_emule::cb_register_consumer(cb, &second_consumer);
    ASSERT_TRUE(cb.multi_consumer.load(std::memory_order_acquire));
}

TEST(CBSyncConsumerReset, InitClearsConsumerIdentity) {
    std::array<uint8_t, kPageSize * kNumPages> l1{};
    tt_emule::Core core({0, 0}, l1.data(), l1.size());
    auto& cb = core.cb_sync_array()[kCbId];

    register_two_consumers(cb);
    core.init_cb_sync(kCbId, l1.data(), kPageSize, kNumPages);

    EXPECT_EQ(cb.consumer.load(std::memory_order_acquire), nullptr);
    EXPECT_FALSE(cb.multi_consumer.load(std::memory_order_acquire));
}

TEST(CBSyncConsumerReset, ResetClearsConsumerIdentityBeforeNextProgram) {
    std::array<uint8_t, kPageSize * kNumPages> l1{};
    tt_emule::Core core({0, 0}, l1.data(), l1.size());
    auto& cb = core.cb_sync_array()[kCbId];

    register_two_consumers(cb);
    core.reset_cb_sync();

    EXPECT_EQ(cb.consumer.load(std::memory_order_acquire), nullptr);
    EXPECT_FALSE(cb.multi_consumer.load(std::memory_order_acquire));

    core.init_cb_sync(kCbId, l1.data(), kPageSize, kNumPages);
    int next_program_consumer;
    tt_emule::cb_register_consumer(cb, &next_program_consumer);

    EXPECT_EQ(cb.consumer.load(std::memory_order_acquire), &next_program_consumer);
    EXPECT_FALSE(cb.multi_consumer.load(std::memory_order_acquire));
}

}  // namespace
