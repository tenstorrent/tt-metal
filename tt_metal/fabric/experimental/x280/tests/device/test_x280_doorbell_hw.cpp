// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The MSI catcher doorbell (README § 4, inbound paths; § 3, obligation 3): a NOC
// write of a non-zero u32 to X280_MSI_CATCHER_BASE at the tile's own (X, Y),
// from the host and from a Tensix kernel, wakes hart 0 from WFI through PLIC
// source 6. firmware-doorbell.bin enables the source itself, parks in WFI with a
// 10 ms timer as its only other wake, and disables both before it returns on
// STOP, so the case is reversible.

#include "x280_device_test_support.hpp"
#include "x280_probe_support.hpp"

#include "x280.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace x280;
using namespace std::chrono_literals;

constexpr uint64_t kRun = 0xD00BE11000000001ull;
constexpr uint32_t kHostValue = 0xD00B0001u;
constexpr uint32_t kTensixValueBase = 0xD00B0100u;
constexpr uint32_t kTensixRings = 20;
constexpr uint32_t kRecordLine = X280_PROBE_DB_ARMED;  // the record's 64-byte line
static_assert(kRecordLine % X280_PROBE_LINE_BYTES == 0);

class X280DoorbellHw : public x280::test::X280ProbeHwBase {
protected:
    // One doorbell's record: claimed source 6, one value popped, the value written.
    void expect_wake(uint64_t wakes, uint32_t value) {
        EXPECT_EQ(block_->read_u64(X280_PROBE_DB_WAKES), wakes);
        EXPECT_EQ(block_->read_u64(X280_PROBE_DB_SOURCE), X280_MSI_PLIC_SOURCE);
        EXPECT_EQ(block_->read_u64(X280_PROBE_DB_POPS), 1u);
        EXPECT_EQ(block_->read_u64(X280_PROBE_DB_VALUE), value);
    }
};

}  // namespace

TEST_F(X280DoorbellHw, HostAndTensixWritesWakeHartZeroFromWfi) {
    block_->reset(kRun);
    const SessionId id = enqueue(x280::test::test_image("X280_DOORBELL_FW_IMAGE", "firmware-doorbell.bin"), kRun);
    auto& session = runtime_->session(id);
    const auto running = session.wait_until_running(3s);
    ASSERT_EQ(running.outcome, WaitOutcome::Running) << describe(running);
    ASSERT_TRUE(block_->wait_u64(X280_PROBE_DB_ARMED, 1s, [](uint64_t armed) { return armed == kRun; }))
        << "hart 0 never armed the doorbell; result " << std::hex << running.hartstates[0].state.result;

    // Hart 0 left only source 6 enabled in its M context.
    EXPECT_EQ(block_->read_u64(X280_PROBE_DB_ENABLE), 1u << X280_MSI_PLIC_SOURCE);
    EXPECT_EQ(block_->read_u64(X280_PROBE_DB_WAKES), 0u);

    const TranslatedCoord at = runtime_->virtual_core_from_tile(tile_->id(), x280::test::NocProbe::kNoc);
    const int chip = tile_->id().chip_id;

    // Host writer.
    uint32_t value = kHostValue;
    x280::test::cluster().write_reg(&value, tt_cxy_pair(chip, at.x(), at.y()), X280_MSI_CATCHER_BASE);
    ASSERT_TRUE(block_->wait_u64(X280_PROBE_DB_WAKES, 1s, [](uint64_t wakes) { return wakes == 1; }))
        << "no wake from the host doorbell; timer wakes " << block_->read_u64(X280_PROBE_DB_TIMER_WAKES);
    expect_wake(1, kHostValue);

    // Tensix writer: the kernel rings, then reads the wake counter out of LIM
    // until it moves, timing both on its own wall clock.
    const uint32_t aiclk_mhz = static_cast<uint32_t>(x280::test::cluster().get_device_aiclk(chip));
    ASSERT_GT(aiclk_mhz, 0u);
    std::vector<double> latency_us;
    for (uint32_t ring = 0; ring < kTensixRings; ++ring) {
        const uint64_t before = block_->read_u64(X280_PROBE_DB_WAKES);
        const auto rung = probe_->doorbell(
            at,
            kTensixValueBase + ring,
            block_->address() + kRecordLine,
            X280_PROBE_DB_WAKES - kRecordLine,
            static_cast<uint32_t>(before),
            aiclk_mhz * 100000u);  // 100 ms
        ASSERT_TRUE(rung.woke) << "ring " << ring << ": counter stayed at " << before << " after " << rung.polls
                               << " polls";
        EXPECT_EQ(rung.counter, before + 1);
        expect_wake(before + 1, kTensixValueBase + ring);
        latency_us.push_back(static_cast<double>(rung.ticks) / aiclk_mhz);
    }

    std::sort(latency_us.begin(), latency_us.end());
    const uint64_t timer_wakes = block_->read_u64(X280_PROBE_DB_TIMER_WAKES);
    printf(
        "\nTensix doorbell to X280 wake count seen in LIM, %u rings at %u MHz AICLK: min %.2f us, median %.2f us, "
        "max %.2f us\ntimer wakes %llu, other wakes %llu\n",
        kTensixRings,
        aiclk_mhz,
        latency_us.front(),
        latency_us[latency_us.size() / 2],
        latency_us.back(),
        static_cast<unsigned long long>(timer_wakes),
        static_cast<unsigned long long>(block_->read_u64(X280_PROBE_DB_OTHER_WAKES)));
    // The timer alone would wake the hart up to 10 ms late; every ring well
    // under 1 ms is the doorbell ending WFI.
    EXPECT_LT(latency_us.back(), 1000.0);
    EXPECT_EQ(block_->read_u64(X280_PROBE_DB_OTHER_WAKES), 0u);

    const auto cleaned = runtime_->cleanup(id, 3s);
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
    EXPECT_TRUE(active_all_idle(cleaned.hartstates, kRun)) << describe(cleaned);
    EXPECT_EQ(cleaned.hartstates[0].state.result, 1u + kTensixRings) << describe(cleaned);
    EXPECT_TRUE(tile_->healthy());
}
