// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One hart traps on purpose. The trap path parks it in WFI, the bounded cleanup
// latches the tile unhealthy, and X280Runtime::revive takes all four harts back
// to resident idle, so the binary leaves the tile serving. It stays a separate
// binary: a revive that fails leaves the tile for an ASIC reset.

#include "x280/device/x280_runtime.hpp"

#include "x280_runtime_impl.hpp"

#include "x280.h"
#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <chrono>
#include <iostream>
#include <string>

namespace {

using namespace x280;
using namespace std::chrono_literals;

// The sentinel runs and returns: the tile serves again.
void expect_sentinel(X280Runtime& runtime, X280Tile& tile, uint64_t value) {
    ActiveRunRequest request;
    request.image_path = x280::test::test_image("X280_ACTIVE_FW_IMAGE", "firmware.bin");
    request.run_value = value;
    request.cleanup_timeout = 3s;
    const auto enqueued = runtime.enqueue(tile, request);
    ASSERT_EQ(enqueued.outcome, EnqueueOutcome::Enqueued) << describe(enqueued);
    const auto ready = runtime.session(enqueued.session).wait_until_result_ready(3s);
    ASSERT_EQ(ready.outcome, WaitOutcome::ResultReady) << describe(ready);
    EXPECT_TRUE(active_result_matches(ready.hartstates, value)) << describe(ready);
    const auto cleaned = runtime.cleanup(enqueued.session, 3s);
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
}

}  // namespace

TEST(X280ActiveFailure, TrapMakesTheTileUnhealthyAndRefusesReacquisition) {
    const std::string image = x280::test::test_image("X280_ACTIVE_FAULT_FW_IMAGE", "firmware-active-session-fault.bin");

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!opened.runtime && (opened.requires_preparation() || opened.requires_reset())) {
        GTEST_SKIP() << x280::test::unprepared(opened);
    }
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    auto runtime = std::move(*opened.runtime);
    const auto ids = runtime.tile_ids();
    ASSERT_FALSE(ids.empty());

    x280::test::boot_resident_idle(runtime, ids.front());
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    auto& tile = runtime.tile(ids.front());

    ActiveRunRequest request;
    request.image_path = image;
    request.run_value = 0xFA01700000000001ull;
    request.cleanup_timeout = 100ms;
    const auto enqueued = runtime.enqueue(tile, request);
    ASSERT_TRUE(enqueued.session) << describe(enqueued);
    const SessionId id = enqueued.session;
    auto& session = runtime.session(id);

    const auto running = session.wait_until_running(3s);
    EXPECT_TRUE(running.outcome == WaitOutcome::Running || running.outcome == WaitOutcome::TimedOut)
        << describe(running);

    const auto result = session.wait_until_result_ready(100ms);
    EXPECT_EQ(result.outcome, WaitOutcome::TimedOut) << describe(result);
    const auto& fault = result.hartstates[2];
    EXPECT_EQ(fault.state.phase, X280_HART_PHASE_RUNNING_ACTIVE_FW);
    EXPECT_EQ(fault.trap.mcause, 3u);
    EXPECT_NE(fault.trap.mepc, 0u);
    EXPECT_EQ(fault.trap.trap_valid, 1u);
    for (uint32_t hart : {0u, 1u, 3u}) {
        EXPECT_TRUE(hart_result_ready(result.hartstates[hart], request.run_value));
    }
    EXPECT_EQ(result.timeout, 100ms);

    const auto cleaned = runtime.cleanup(id, 100ms);
    EXPECT_EQ(cleaned.outcome, CleanupOutcome::StillRunningActiveFw) << describe(cleaned);
    EXPECT_EQ(cleaned.hartstates[2].trap.mcause, 3u);

    // A failed restoration is terminal, so the session carrying the evidence is
    // reaped and the tile keeps the snapshot.
    EXPECT_FALSE(runtime.session_live(id));
    EXPECT_TRUE(runtime.sessions().empty());
    EXPECT_ANY_THROW((void)runtime.session(id));

    EXPECT_FALSE(tile.healthy());
    const auto failure = tile.last_failure();
    ASSERT_TRUE(failure.has_value());
    const auto& stranded = (*failure)[2];
    EXPECT_EQ(stranded.hart_id, 2u);
    EXPECT_EQ(stranded.trap.mcause, 3u);
    EXPECT_EQ(stranded.trap.trap_valid, 1u);
    EXPECT_EQ(stranded.trap.run_value, request.run_value);

    const auto refused = runtime.enqueue(tile, request);
    EXPECT_EQ(refused.outcome, EnqueueOutcome::Unhealthy) << describe(refused);
    EXPECT_FALSE(refused.session);

    // The latch is not the end: an RNMI takes all four harts back to resident
    // idle, including hart 2, which the trap path stopped.
    const auto revived = runtime.revive(tile);
    std::cout << "[ revive  ] trapped hart 2: " << describe(revived) << "\n";
    ASSERT_EQ(revived.outcome, ReviveOutcome::Revived) << describe(revived);
    EXPECT_TRUE(tile.healthy());
    const auto& hart2 = revived.harts[2];
    EXPECT_TRUE(hart2.acknowledged);
    EXPECT_GE(hart2.revive_epc, X280_ACTIVE_FW_LOAD_ADDR);
    EXPECT_LT(hart2.revive_epc, X280_ACTIVE_FW_REGION_END);
    EXPECT_EQ(hart2.before.trap.trap_valid, 1u);
    EXPECT_EQ(hart2.after.trap.trap_valid, 0u);
    // The other three were waiting for the RETURN_ACK the failed cleanup never sent.
    for (uint32_t hart : {0u, 1u, 3u}) {
        EXPECT_EQ(revived.harts[hart].before.state.phase, X280_HART_PHASE_RETURNED_TO_IDLE) << hart;
    }
    expect_sentinel(runtime, tile, 0xFA01700000000002ull);
}

TEST(X280ActiveFailure, ATrapIsRevivedByCleanupWhenTheRequestOptsIn) {
    const std::string image = x280::test::test_image("X280_ACTIVE_FAULT_FW_IMAGE", "firmware-active-session-fault.bin");

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!opened.runtime && (opened.requires_preparation() || opened.requires_reset())) {
        GTEST_SKIP() << x280::test::unprepared(opened);
    }
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    auto runtime = std::move(*opened.runtime);
    const auto ids = runtime.tile_ids();
    ASSERT_FALSE(ids.empty());
    x280::test::boot_resident_idle(runtime, ids.front());
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    auto& tile = runtime.tile(ids.front());

    ActiveRunRequest request;
    request.image_path = image;
    request.run_value = 0xFA01700000000011ull;
    request.cleanup_timeout = 100ms;
    request.revive_on_cleanup_timeout = true;
    const auto enqueued = runtime.enqueue(tile, request);
    ASSERT_TRUE(enqueued.session) << describe(enqueued);
    (void)runtime.session(enqueued.session).wait_until_result_ready(100ms);

    const auto cleaned = runtime.cleanup(enqueued.session, 100ms);
    std::cout << "[ revive  ] cleanup escalation: " << describe(cleaned) << "\n";
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::Revived) << describe(cleaned);
    ASSERT_TRUE(cleaned.revive.has_value());
    EXPECT_EQ(cleaned.revive->harts[2].before.trap.mcause, 3u);
    EXPECT_TRUE(tile.healthy());
    EXPECT_FALSE(runtime.session_live(enqueued.session));
    expect_sentinel(runtime, tile, 0xFA01700000000012ull);
}
