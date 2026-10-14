// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Intentionally strands one hart in CEASE. Never include this source in the
// reversible device-test binary.

#include "x280/device/x280_runtime.hpp"

#include "x280_runtime_impl.hpp"

#include "x280.h"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <chrono>
#include <cstdlib>
#include <string>

namespace {

using namespace x280;
using namespace std::chrono_literals;

bool enabled() {
    const char* value = std::getenv("X280_ALLOW_ACTIVE_SESSION_FAILURE");
    return value != nullptr && std::string(value) == "1";
}

}  // namespace

TEST(X280ActiveFailure, TrapMakesTheTileUnhealthyAndRefusesReacquisition) {
    if (!enabled()) {
        GTEST_SKIP() << "set X280_ALLOW_ACTIVE_SESSION_FAILURE=1 to run the intentional CEASE test";
    }
    const char* image = std::getenv("X280_ACTIVE_FAULT_FW_IMAGE");
    if (image == nullptr || *image == '\0') {
        GTEST_SKIP() << "set X280_ACTIVE_FAULT_FW_IMAGE to firmware-active-session-fault.bin";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    auto runtime = std::move(*opened.runtime);
    const auto ids = runtime.tile_ids();
    ASSERT_FALSE(ids.empty());

    auto& tile = runtime.tile(ids.front());
    const auto initial = tile.inspect();
    ASSERT_TRUE(initial.observation_valid) << initial.invalid_reason;
    if (!initial.resident_idle_live) {
        GTEST_SKIP() << "intentional failure test requires an already-live resident idle";
    }

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
}
