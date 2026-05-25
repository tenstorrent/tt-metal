// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api
// --gtest_filter="MeshDeviceFixture.Dirty_CB_SanityCheck:MeshDeviceFixture.Dirty_CB_ProducerOnly_NoViolation"

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

// A kernel that pushes N pages onto a CB but only pops N-1 leaves the CB in a
// "dirty" state: occupied > 0 at program exit. On silicon the residual page
// offset survives in the CB pointers, so the next program launch immediately
// back-pressures (cb_reserve_back blocks forever). This test exercises that
// exact scenario and verifies the emulator's sanitizer catches it.
//
// Note this kernel pops at least once (total_popped > 0), i.e. the CB has a real
// consumer that under-drains — which is the genuine leak the sweep targets. A CB
// that is pushed but NEVER popped (no consumer) is a different, legitimate case;
// see Dirty_CB_ProducerOnly_NoViolation below.
TEST_F(MeshDeviceFixture, Dirty_CB_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 1. Create a CB with 2 pages (enough headroom to push twice without
    //    needing to wrap, since pop happens between pushes).
    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // 2. Kernel pushes 2 pages, pops 1 → leaves 1 page on the CB at exit.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // First push
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            // Drain it so the second push doesn't block on a full CB
            cb_wait_front(0, 1);
            cb_pop_front(0, 1);
            // Second push, NOT drained — CB ends with occupied=1.
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Verify the emulator catches the dirty state.
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Dirty CB Detected: Core \\(0, 0\\) CB 0 was not flushed!.*");
}

// Positive control: a CB that is pushed but NEVER popped is NOT a leak. It models
// a globally-allocated/sharded output CB, or a producer-only single-kernel program
// that DMAs its result straight out (e.g. nonzero) — there is no consumer, so the
// leftover page at exit is by design. With total_popped == 0 the Dirty-CB sweep
// must NOT abort. This guards against regressing to the old "any occupied CB
// aborts" rule, which false-positived every sharded/producer-only op across the
// ttnn sweeps.
TEST_F(MeshDeviceFixture, Dirty_CB_ProducerOnly_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Producer-only: push one page, never pop. occupied == 1 at exit but
    // total_popped == 0 (no consumer) -> not a dirty-CB leak.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort. If the sweep regresses to flagging any occupied CB,
    // LaunchProgram SIGABRTs and this test fails.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
