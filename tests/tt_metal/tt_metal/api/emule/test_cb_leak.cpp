// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Dirty_CB_*"

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

// A CB is "dirty" when the kernel exits with a cb_reserve_back that no
// cb_push_back ever followed (or a cb_wait_front with no following cb_pop_front):
// the producer claimed the handshake but never handed the data off, so on silicon
// the consumer's matching cb_wait_front hangs. Detection is a trailing-dangling
// flag (set by reserve/wait, cleared by ANY following push/pop), decoupled from
// the reserve/wait *window* counters the CB Boundary check uses — because on
// silicon cb_reserve_back is a non-cumulative free-space wait, so a net
// "reserved − pushed" count false-positives on lookahead producers that reserve
// more than they push but always push after their last reserve (see
// Dirty_CB_LookaheadReserve_NoViolation). The sanitizer must catch a truly
// un-followed reserve/wait while leaving the lookahead idiom alone.
TEST_F(MeshDeviceFixture, Dirty_CB_ReserveWithoutPush) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    // This test validates the Dirty CB check itself, so force it on regardless
    // of any environment-level opt-out (TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB) that a
    // regression run may have exported to skip the check elsewhere.
    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Reserve a page but never push it -> reserve without a matching push.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            // MISSING: cb_push_back(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*Dirty CB Detected: Core \\(0, 0\\) CB 0 was not flushed!.*");
}

// The consumer-side mirror: a kernel that waits on a page but never pops it
// leaves the CB un-flushed — the consumer claimed read access it never released,
// so the read pointer desyncs. The sanitizer must catch this too.
TEST_F(MeshDeviceFixture, Dirty_CB_WaitWithoutPop) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    // This test validates the Dirty CB check itself, so force it on regardless
    // of any environment-level opt-out (TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB) that a
    // regression run may have exported to skip the check elsewhere.
    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Push a page so the wait_front below has data and does not block, then wait
    // on it but never pop -> wait without a matching pop.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            cb_wait_front(0, 1);
            // MISSING: cb_pop_front(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*Dirty CB Detected: Core \\(0, 0\\) CB 0 was not flushed!.*");
}

// Lookahead / double-buffer producer: reserves more pages than it pushes on
// every iteration (headroom for in-flight reads that the free-space count can't
// see), but pushes every block it produces, so a push always follows the last
// reserve. This is the canonical DRAM-sharded matmul in1 reader pattern. On
// silicon cb_reserve_back is a non-cumulative free-space wait, so the cumulative
// "reserved 4, pushed 3 -> net 1" is NOT a leak — nothing is stranded. This must
// NOT abort. It is the regression guard for the false positive that a net-count
// Dirty-CB check produced (it flagged ~num_blocks*block_tiles "unpushed" pages on
// a correct matmul reader); the trailing-dangling-flag detection fixes it.
TEST_F(MeshDeviceFixture, Dirty_CB_LookaheadReserve_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    // 4 pages deep so the 2-page lookahead reservations fit.
    CircularBufferConfig cb_config =
        CircularBufferConfig(4 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Reserve 2 (headroom), push 1, reserve 2 (headroom), push 1, push 1.
    // Cumulative reserved (4) > pushed (3), but every reserve is followed by a
    // push, so nothing dangles. The old net-count check aborted here.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 2);
            cb_push_back(0, 1);
            cb_reserve_back(0, 2);
            cb_push_back(0, 1);
            cb_push_back(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort — lookahead over-reservation is correct on silicon.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control: a kernel whose reserve/push and wait/pop all balance leaves
// the CB flushed, even though pages were left occupied (the producer pushed but
// nothing was popped). Leftover occupancy is NOT what this check is about — only
// unmatched reserve/wait. This must NOT abort. It also guards against regressing
// to the old (mistaken) "any occupied CB aborts" rule.
TEST_F(MeshDeviceFixture, Dirty_CB_Balanced_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    // This test validates the Dirty CB check itself, so force it on regardless
    // of any environment-level opt-out (TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB) that a
    // regression run may have exported to skip the check elsewhere.
    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Every reserve is matched by a push; no wait_front is issued so there is
    // nothing to pop. The CB is left occupied (1 page) but fully flushed.
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

    // Must NOT abort.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// The per-check opt-out: with the master switch on but
// TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB set, a reserve-without-push (which
// Dirty_CB_ReserveWithoutPush proves aborts by default) must run to completion —
// the runner's sweep_per_kernel_dirty_cbs returns early. This lets a regression
// run proceed past a known un-flushed-CB bug while every other sanitizer stays
// active.
TEST_F(MeshDeviceFixture, Dirty_CB_SkipEnv_Suppresses) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::setenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Same un-flushed-CB bug as Dirty_CB_ReserveWithoutPush.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            // MISSING: cb_push_back(0, 1);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort — the Dirty CB check is suppressed by the opt-out env var.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");
    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
