// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.CB_Reservation_*"

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

TEST_F(MeshDeviceFixture, CB_Reservation_Overflow_SanityCheck) {
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    Program program = CreateProgram();
    uint32_t cb_id = 0;
    uint32_t num_pages = 2;
    uint32_t page_size = 1024;
    CircularBufferConfig cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, tt::DataFormat::Float16_b}})
                                         .set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Attempt to reserve 3 pages on a CB that only has 2
            cb_reserve_back(0, 3);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*\\[ASAN ERROR\\] CB Reservation Overflow:.*");
}

// CB Reservation Overflow is the one check that is ALWAYS ON (not gated by
// TT_METAL_EMULE_ASAN) — gating it would let an over-reserve deadlock on the
// free-space wait instead of reporting a clear error. This test explicitly
// clears the master switch and confirms the over-reserve still aborts.
TEST_F(MeshDeviceFixture, CB_Reservation_Overflow_AlwaysOn) {
    ::unsetenv("TT_METAL_EMULE_ASAN");  // master switch OFF — check must still fire

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    Program program = CreateProgram();
    uint32_t cb_id = 0;
    uint32_t num_pages = 2;
    uint32_t page_size = 1024;
    CircularBufferConfig cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, tt::DataFormat::Float16_b}})
                                         .set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 3);  // 3 > 2 total pages
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*\\[ASAN ERROR\\] CB Reservation Overflow:.*");
}

// Boundary positive control: reserving EXACTLY the CB's capacity (n ==
// num_pages) is legal and must NOT abort. Guards the comparison from being `>=`
// instead of `>`. Reserve 2 then push 2 so the CB also exits flushed.
//
// ORDERING: this non-death control MUST come after all EXPECT_DEATH tests above.
// A non-death LaunchProgram spins up the emule fiber worker-thread pool in the
// parent process; a later EXPECT_DEATH then fork()s and the child inherits that
// pool's locked state with the worker threads gone, so the fiber scheduler makes
// no progress and the watchdog aborts (~124 s). Death tests fork() before the
// pool exists, so keeping every death test first lets each fork from a clean,
// single-threaded parent. (The regression runner also splits them into separate
// invocations; this ordering additionally keeps the bare `CB_Reservation_*` glob
// runnable.)
TEST_F(MeshDeviceFixture, CB_Reservation_ExactCapacity_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    Program program = CreateProgram();
    uint32_t cb_id = 0;
    uint32_t num_pages = 2;
    uint32_t page_size = 1024;
    CircularBufferConfig cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, tt::DataFormat::Float16_b}})
                                         .set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Reserve exactly the full capacity (2 of 2) — legal — then flush.
            cb_reserve_back(0, 2);
            cb_push_back(0, 2);
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

}  // namespace tt::tt_metal
