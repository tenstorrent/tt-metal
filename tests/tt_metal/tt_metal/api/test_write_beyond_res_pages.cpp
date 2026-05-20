// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.CB_Boundary_Violation_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// Sanity check for the CB-boundary sanitizer. The kernel reserves only 1 page
// on a 2-page CB, then issues a noc_async_read whose destination is page 1 of
// the same CB — i.e. one page beyond the reservation. The destination address
// flows through __emule_local_l1_to_ptr, where the sanitizer recognizes the
// address as belonging to a CB region but landing outside the currently-
// reserved sub-range, and aborts before the memcpy.
TEST_F(MeshDeviceFixture, CB_Boundary_Violation_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 2-page CB on (0,0); 1024-byte pages.
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Kernel: reserve 1 page, but target page 1 (offset = page_size) as the
    // noc_async_read destination. We use the CB's own write_ptr as the NoC
    // source — it doesn't matter what we read, because the destination check
    // aborts before the memcpy ever runs.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            uint32_t cb_write_addr = get_write_ptr(0);
            uint64_t src_noc = get_noc_addr(cb_write_addr);
            // Destination is one full page past the reserved region.
            noc_async_read(src_noc, cb_write_addr + 1024, 1024);
            noc_async_read_barrier();
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*CB Boundary Violation: Attempted to access CB 0.*reserved.*");
}

// Read-side counterpart. The kernel pushes one page (so the consumer side has
// occupied>0), then cb_wait_fronts only 1 page on a 2-page CB. The waited
// window covers page 0 (at read_idx=0). The kernel then issues a
// noc_async_write whose SOURCE is page 1 — a page the consumer has not waited
// on, which on silicon the producer may still be filling. The source address
// flows through __emule_local_l1_to_ptr; the sanitizer sees the access is
// outside both windows (write reservation is empty after the push) and aborts.
TEST_F(MeshDeviceFixture, CB_Boundary_Violation_Read_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Stage: push one page so write_idx advances to 1, occupied=1.
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            // Wait for 1 page — read window covers page 0 only.
            cb_wait_front(0, 1);
            uint32_t read_addr = get_read_ptr(0);
            // noc_async_write SOURCE is page 1 (unwaited).
            uint64_t dst_noc = get_noc_addr(read_addr);
            noc_async_write(read_addr + 1024, dst_noc, 1024);
            noc_async_write_barrier();
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*CB Boundary Violation: Attempted to access CB 0.*Read window.*");
}

// Wraparound sanity check. The kernel cycles the CB once (push 1, pop 1)
// so write_idx=1 and read_idx=1 on a 2-page CB. Then it reserves 2 pages —
// the reservation spans pages [1, 0) modular = {page 1, page 0}. A
// noc_async_read whose destination is either page 0 or page 1 should be
// inside the wrapped window and pass. We exercise the wrapped page (page 0,
// which sits BELOW the current write_ptr in raw address space) to make sure
// the modular page-distance math accepts it. No abort expected.
TEST_F(MeshDeviceFixture, CB_Boundary_Wraparound_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Cycle once so write_idx=1, read_idx=1 on a 2-page CB.
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            cb_wait_front(0, 1);
            cb_pop_front(0, 1);
            // Reserve 2 pages — window wraps: page 1 then page 0.
            cb_reserve_back(0, 2);
            uint32_t write_addr = get_write_ptr(0);
            uint64_t src_noc = get_noc_addr(write_addr);
            // Write into the wrapped slot (page 0). write_addr currently
            // points to page 1 (cb.base + 1024); page 0 is at write_addr - 1024.
            noc_async_read(src_noc, write_addr - 1024, 1024);
            noc_async_read_barrier();
            cb_push_back(0, 2);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // The kernel leaves CB 0 with occupied=2 at exit, so the dirty-CB
    // sanitizer will fire AFTER the (legitimate) wraparound write succeeds.
    // That confirms the boundary check let the wrap-around access through.
    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*Dirty CB Detected.*");
}

}  // namespace tt::tt_metal
