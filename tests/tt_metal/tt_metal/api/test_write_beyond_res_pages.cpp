// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.CB_Boundary_*"

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

// Wraparound positive control. On a 3-page CB we cycle the write/read pointers
// forward to write_idx=2 (occupied=0), then reserve 2 pages. The reserved write
// window now WRAPS to a strict subset — page 2 (at write_idx) and page 0 (the
// wrapped continuation) — while page 1 is left OUTSIDE the window. A
// noc_async_read whose destination is the wrapped page 0 (which sits at a LOWER
// raw address than the write pointer) must be accepted by the modular
// page-distance math. The kernel then pushes the 2 reserved pages so it exits
// flushed (reserved==0, waited==0) — neither the boundary check nor the
// redefined Dirty-CB check may fire. No abort expected.
//
// (A 3-page CB, not 2, is deliberate: with a 2-page CB a 2-page reservation
// spans the whole buffer, so every page trivially passes and the test cannot
// tell "accepted because wrapped correctly" apart from "the check went dormant"
// — the additive-counter false-negative mode. With 3 pages the window is a
// strict subset, which the paired Violation test below relies on.)
TEST_F(MeshDeviceFixture, CB_Boundary_Wraparound_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(3 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Cycle twice so write_idx=2, read_idx=2 on a 3-page CB (occupied=0).
            for (uint32_t i = 0; i < 2; ++i) {
                cb_reserve_back(0, 1);
                cb_push_back(0, 1);
                cb_wait_front(0, 1);
                cb_pop_front(0, 1);
            }
            // Reserve 2 — window wraps: page 2 then page 0 (page 1 excluded).
            cb_reserve_back(0, 2);
            uint32_t write_addr = get_write_ptr(0);   // page 2 = cb.base + 2*1024
            uint64_t src_noc = get_noc_addr(write_addr);
            // Destination is the wrapped slot (page 0 = write_addr - 2*1024).
            noc_async_read(src_noc, write_addr - 2048, 1024);
            noc_async_read_barrier();
            cb_push_back(0, 2);                        // flush: reserved -> 0
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Wrapped access is legal and the kernel exits flushed → no sanitizer fires.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Wraparound negative control. Identical setup (3-page CB, write_idx=2, reserve
// 2 → wrapped window {page 2, page 0}), but the noc_async_read destination is
// page 1 — the one page the reservation does NOT cover. Page 1's modular
// distance from write_idx(=2) is (1+3-2)%3 = 2, which is not < the 2 reserved
// pages, so it is outside the (wrapped) write window and the boundary check must
// abort. This proves the check stays ACTIVE through a wrap and rejects
// out-of-window pages, rather than passing everything (the additive-counter
// dormancy failure mode).
TEST_F(MeshDeviceFixture, CB_Boundary_Wraparound_Violation_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(3 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Cycle twice so write_idx=2, read_idx=2 on a 3-page CB (occupied=0).
            for (uint32_t i = 0; i < 2; ++i) {
                cb_reserve_back(0, 1);
                cb_push_back(0, 1);
                cb_wait_front(0, 1);
                cb_pop_front(0, 1);
            }
            cb_reserve_back(0, 2);
            uint32_t write_addr = get_write_ptr(0);   // page 2 = cb.base + 2*1024
            uint64_t src_noc = get_noc_addr(write_addr);
            // Destination is page 1 — OUTSIDE the wrapped {page 2, page 0} window.
            noc_async_read(src_noc, write_addr - 1024, 1024);
            noc_async_read_barrier();
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*CB Boundary Violation: Attempted to access CB 0.*");
}

// Positive control: accessing a CB page with NO active reservation/wait window
// (reserved==0 && waited==0) is raw get_write_ptr/get_read_ptr addressing — the
// pattern used by globally-allocated/sharded CBs and single-buffered scratch.
// The boundary check is only meaningful relative to an active window, so this
// must NOT abort. This is the exact shape (reserved==0, waited==0) that the old
// check false-positived across the ttnn sweeps (expand/reshape/roll/to_memory_config/…).
TEST_F(MeshDeviceFixture, CB_Boundary_NoActiveWindow_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    // No cb_reserve_back / cb_wait_front: the in-bounds access to page 0 reaches
    // __emule_local_l1_to_ptr with reserved==0 && waited==0. No push, so the CB
    // ends empty (occupied==0) — nothing else can fire either.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_write_ptr(0);          // page 0, in-bounds
            uint64_t src = get_noc_addr(addr);
            noc_async_read(src, addr, 1024);           // dst goes through __emule_local_l1_to_ptr
            noc_async_read_barrier();
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort. If the window check regresses to firing without an active
    // reservation/wait, LaunchProgram SIGABRTs and this test fails.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
