// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.CB_Boundary_*"

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
        detail::LaunchProgram(device, program), ".*CB Boundary Violation: Attempted to access CB 0.*reserved.*");
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
        detail::LaunchProgram(device, program), ".*CB Boundary Violation: Attempted to access CB 0.*Read window.*");
}

// Wraparound negative control. Identical setup (3-page CB, write_idx=2, reserve
// 2 → wrapped window {page 2, page 0}), but the noc_async_read destination is
// page 1 — the one page the reservation does NOT cover. Page 1's modular
// distance from write_idx(=2) is (1+3-2)%3 = 2, which is not < the 2 reserved
// pages, so it is outside the (wrapped) write window and the boundary check must
// abort. This proves the check stays ACTIVE through a wrap and rejects
// out-of-window pages, rather than passing everything (the additive-counter
// dormancy failure mode).
//
// ORDERING: kept with the other death tests, before every non-death control
// below. A prior non-death LaunchProgram leaves the emule fiber worker pool alive
// in the parent; a later EXPECT_DEATH fork()s and the child inherits that pool's
// locked state without its threads, hanging until the watchdog aborts (~124 s).
// Death-first keeps each fork clean. (See the note in the CB_Reservation test.)
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

// Positive control: accessing a CB page with NO active reservation/wait window
// (reserved==0 && waited==0) is raw get_write_ptr/get_read_ptr addressing — the
// pattern used by globally-allocated/sharded CBs and single-buffered scratch.
// The boundary check is only meaningful relative to an active window, so this
// must NOT abort. This is the exact shape (reserved==0, waited==0) that the old
// check false-positived across the TT-NN sweeps (expand/reshape/roll/to_memory_config/…).
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

// Produced-region reuse control: a write back into an already-produced page
// [read_idx, write_idx), outside the active reserve window, must NOT abort (the
// conv activation-reuse pattern). See SANITIZER_CHECKS.md §7.
TEST_F(MeshDeviceFixture, CB_Boundary_ProducedRegionReuse_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 4-page CB so the produced region and the fresh reservation are distinct.
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    CircularBufferConfig cb_config =
        CircularBufferConfig(4 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Produce 3 pages: write_idx advances to 3, read_idx stays 0
            // (producer never pops), so [0,3) is the produced region.
            cb_reserve_back(0, 3);
            cb_push_back(0, 3);
            // Fresh reservation: window covers only page 3.
            cb_reserve_back(0, 1);
            uint32_t write_addr = get_write_ptr(0);    // page 3 = cb.base + 3*1024
            uint64_t src_noc = get_noc_addr(write_addr);
            // Reuse: write back into page 1 — a PRODUCED page, behind write_idx and
            // outside the reserved window. Must be accepted.
            noc_async_read(src_noc, write_addr - 2048, 1024);
            noc_async_read_barrier();
            cb_push_back(0, 1);                         // flush: reserved -> 0
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Reuse of produced data is legal and the kernel exits flushed → no abort.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Globally-allocated CB exemption positive control. A CB backed by an explicit
// L1 buffer (set_globally_allocated_address) is addressed across its whole
// backing via computed offsets and only reserves nominally — so the boundary
// sub-check is skipped for it (asan_l1_checks.h: `!cb.globally_allocated`). Here
// the kernel reserves 1 page but accesses page 1 (outside the 1-page window);
// with the exemption in force this must NOT abort. If the `!cb.globally_allocated`
// guard regresses, every real sharded/matmul CB (cb_in0_sharded, DRAM-sharded
// readers) would false-positive — this pins that guard.
TEST_F(MeshDeviceFixture, CB_Boundary_GloballyAllocated_Exempt_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t num_pages = 2;

    // Backing L1 buffer makes the CB globally allocated (sharded-style addressing).
    // Single-bank (one buffer page spanning the whole CB) so its bank size equals
    // the CB total_size — a globally-allocated CB must fit inside its backing bank.
    auto backing = Buffer::create(device, num_pages * page_size, num_pages * page_size, BufferType::L1);
    CircularBufferConfig cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, tt::DataFormat::Float16_b}})
                                         .set_page_size(cb_id, page_size)
                                         .set_globally_allocated_address(*backing);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);                     // active window covers page 0 only
            uint32_t cb_write_addr = get_write_ptr(0);
            uint64_t src_noc = get_noc_addr(cb_write_addr);
            // Destination is page 1 — OUTSIDE the 1-page window, but the CB is
            // globally allocated, so the boundary check is exempt and must pass.
            noc_async_read(src_noc, cb_write_addr + 1024, 1024);
            noc_async_read_barrier();
            cb_push_back(0, 1);                         // clear the dangling reserve
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort — globally-allocated CBs are exempt from the boundary window check.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
