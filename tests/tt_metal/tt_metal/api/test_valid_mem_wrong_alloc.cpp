// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Object_Intent_*"

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

// IMPORTANT (test-design constraint): the Object-Intent check exempts any live
// buffer whose L1 start address appears in the kernel's runtime args — that is
// the "I/O tensor this kernel was handed" rule (in-place ops / fused producers
// are allowed to write a buffer they were given). So a violation test must NOT
// pass the victim buffer's absolute address as a runtime arg, or it authorizes
// the very write it is trying to flag. These tests pass only the victim's BYTE
// OFFSET from the resolved buffer; the victim's address never enters the args.
TEST_F(MeshDeviceFixture, Object_Intent_Provenance_Violation_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 1. Allocate two distinct buffers sequentially in L1 memory.
    //
    // L1 allocates top-down by default (see Buffer::bottom_up_ in buffer.cpp),
    // so the first Buffer::create lands at the highest address and the second
    // lands immediately below it. Allocate Buffer B first so it occupies the
    // upper slot — Buffer A then sits directly under it with addr_a < addr_b.
    // That arrangement matches the narrative the kernel computes below
    // (overshoot Buffer A upward to stomp Buffer B).
    uint32_t buf_size = 1024; // 1 KB each
    auto buffer_b = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    auto buffer_a = Buffer::create(device, buf_size, buf_size, BufferType::L1);

    uint32_t addr_a = buffer_a->address();
    uint32_t addr_b = buffer_b->address();
    ASSERT_LT(addr_a, addr_b);
    ASSERT_EQ((addr_b - addr_a) % sizeof(uint32_t), 0u);

    // Prove the kernel's computed write target lands inside Buffer B while
    // being outside Buffer A.
    uint32_t target_addr = addr_a + ((addr_b - addr_a) / sizeof(uint32_t)) * sizeof(uint32_t);
    ASSERT_GE(target_addr, addr_b);
    ASSERT_LT(target_addr, addr_b + buf_size);
    ASSERT_FALSE(target_addr >= addr_a && target_addr < addr_a + buf_size);

    // 2. Add an inline kernel that overshoots Buffer A to stomp on Buffer B.
    // It is handed Buffer A's base and the BYTE GAP to Buffer B — never B's
    // absolute address (see the I/O-tensor exemption note above).
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a   = get_arg_val<uint32_t>(0);
            uint32_t gap_bytes = get_arg_val<uint32_t>(1);  // (addr_b - addr_a), NOT B's address

            // Establish pointer provenance linked explicitly to Buffer A's base address context
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);

            // Overshoot by the gap to land inside Buffer B.
            uint32_t elements_to_b = gap_bytes / sizeof(uint32_t);

            // This write lands on a perfectly valid, aligned, and allocated L1 memory address (Buffer B).
            // However, because the pointer tracking metadata was anchored to Object A,
            // crossing this structural boundary constitutes an illegal object intent mutation.
            ptr_a[elements_to_b] = 0x900DDEAD;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Pass Buffer A's base and the byte gap to B (NOT addr_b). Passing addr_b
    // would put B in io_arg_starts and exempt it from the snapshot, masking the
    // violation. The gap (addr_b - addr_a) matches no buffer's start address.
    SetRuntimeArgs(program, kernel, logical_core, {addr_a, addr_b - addr_a});

    // 3. The sanitizer should catch that the execution sequence breached pointer provenance bounds
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Object Intent Violation: Attempted to modify memory belonging to an adjacent object context.*"
    );
}

// Positive control: when the kernel resolves BOTH buffers via __emule_local_l1_to_ptr
// and writes only within their own bounds, both buffers end up in the per-core
// "resolved set". The post-launch comparison should skip them and the program
// should complete normally. If the sanitizer false-positives (e.g. compares
// against a stale snapshot, or fails to record a resolution), this test will
// crash and fail. It guards the well-behaved path from being broken by future
// changes to the resolved-set tracking or snapshot logic.
TEST_F(MeshDeviceFixture, Object_Intent_Provenance_NoViolation_Control) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t buf_size = 1024;
    auto buffer_b = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    auto buffer_a = Buffer::create(device, buf_size, buf_size, BufferType::L1);

    // Resolve both buffers, write only within bounds of each — the
    // "intended write set" covers every buffer whose bytes change.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a = get_arg_val<uint32_t>(0);
            uint32_t base_b = get_arg_val<uint32_t>(1);
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);
            volatile uint32_t* ptr_b = (volatile uint32_t*)__emule_local_l1_to_ptr(base_b);
            ptr_a[0] = 0xAAAAAAAA;
            ptr_b[0] = 0xBBBBBBBB;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    SetRuntimeArgs(program, kernel, logical_core, {buffer_a->address(), buffer_b->address()});

    // Must NOT abort. If the sanitizer is over-eager, LaunchProgram will SIGABRT
    // and the test harness will mark this as failed.
    detail::LaunchProgram(device, program);
    SUCCEED();
}

// Non-adjacent stomp: three buffers laid out in L1, and the kernel
// resolves only the lowest one (A) but overshoots two slots upward into the
// highest one (C), skipping the middle buffer (B) entirely. This proves the
// check fires on any modified-but-unresolved buffer, not just on the immediate
// neighbor of the resolved buffer — which is the entire point of the
// "provenance" framing (the violating address can be arbitrarily far from
// the intended object as long as it lands inside *some* allocated buffer).
TEST_F(MeshDeviceFixture, Object_Intent_Provenance_NonAdjacent_Violation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Top-down: first allocated lands highest. Order so addr_a < addr_b < addr_c.
    uint32_t buf_size = 1024;
    auto buffer_c = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    auto buffer_b = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    auto buffer_a = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    uint32_t addr_a = buffer_a->address();
    uint32_t addr_b = buffer_b->address();
    uint32_t addr_c = buffer_c->address();
    ASSERT_LT(addr_a, addr_b);
    ASSERT_LT(addr_b, addr_c);
    ASSERT_EQ((addr_c - addr_a) % sizeof(uint32_t), 0u);

    uint32_t target_addr = addr_a + ((addr_c - addr_a) / sizeof(uint32_t)) * sizeof(uint32_t);
    ASSERT_GE(target_addr, addr_c);
    ASSERT_LT(target_addr, addr_c + buf_size);
    ASSERT_FALSE(target_addr >= addr_a && target_addr < addr_a + buf_size);
    ASSERT_FALSE(target_addr >= addr_b && target_addr < addr_b + buf_size);

    // The kernel only resolves Buffer A and is handed the BYTE GAP to Buffer C
    // (never addr_c — passing it would exempt C as an I/O tensor). The gap is
    // used for arithmetic only; C never goes through __emule_local_l1_to_ptr, so
    // Buffer C never enters the resolved set. The store lands inside Buffer C
    // (valid L1, valid allocation, but not the kernel's intended object).
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a   = get_arg_val<uint32_t>(0);
            uint32_t gap_bytes = get_arg_val<uint32_t>(1);  // (addr_c - addr_a), NOT C's address
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);
            uint32_t elements_to_c = gap_bytes / sizeof(uint32_t);
            ptr_a[elements_to_c] = 0xDEADBEEF;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    SetRuntimeArgs(program, kernel, logical_core, {addr_a, addr_c - addr_a});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Object Intent Violation: Attempted to modify memory belonging to an adjacent object context.*"
    );
}

// Positive control for the I/O-tensor exemption. This is the mirror of the
// Violation test: the kernel again resolves only Buffer A and overshoots into
// Buffer B — but here Buffer B's ABSOLUTE address IS passed as a runtime arg.
// Per the exemption, a buffer whose start address appears in the kernel's
// runtime args is one the kernel was explicitly handed to operate on (in-place
// ops, fused producers), so writing to it is legitimate and must NOT abort,
// even though the kernel never resolved a pointer into it. Guards the exemption
// from regressing (which would re-introduce false positives on real in-place
// TT-NN ops).
TEST_F(MeshDeviceFixture, Object_Intent_IOArg_Exempt_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t buf_size = 1024;
    auto buffer_b = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    auto buffer_a = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    uint32_t addr_a = buffer_a->address();
    uint32_t addr_b = buffer_b->address();
    ASSERT_LT(addr_a, addr_b);

    // Resolve only A, then write into B. Critically, B's ABSOLUTE address is
    // passed as a runtime arg below, so B is an exempt I/O tensor.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a = get_arg_val<uint32_t>(0);
            uint32_t base_b = get_arg_val<uint32_t>(1);
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);
            uint32_t elements_to_b = (base_b - base_a) / sizeof(uint32_t);
            ptr_a[elements_to_b] = 0x900DDEAD;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // Pass B's ABSOLUTE address (not a gap) so B is exempt as an I/O tensor.
    SetRuntimeArgs(program, kernel, logical_core, {addr_a, addr_b});

    // Must NOT abort — B was handed to the kernel as a runtime arg.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Read-only violation: missing support for this

}  // namespace tt::tt_metal
