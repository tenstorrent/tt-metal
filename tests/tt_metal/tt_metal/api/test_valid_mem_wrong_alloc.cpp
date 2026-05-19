// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Object*"

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

TEST_F(MeshDeviceFixture, Object_Intent_Provenance_Violation_SanityCheck) {
    ::setenv("TT_EMULE_STRICT_TENSOR", "1", 1);
    ::setenv("TT_EMULE_STRICT_OBJECT_INTENT", "1", 1);

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

    // 2. Add an inline kernel that overshoots Buffer A to stomp on Buffer B
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a = get_arg_val<uint32_t>(0);
            uint32_t base_b = get_arg_val<uint32_t>(1);

            // Establish pointer provenance linked explicitly to Buffer A's base address context
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);
            
            // Calculate the exact stride gap separating the two valid objects
            uint32_t elements_to_b = (base_b - base_a) / sizeof(uint32_t);
            
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
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    
    // Pass both buffer addresses to enable the boundary calculation inside the kernel
    SetRuntimeArgs(program, kernel, logical_core, {addr_a, addr_b});

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
    ::setenv("TT_EMULE_STRICT_TENSOR", "1", 1);
    ::setenv("TT_EMULE_STRICT_OBJECT_INTENT", "1", 1);

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
    ::setenv("TT_EMULE_STRICT_TENSOR", "1", 1);
    ::setenv("TT_EMULE_STRICT_OBJECT_INTENT", "1", 1);

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

    // The kernel only resolves Buffer A. base_c is passed as a plain uint32_t
    // for arithmetic only — it never goes through __emule_local_l1_to_ptr, so
    // Buffer C never enters the resolved set. The store lands inside Buffer C
    // (valid L1, valid allocation, but not the kernel's intended object).
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_a = get_arg_val<uint32_t>(0);
            uint32_t base_c = get_arg_val<uint32_t>(1);
            volatile uint32_t* ptr_a = (volatile uint32_t*)__emule_local_l1_to_ptr(base_a);
            uint32_t elements_to_c = (base_c - base_a) / sizeof(uint32_t);
            ptr_a[elements_to_c] = 0xDEADBEEF;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    SetRuntimeArgs(program, kernel, logical_core, {addr_a, addr_c});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Object Intent Violation: Attempted to modify memory belonging to an adjacent object context.*"
    );
}

// Read-only violation: missing support for this

}  // namespace tt::tt_metal