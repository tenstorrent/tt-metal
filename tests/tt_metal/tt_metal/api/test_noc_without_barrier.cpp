// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.NoC_Barrier_*"

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, NoC_Barrier_Missing_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 1. Create a CB
    uint32_t cb_id = 0;
    CircularBufferConfig cb_config = CircularBufferConfig(2048, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Allocate a real L1 buffer as the noc_async_read destination so the
    // tensor-area sanitizer doesn't fire before cb_pop_front triggers the
    // barrier check.
    auto dst_buf = Buffer::create(device, 1024, 1024, BufferType::L1);

    // 2. Kernel that reads then pops WITHOUT a barrier. Popping frees the page
    //    for the producer to refill while the read is still in flight — a race.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst = get_arg_val<uint32_t>(0);
            uint64_t src_addr = get_noc_addr(0x20000);

            noc_async_read(src_addr, dst, 1024);
            // MISSING: noc_async_read_barrier();
            cb_pop_front(0, 1);
        }
    )";

    auto kernel = CreateKernelFromString(program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Race Condition: cb_pop_front.*called while a NoC read is still pending.*");
}

// Positive control: the SAME read-then-pop sequence with the barrier PRESENT
// must NOT abort. noc_async_read_barrier() clears the pending-read counter, so
// the subsequent cb_pop_front sees zero in-flight reads. Guards the check from
// firing when the kernel correctly barriers (and confirms the barrier actually
// resets the counter). The CB is cycled in balance so nothing else fires.
TEST_F(MeshDeviceFixture, NoC_Barrier_Present_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2048, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    auto dst_buf = Buffer::create(device, 1024, 1024, BufferType::L1);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst = get_arg_val<uint32_t>(0);
            uint64_t src_addr = get_noc_addr(0x20000);

            // Balance the CB so the pop is legal and no Dirty-CB fires.
            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            cb_wait_front(0, 1);

            noc_async_read(src_addr, dst, 1024);
            noc_async_read_barrier();   // clears the pending-read counter
            cb_pop_front(0, 1);         // no pending read -> no race
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    // Must NOT abort.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
