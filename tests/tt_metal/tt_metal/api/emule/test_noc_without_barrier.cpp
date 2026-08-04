// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.NoC_Barrier_*"

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
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
    CircularBufferConfig cb_config =
        CircularBufferConfig(2048, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
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

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Race Condition: cb_pop_front.*called while a NoC read is still pending.*");
}

// Page-accessor read path. The missing-barrier check must also fire when the
// pending counter is incremented via the page-accessor read (noc_async_read_page
// / noc_async_read_tile with a TensorAccessor), not just the raw noc_async_read.
// This is the increment site real reader kernels overwhelmingly use, and it is a
// distinct source line from the raw read — a refactor that moved/dropped it would
// leave a genuine missing-barrier bug on the page-accessor path uncaught otherwise.
// (Kept before the non-death controls below so its EXPECT_DEATH forks from a
// still-single-threaded parent — see the ordering note in the regression runner.)
TEST_F(MeshDeviceFixture, NoC_Barrier_Missing_AddrGen_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2048, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // DRAM source for the page-accessor read + a real L1 destination (so the
    // tensor/OOB check doesn't pre-empt the pop-time race check).
    auto src_buf = Buffer::create(device, 1024, 1024, BufferType::DRAM);
    auto dst_buf = Buffer::create(device, 1024, 1024, BufferType::L1);

    // The TensorAccessor reads its bank layout from compile-time args.
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(src_buf).append_to(reader_ct_args);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dram_base = get_arg_val<uint32_t>(0);
            uint32_t dst       = get_arg_val<uint32_t>(1);
            // Page-accessor read increments __emule_pending_noc_reads via the page path.
            constexpr auto args = TensorAccessorArgs<0>();
            auto s = TensorAccessor(args, dram_base, 1024);
            noc_async_read_page(0, s, dst);
            // MISSING: noc_async_read_barrier();
            cb_pop_front(0, 1);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});
    SetRuntimeArgs(program, kernel, logical_core, {src_buf->address(), dst_buf->address()});

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

// Multi-read + single barrier. Two reads bring the pending counter to 2; a single
// noc_async_read_barrier() must CLEAR it to zero (not decrement it by one), so the
// following cb_pop_front sees no pending read and does not abort. This pins the
// clear-to-zero semantic: if noc_async_read_barrier ever regressed to a decrement,
// the counter would still be 1 at the pop and this legitimate ≥2-read kernel would
// false-positive — which no single-read test can catch.
TEST_F(MeshDeviceFixture, NoC_Barrier_MultiRead_SingleBarrier_NoViolation) {
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

            cb_reserve_back(0, 1);
            cb_push_back(0, 1);
            cb_wait_front(0, 1);

            noc_async_read(src_addr, dst, 512);         // pending -> 1
            noc_async_read(src_addr, dst + 512, 512);   // pending -> 2
            noc_async_read_barrier();                   // clears to 0 (NOT decrement to 1)
            cb_pop_front(0, 1);                          // pending == 0 -> no race
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    // Must NOT abort — the single barrier cleared both in-flight reads.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
