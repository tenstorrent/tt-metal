// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.Multicast_SourceInRectangle*"
//
// A multicast whose rectangle contains the sender reaches the sender only when the API sets
// NOC_CMD_BRCST_SRC_INCLUDE (the *_loopback_src variants); every other multicast skips it, on both NOCs.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace {

// Kernel mode order; loopback modes are the only ones that set NOC_CMD_BRCST_SRC_INCLUDE.
enum McastMode : uint32_t {
    WRITE,
    WRITE_LOOPBACK_SRC,
    SEMAPHORE_SET,
    SEMAPHORE_SET_LOOPBACK_SRC,
    SEMAPHORE_INC,
    SEMAPHORE_CLASS_INC,
    INLINE_DW_WRITE,
    NUM_MODES,
};

constexpr const char* kModeNames[NUM_MODES] = {
    "noc_async_write_multicast",
    "noc_async_write_multicast_loopback_src",
    "noc_semaphore_set_multicast",
    "noc_semaphore_set_multicast_loopback_src",
    "noc_semaphore_inc_multicast",
    "Semaphore::inc_multicast",
    "noc_inline_mcast_dw_write",
};

bool is_loopback(uint32_t mode) { return mode == WRITE_LOOPBACK_SRC || mode == SEMAPHORE_SET_LOOPBACK_SRC; }

constexpr uint32_t kNumCases = 2 /*noc*/ * 2 /*rectangle*/ * NUM_MODES;
constexpr uint32_t kPageBytes = 512;  // results in [0, 8 * kNumCases), scratch words above

// The sender runs every (noc, rectangle, mode) case in turn: reset the target word on both cores, multicast
// one 1 into it, then read back both copies. The remote core only stays resident until the sender finishes.
const char* kKernel = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_modes = get_compile_time_arg_val(5);
    const uint32_t target = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t ready = get_semaphore(get_compile_time_arg_val(3));
    const uint32_t finish = get_semaphore(get_compile_time_arg_val(4));
    const uint32_t sx = get_arg_val<uint32_t>(2), sy = get_arg_val<uint32_t>(3);
    const uint32_t rx = get_arg_val<uint32_t>(4), ry = get_arg_val<uint32_t>(5);
    if (get_arg_val<uint32_t>(1) == 0) {
        noc_semaphore_inc(get_noc_addr(sx, sy, ready), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(finish), 1);
        return;
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
    const auto output = TensorAccessor(TensorAccessorArgs<6>(), get_arg_val<uint32_t>(0));
    cb_reserve_back(cb, 1);
    const uint32_t results = get_write_ptr(cb);
    const uint32_t readback = results + page_bytes - 96;  // 32-byte-aligned scratch words
    const uint32_t zero = results + page_bytes - 64;
    const uint32_t one = results + page_bytes - 32;
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one) = 1;
    uint32_t idx = 0;
    for (uint32_t noc = 0; noc < 2; ++noc) {
        for (uint32_t includes_source = 0; includes_source < 2; ++includes_source) {
            // NOC1 names the rectangle from its own origin, so start and end trade places.
            const uint32_t x0 = includes_source ? sx : rx, y0 = includes_source ? sy : ry;
            const uint32_t ax = noc == 0 ? x0 : rx, ay = noc == 0 ? y0 : ry;
            const uint32_t bx = noc == 0 ? rx : x0, by = noc == 0 ? ry : y0;
            const uint64_t mcast = get_noc_multicast_addr(ax, ay, bx, by, target);
            const uint32_t loopback_dests = includes_source ? 2 : 1;
            for (uint32_t mode = 0; mode < num_modes; ++mode) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target), 0);
                noc_semaphore_set_remote(zero, get_noc_addr(rx, ry, target));
                noc_async_write_barrier();
                switch (mode) {
                    case 0: noc_async_write_multicast(one, mcast, 4, 1, false, noc); break;
                    case 1: noc_async_write_multicast_loopback_src(one, mcast, 4, loopback_dests, false, noc); break;
                    case 2: noc_semaphore_set_multicast(one, mcast, 1, false, noc); break;
                    case 3: noc_semaphore_set_multicast_loopback_src(one, mcast, loopback_dests, false, noc); break;
                    case 4: noc_semaphore_inc_multicast(mcast, 1, 1, noc); break;
                    case 5: Semaphore<>(get_compile_time_arg_val(2)).inc_multicast(Noc(noc), ax, ay, bx, by, 1, 1); break;
                    case 6: noc_inline_mcast_dw_write(mcast, 1, 0xF, noc); break;
                }
                noc_async_write_barrier(noc);
                noc_async_atomic_barrier(noc);
                noc_async_read(get_noc_addr(sx, sy, target), readback, 4);
                noc_async_read_barrier();
                words[idx++] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readback);
                noc_async_read(get_noc_addr(rx, ry, target), readback, 4);
                noc_async_read_barrier();
                words[idx++] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readback);
            }
        }
    }
    cb_push_back(cb, 1);
    cb_wait_front(cb, 1);
    noc_async_write_page(0, output, get_read_ptr(cb), page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb, 1);
    noc_semaphore_inc(get_noc_addr(rx, ry, finish), 1);
    noc_async_atomic_barrier();
}
)";

}  // namespace

TEST_F(UnitMeshFixture, Multicast_SourceInRectangle_ExcludesSenderUnlessLoopback) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar's atomic multicast always sets SRC_INCLUDE; this fence is for the Gen1 NOC.";
    }
    auto& mesh_device = this->device();
    const CoreCoord source(0, 0), remote(1, 0);
    const CoreRange cores(source, remote);
    const CoreCoord source_noc = mesh_device.worker_core_from_logical_core(source);
    const CoreCoord remote_noc = mesh_device.worker_core_from_logical_core(remote);

    auto output = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kPageBytes},
        {.page_size = kPageBytes, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto& cq = mesh_device.mesh_command_queue();
    std::vector<uint32_t> result(kPageBytes / sizeof(uint32_t), 0xDEADBEEF);
    distributed::EnqueueWriteMeshBuffer(cq, output, result, true);

    Program program = CreateProgram();
    const uint32_t target = CreateSemaphore(program, cores, 0);
    const uint32_t ready = CreateSemaphore(program, cores, 0);
    const uint32_t finish = CreateSemaphore(program, cores, 0);
    CreateCircularBuffer(
        program,
        source,
        CircularBufferConfig(kPageBytes, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, kPageBytes));
    std::vector<uint32_t> compile_args{0, kPageBytes, target, ready, finish, NUM_MODES};
    TensorAccessorArgs(*output).append_to(compile_args);
    auto kernel = CreateKernelFromString(
        program,
        kKernel,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});
    for (const auto& core : {source, remote}) {
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(output->address()),
             static_cast<uint32_t>(core == source),
             static_cast<uint32_t>(source_noc.x),
             static_cast<uint32_t>(source_noc.y),
             static_cast<uint32_t>(remote_noc.x),
             static_cast<uint32_t>(remote_noc.y)});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    distributed::EnqueueReadMeshBuffer(cq, result, output, true);

    uint32_t idx = 0;
    for (uint32_t noc = 0; noc < 2; ++noc) {
        for (uint32_t includes_source = 0; includes_source < 2; ++includes_source) {
            for (uint32_t mode = 0; mode < NUM_MODES; ++mode) {
                SCOPED_TRACE(
                    std::string(kModeNames[mode]) + " noc=" + std::to_string(noc) +
                    (includes_source ? " rectangle={sender,remote}" : " rectangle={remote}"));
                const uint32_t expected_self = (includes_source && is_loopback(mode)) ? 1 : 0;
                EXPECT_EQ(result[idx++], expected_self) << "sender's copy";
                EXPECT_EQ(result[idx++], 1u) << "remote's copy";
            }
        }
    }
    ASSERT_EQ(idx, 2 * kNumCases);
}

}  // namespace tt::tt_metal
