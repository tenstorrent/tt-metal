// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.Semaphore_*"

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, Semaphore_Direct_Write_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // EMULE_SEM_BASE is a JIT-time define injected by the runner: the
    // firmware-style L1 offset of the reserved Semaphore region. Any scalar
    // pointer access into that range must trip the ASAN guard inside
    // __emule_local_l1_to_ptr and abort the kernel thread.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t sem_addr = EMULE_SEM_BASE;
            volatile uint32_t* illegal_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(sem_addr);
            *illegal_ptr = 0xABCD;
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true),
        ".*Illegal Semaphore Access: Offset 0x.*is inside the reserved Semaphore region.*");
}

// Positive control: a scalar access to an ordinary allocated L1 buffer — well
// OUTSIDE the reserved semaphore region — must NOT abort. Guards the semaphore
// check from flagging normal L1 addressing (the region test must be a precise
// [start, end) containment, not an over-broad lower-bound).
TEST_F(UnitMeshFixture, Semaphore_OutsideRegion_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // A normal L1 buffer is allocated well away from the reserved semaphore
    // region (which lives in the low system area near EMULE_SEM_BASE).
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 1024},
        {.page_size = 1024, .buffer_type = BufferType::L1},
        &this->device());
    uint32_t addr = static_cast<uint32_t>(buf->address());

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t a = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(a);
            *ptr = 0x1234;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {addr});

    // Must NOT abort.
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Negative control: the canonical raw-pointer semaphore idiom — a kernel
// casting its OWN get_semaphore() address and reading/writing the word directly
// (matmul reader_bmm_tile_layout_in0_receiver.cpp's mcast-payload read,
// sdpa_decode writer_decode_all.cpp's nibble poll). Legal on silicon (the
// reserved region is plain L1), so it must NOT trip the Illegal-Semaphore
// check: the JIT patch pass's semaphore-provenance rules (S1 inline / S2
// store-then-cast) route these casts through __emule_sem_l1_to_ptr. Exercises
// both forms through the real JIT pipeline.
TEST_F(UnitMeshFixture, Semaphore_RawGetSemaphoreCast_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();
    uint32_t sem_id = CreateSemaphore(program, logical_core, /*initial_value=*/3);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Inline form (patch-pass rule S1), nested compile-time arg.
            volatile tt_l1_ptr uint32_t* p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(0)));
            uint32_t v = *p;
            *p = v + 1;
            // Store-then-cast form (patch-pass rule S2) — the shape of the
            // reader_bmm in0_receiver / writer_decode_all kernels.
            uint32_t sem_addr = get_semaphore(get_compile_time_arg_val(0));
            volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            *q = v;
        }
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {sem_id}});

    // Must NOT abort.
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Negative control: the sanctioned Semaphore<> API's own remote-relay path
// (sdpa reader_interleaved.cpp's KV-chain forward). relay_unicast reads the
// LOCAL semaphore word as its NOC source — emule's noc_semaphore_set_remote
// must translate that source via the sanctioned-semaphore path, not the
// checked chokepoint, or the API itself trips the Illegal-Semaphore check.
TEST_F(UnitMeshFixture, Semaphore_RelayUnicast_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {1, 0};
    CoreRange both_cores(sender_core, receiver_core);
    Program program = CreateProgram();
    uint32_t src_sem_id = CreateSemaphore(program, both_cores, /*initial_value=*/0);
    uint32_t dst_sem_id = CreateSemaphore(program, both_cores, /*initial_value=*/0);

    std::string sender_src = R"(
        #include "api/dataflow/dataflow_api.h"
        #include "api/dataflow/noc_semaphore.h"
        void kernel_main() {
            uint32_t rx_x = get_arg_val<uint32_t>(0);
            uint32_t rx_y = get_arg_val<uint32_t>(1);
            Noc noc;
            Semaphore<> src_sem(get_compile_time_arg_val(0));
            Semaphore<> dst_sem(get_compile_time_arg_val(1));
            src_sem.set(7);
            src_sem.relay_unicast(noc, dst_sem, rx_x, rx_y);
        }
    )";
    std::string receiver_src = R"(
        #include "api/dataflow/dataflow_api.h"
        #include "api/dataflow/noc_semaphore.h"
        void kernel_main() {
            Semaphore<> dst_sem(get_compile_time_arg_val(0));
            dst_sem.wait_min(7);
        }
    )";
    auto sender_kernel = CreateKernelFromString(
        program,
        sender_src,
        sender_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {src_sem_id, dst_sem_id}});
    CreateKernelFromString(
        program,
        receiver_src,
        receiver_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {dst_sem_id}});
    CoreCoord rx_virtual = this->device().worker_core_from_logical_core(receiver_core);
    SetRuntimeArgs(program, sender_kernel, sender_core, {rx_virtual.x, rx_virtual.y});

    // Must NOT abort (and must not hang: the relay's value wakes the waiter).
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Guard: the semaphore-provenance exemption is REGION-BOUNDED, not a blanket
// pass for anything derived from get_semaphore. A sem-derived address that
// wanders out of the reserved region falls through to the full check chain and
// must still die (here: Out-of-Bounds Write — well above the unreserved base,
// inside no allocated tensor).
TEST_F(UnitMeshFixture, Semaphore_SemDerivedOutsideRegion_StillChecked) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();
    uint32_t sem_id = CreateSemaphore(program, logical_core, /*initial_value=*/0);
    // Allocate a buffer so the live-tensor range set is armed (non-null).
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 1024},
        {.page_size = 1024, .buffer_type = BufferType::L1},
        &this->device());

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t sem_addr = get_semaphore(get_compile_time_arg_val(0));
            volatile tt_l1_ptr uint32_t* p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr + 0x100000);
            *p = 1;
        }
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {sem_id}});

    EXPECT_DEATH(
        LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true), ".*Out-of-Bounds Write.*");
    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
