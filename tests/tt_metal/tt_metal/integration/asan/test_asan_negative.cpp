// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-path ASan negative tests. Each test triggers a memory violation that
// the per-buffer poison hook in AllocatorImpl should detect via ASan. Under
// ASan-instrumented JIT kernel .so files, the violation aborts the process
// with a non-zero exit. The regression script's wrapper treats that as PASS.
//
// IF YOU SEE A FAIL() LINE BELOW: the hook regressed, ASan did not fire, and
// the test ran past where it should have died.
//
// Modeled on the Quasar matmul tests in
// tests/tt_metal/tt_metal/integration/matmul/test_matmul_X_tile.cpp:
// distributed::MeshBuffer::create + fixture->RunProgram is the path that
// works under emulation. CreateBuffer(InterleavedBufferConfig) hangs in
// device init under MeshDispatchFixture and must not be used here.

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>

#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {

namespace {

// Allocate an L1 mesh buffer with the same layout used by the regression's
// existing matmul tests. Returns the (mesh-replicated) L1 address.
std::shared_ptr<distributed::MeshBuffer> create_l1_mesh_buffer(distributed::MeshDevice* mesh_device, uint32_t size) {
    distributed::DeviceLocalBufferConfig l1_cfg{.page_size = size, .buffer_type = BufferType::L1, .bottom_up = false};
    distributed::ReplicatedBufferConfig replicated_cfg{.size = size};
    return distributed::MeshBuffer::create(replicated_cfg, l1_cfg, mesh_device);
}

// DRAM analog: replicated DRAM mesh buffer.
std::shared_ptr<distributed::MeshBuffer> create_dram_mesh_buffer(distributed::MeshDevice* mesh_device, uint32_t size) {
    distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size = size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig replicated_cfg{.size = size};
    return distributed::MeshBuffer::create(replicated_cfg, dram_cfg, mesh_device);
}

}  // namespace

// Positive control. Allocates one L1 buffer and writes 1 byte at offset 0.
// With the per-buffer alloc hook unpoisoning the buffer's region, the store
// inside the JIT'd kernel must succeed without firing ASan. If ASan DOES
// fire, the alloc hook isn't running — and the OOB test below is passing on
// the initial blanket poison, not real per-buffer poisoning.
//
// This test runs as a normal gtest pass (no WILL_FAIL TRUE in the regression
// script); a failure here is a regression of the alloc-side hook.
TEST_F(MeshDispatchFixture, AsanL1BufferInBoundsWrite) {
    ASSERT_GT(devices_.size(), 0u);
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kBufSize = 4096;
    auto buffer = create_l1_mesh_buffer(mesh_device.get(), kBufSize);
    const uint32_t l1_addr = static_cast<uint32_t>(buffer->address());

    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto range = distributed::MeshCoordinateRange(zero, zero);
    Program program = CreateProgram();
    workload.add_program(range, std::move(program));
    auto& program_ = workload.get_programs().at(range);

    const CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/asan_l1_inbounds_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });
    SetRuntimeArgs(program_, kernel, core, {l1_addr});

    RunProgram(mesh_device, workload);
    // Reaching this line means the in-bounds write succeeded. Good.
}

TEST_F(MeshDispatchFixture, AsanL1BufferOverflow) {
    ASSERT_GT(devices_.size(), 0u);
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kBufSize = 4096;
    auto buffer = create_l1_mesh_buffer(mesh_device.get(), kBufSize);
    // Allocate a guard buffer immediately above so the byte right past
    // `buffer` is firmly inside another buffer's poisoned (yet-to-be-
    // accessed) range.
    auto guard_buffer = create_l1_mesh_buffer(mesh_device.get(), kBufSize);
    (void)guard_buffer;
    const uint32_t l1_addr = static_cast<uint32_t>(buffer->address());

    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto range = distributed::MeshCoordinateRange(zero, zero);
    Program program = CreateProgram();
    workload.add_program(range, std::move(program));
    auto& program_ = workload.get_programs().at(range);

    const CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/asan_l1_oob_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });
    SetRuntimeArgs(program_, kernel, core, {l1_addr, kBufSize});

    RunProgram(mesh_device, workload);

    FAIL() << "expected ASan to abort on L1 OOB write";
}

// TODO(asan): use-after-free repoisoning isn't catching writes through a
// freed MeshBuffer's saved address. Either MeshBuffer's destructor path
// doesn't reach AllocatorImpl::deallocate_buffer (and so on_buffer_deallocated
// isn't running), or the kernel-side address translation doesn't hit the
// repoisoned range. Disabled until investigated. The OOB test above proves
// the per-buffer poisoning hook is wired correctly on the alloc side; the
// UAF test would prove the symmetric dealloc side.
TEST_F(MeshDispatchFixture, AsanL1BufferUseAfterFree) {
    ASSERT_GT(devices_.size(), 0u);
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kBufSize = 4096;
    auto buffer = create_l1_mesh_buffer(mesh_device.get(), kBufSize);
    const uint32_t l1_addr = static_cast<uint32_t>(buffer->address());
    buffer.reset();  // dealloc -> on_buffer_deallocated -> __emule_buffer_free -> repoison

    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto range = distributed::MeshCoordinateRange(zero, zero);
    Program program = CreateProgram();
    workload.add_program(range, std::move(program));
    auto& program_ = workload.get_programs().at(range);

    const CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/asan_l1_uaf_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });
    SetRuntimeArgs(program_, kernel, core, {l1_addr});

    RunProgram(mesh_device, workload);

    FAIL() << "expected ASan to abort on use-after-free L1 write";
}

// Symmetric DRAM use-after-free. The freed DRAM bank pages are repoisoned by
// the per-buffer dealloc hook. The kernel issues a NOC read from the freed
// bank-relative address; under ASan the host-side memcpy in the resolver
// traps with use-after-poison.
TEST_F(MeshDispatchFixture, AsanDramBufferUseAfterFree) {
    ASSERT_GT(devices_.size(), 0u);
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kBufSize = 4096;
    auto buffer = create_dram_mesh_buffer(mesh_device.get(), kBufSize);
    const uint32_t dram_addr = static_cast<uint32_t>(buffer->address());
    buffer.reset();

    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto range = distributed::MeshCoordinateRange(zero, zero);
    Program program = CreateProgram();
    workload.add_program(range, std::move(program));
    auto& program_ = workload.get_programs().at(range);

    const CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/asan_dram_uaf_read.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });
    constexpr uint32_t kBankId = 0;
    SetRuntimeArgs(program_, kernel, core, {dram_addr, kBankId});

    RunProgram(mesh_device, workload);

    FAIL() << "expected ASan to abort on use-after-free DRAM read";
}

}  // namespace tt::tt_metal
