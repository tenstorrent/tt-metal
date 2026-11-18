// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Creating a custom kernel config buffer (experimental::CreateKernelConfigBuffer).
// Needs a device in slow dispatch; the fixture opens it with the HYBRID allocator.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "tests/tt_metal/tt_metal/api/allocator/hybrid_allocator_fixture.hpp"

namespace tt::tt_metal {

using CustomKernelConfigBufferTest = HybridAllocatorTest;

namespace {

const CoreCoord kBlockedCore(0, 0);
const CoreCoord kFreeCore(1, 0);
const distributed::MeshCoordinate kDevice(0, 0);

// A blank kernel on both cores, on every device of the mesh.
distributed::MeshWorkload make_workload(distributed::MeshDevice& device) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRange(kBlockedCore, kFreeCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    return workload;
}

// A per-core L1 allocation on `core` alone, taking the slot the next allocation would get there.
std::shared_ptr<distributed::MeshBuffer> allocate_on(
    distributed::MeshDevice& device, const CoreCoord& core, bool bottom_up = false) {
    constexpr uint32_t kSize = 1024;
    BufferShardingArgs sharding_args(
        ShardSpecBuffer(CoreRangeSet(CoreRange(core)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kSize},
        distributed::DeviceLocalBufferConfig{
            .page_size = kSize, .buffer_type = BufferType::L1, .sharding_args = sharding_args, .bottom_up = bottom_up},
        &device);
}

}  // namespace

// The buffer has a shard on each of the workload's cores and is placed per core: a core whose first free slot is
// taken gets a different address from its neighbour.
TEST_F(CustomKernelConfigBufferTest, CreatesPerCoreBufferForWorkload) {
    auto& device = *this->devices_[0];
    auto blocker = allocate_on(device, kBlockedCore);

    auto workload = make_workload(device);
    auto buffer = experimental::CreateKernelConfigBuffer(device, workload);

    EXPECT_GT(buffer.size(kDevice, kBlockedCore), 0u);
    EXPECT_GT(buffer.size(kDevice, kFreeCore), 0u);
    EXPECT_NE(buffer.address(kDevice, kBlockedCore), buffer.address(kDevice, kFreeCore));
}

// Each core's shard is sized for its own kernel group: kFreeCore also runs a second kernel, so its block, and its
// shard, is larger than kBlockedCore's.
TEST_F(CustomKernelConfigBufferTest, SizesEachCoreForItsKernelGroup) {
    auto& device = *this->devices_[0];

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRange(kBlockedCore, kFreeCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        kFreeCore,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));

    auto buffer = experimental::CreateKernelConfigBuffer(device, workload);

    EXPECT_LT(buffer.size(kDevice, kBlockedCore), buffer.size(kDevice, kFreeCore));
    EXPECT_EQ(buffer.max_size(), buffer.size(kDevice, kFreeCore));
}

// A lockstep address: both cores' shards at one address, sized alike, so a program with Tensix program semaphores
// may run from it.
TEST_F(CustomKernelConfigBufferTest, LockstepAddressPutsEveryShardAtOneAddress) {
    auto& device = *this->devices_[0];

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRange(kBlockedCore, kFreeCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateSemaphore(program, CoreRange(kBlockedCore, kFreeCore), 0);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));

    auto buffer = experimental::CreateKernelConfigBuffer(device, workload, /*lockstep_address=*/true);

    EXPECT_EQ(buffer.address(kDevice, kBlockedCore), buffer.address(kDevice, kFreeCore));
    EXPECT_EQ(buffer.size(kDevice, kBlockedCore), buffer.size(kDevice, kFreeCore));
}

// A lockstep address needs the cores' free L1 to start at the same address: here one core's lowest slot is taken.
TEST_F(CustomKernelConfigBufferTest, LockstepAddressRejectsCoresWithDifferentFreeL1) {
    auto& device = *this->devices_[0];
    auto blocker = allocate_on(device, kBlockedCore, /*bottom_up=*/true);

    auto workload = make_workload(device);
    EXPECT_ANY_THROW(experimental::CreateKernelConfigBuffer(device, workload, /*lockstep_address=*/true));
}

TEST_F(CustomKernelConfigBufferTest, RejectsProgramSemaphores) {
    auto& device = *this->devices_[0];

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRange(kBlockedCore, kFreeCore),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateSemaphore(program, CoreRange(kBlockedCore, kFreeCore), 0);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));

    EXPECT_ANY_THROW(experimental::CreateKernelConfigBuffer(device, workload));
}

}  // namespace tt::tt_metal
