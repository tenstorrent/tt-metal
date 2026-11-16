// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Launching a workload from a custom kernel config buffer (experimental::EnqueueMeshWorkload).
// Needs a device in slow dispatch; the fixture opens it with the HYBRID allocator.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/tt_metal/api/allocator/hybrid_allocator_fixture.hpp"

namespace tt::tt_metal {

using CustomKernelConfigBufferTest = HybridAllocatorTest;

// The allocator manages all of L1 above KERNEL_CONFIG, so the kernel-config window is empty (at
// most alignment padding): every Tensix program must run from a kernel config buffer.
class ZeroKernelConfigWindowTest : public HybridAllocatorTest {
protected:
    void SetUp() override {
        // Reading the HAL creates the runtime options, which cache the allocator mode: set HYBRID first.
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        worker_l1_size_ = hal::get_max_worker_l1_unreserved_size();
        HybridAllocatorTest::SetUp();
    }
};

namespace {

const CoreCoord kBlockedCore(0, 0);
const CoreCoord kFreeCore(1, 0);
const CoreRange kBothCores(kBlockedCore, kFreeCore);
const distributed::MeshCoordinate kDevice(0, 0);

constexpr uint32_t kPageSize = 1024;
constexpr uint32_t kCommonValue = 0xC0FFEE;

const char* kWriteConfigKernel =
    "tests/tt_metal/tt_metal/api/custom_kernel_config_buffer_tests/kernels/write_config.cpp";
const char* kComputeWriteConfigKernel =
    "tests/tt_metal/tt_metal/api/custom_kernel_config_buffer_tests/kernels/compute_write_config.cpp";

// The kernels and where each writes in its core's result page (see the kernels).
enum Kernel : uint32_t { kBrisc, kNcrisc, kCompute };
constexpr uint32_t kBriscSlot = 0;    // runtime arg, common runtime arg, CB 0 write pointer
constexpr uint32_t kNcriscSlot = 3;   // same
constexpr uint32_t kComputeSlot = 6;  // UNPACK, MATH, PACK runtime arg (+0, +1, +2), common runtime arg

// The runtime arg value `kernel` gets on the core at `core_index`.
uint32_t arg_value(Kernel kernel, uint32_t core_index) { return 0xA000 + (kernel * 0x100) + core_index; }

BufferShardingArgs one_page_per_core(const CoreRangeSet& cores) {
    return BufferShardingArgs(
        ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {cores.num_cores(), 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
}

// One `page_size` page on each of `cores`; per-core addresses when `per_core`, else one address.
std::shared_ptr<distributed::MeshBuffer> allocate(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t page_size, bool per_core) {
    BufferShardingArgs sharding_args = one_page_per_core(cores);
    experimental::per_core_allocation::set_per_core_allocation(sharding_args, per_core);
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<DeviceAddr>(page_size) * cores.num_cores()},
        distributed::DeviceLocalBufferConfig{
            .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = sharding_args},
        &device);
}

// write_config.cpp on BRISC and NCRISC and compute_write_config.cpp on the TRISCs, on `cores`,
// writing to `result` and reading CB 0 from `cb_backing`. Each kernel on each core gets its own
// runtime arg value; BRISC's is padded with `extra_args` runtime args to grow the block.
distributed::MeshWorkload make_workload(
    distributed::MeshDevice& device,
    const CoreRange& cores,
    const distributed::MeshBuffer& result,
    const distributed::MeshBuffer& cb_backing,
    uint32_t extra_args = 0) {
    Program program = CreateProgram();
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(kPageSize, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, kPageSize)
            .set_globally_allocated_address(*cb_backing.get_reference_buffer()));
    const KernelHandle brisc = CreateKernel(
        program,
        kWriteConfigKernel,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {kBriscSlot}});
    const KernelHandle ncrisc = CreateKernel(
        program,
        kWriteConfigKernel,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {kNcriscSlot}});
    const KernelHandle compute =
        CreateKernel(program, kComputeWriteConfigKernel, cores, ComputeConfig{.compile_args = {kComputeSlot}});

    for (const auto& [kernel, handle] : {std::pair{kBrisc, brisc}, {kNcrisc, ncrisc}, {kCompute, compute}}) {
        SetCommonRuntimeArgs(program, handle, {static_cast<uint32_t>(result.address()), kCommonValue});
        uint32_t core_index = 0;
        for (const CoreCoord& core : cores) {
            std::vector<uint32_t> args(1 + (kernel == kBrisc ? extra_args : 0), 0);
            args[0] = arg_value(kernel, core_index++);
            SetRuntimeArgs(program, handle, core, args);
        }
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    return workload;
}

// Runs `launch` and expects it to throw a message containing `text`.
template <typename F>
void expect_throw_containing(F&& launch, const std::string& text) {
    try {
        launch();
        ADD_FAILURE() << "expected a throw containing \"" << text << "\"";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
    }
}

}  // namespace

// Each core runs from its own shard, at its own address: every RISC (BRISC, NCRISC, and the three
// TRISCs) runs its kernel from there and reads its runtime args, and the data-movement RISCs their CB
// config, from there. The shard holds the runtime args.
TEST_F(CustomKernelConfigBufferTest, RunsFromBufferAtPerCoreAddresses) {
    auto& device = *this->devices_[0];
    // Larger than the block, so kFreeCore's shard takes the slot this leaves free there.
    auto blocker = allocate(device, CoreRangeSet(CoreRange(kBlockedCore)), 64 * 1024, /*per_core=*/true);
    auto result = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);
    auto cb_backing = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);

    auto workload = make_workload(device, kBothCores, *result, *cb_backing);
    auto buffer = experimental::CreateKernelConfigBuffer(device, workload);
    ASSERT_NE(buffer.address(kDevice, kBlockedCore), buffer.address(kDevice, kFreeCore));

    experimental::EnqueueMeshWorkload(device.mesh_command_queue(), workload, buffer, /*blocking=*/true);

    std::vector<uint32_t> readback;
    distributed::ReadShard(device.mesh_command_queue(), readback, result, kDevice);
    constexpr uint32_t kWordsPerPage = kPageSize / sizeof(uint32_t);
    ASSERT_EQ(readback.size(), 2 * kWordsPerPage);

    IDevice* chip = device.get_devices().at(0);
    uint32_t core_index = 0;
    for (const CoreCoord& core : kBothCores) {
        const uint32_t* page = readback.data() + (core_index * kWordsPerPage);
        for (const auto& [kernel, slot, name] :
             {std::tuple{kBrisc, kBriscSlot, "BRISC"}, std::tuple{kNcrisc, kNcriscSlot, "NCRISC"}}) {
            EXPECT_EQ(page[slot], arg_value(kernel, core_index)) << name << " runtime arg on " << core.str();
            EXPECT_EQ(page[slot + 1], kCommonValue) << name << " common runtime arg on " << core.str();
            EXPECT_EQ(page[slot + 2], cb_backing->address()) << name << " CB write pointer on " << core.str();
        }
        const uint32_t compute_value = arg_value(kCompute, core_index);
        EXPECT_EQ(page[kComputeSlot], compute_value) << "UNPACK runtime arg on " << core.str();
        EXPECT_EQ(page[kComputeSlot + 1], compute_value + 1) << "MATH runtime arg on " << core.str();
        EXPECT_EQ(page[kComputeSlot + 2], compute_value + 2) << "PACK runtime arg on " << core.str();
        EXPECT_EQ(page[kComputeSlot + 3], kCommonValue) << "UNPACK common runtime arg on " << core.str();

        std::vector<uint32_t> shard;
        detail::ReadFromDeviceL1(
            chip, core, static_cast<uint32_t>(buffer.address(kDevice, core)), buffer.size(kDevice, core), shard);
        for (const Kernel kernel : {kBrisc, kNcrisc, kCompute}) {
            EXPECT_NE(std::find(shard.begin(), shard.end(), arg_value(kernel, core_index)), shard.end())
                << "kernel " << kernel << "'s runtime arg not in the shard on " << core.str();
        }
        core_index++;
    }
}

TEST_F(CustomKernelConfigBufferTest, RejectsBufferThatDoesNotFit) {
    auto& device = *this->devices_[0];
    auto result = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);
    auto cb_backing = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);

    // Sized for one core: the buffer does not cover the second.
    auto one_core = make_workload(device, CoreRange(kBlockedCore), *result, *cb_backing);
    auto one_core_buffer = experimental::CreateKernelConfigBuffer(device, one_core);
    auto both_cores = make_workload(device, kBothCores, *result, *cb_backing);
    EXPECT_ANY_THROW(experimental::EnqueueMeshWorkload(device.mesh_command_queue(), both_cores, one_core_buffer, true));

    // Sized for a smaller block on the same cores: 1 KB more runtime args do not fit.
    auto small = make_workload(device, kBothCores, *result, *cb_backing);
    auto small_buffer = experimental::CreateKernelConfigBuffer(device, small);
    auto large = make_workload(device, kBothCores, *result, *cb_backing, /*extra_args=*/256);
    EXPECT_ANY_THROW(experimental::EnqueueMeshWorkload(device.mesh_command_queue(), large, small_buffer, true));
}

// A blaze-like device: no kernel-config window, and L1 full on every core but one. A program on
// that core cannot launch the usual way, but runs from a kernel config buffer placed in its free L1.
TEST_F(ZeroKernelConfigWindowTest, RunsWithNoWindowFromTheOnlyFreeCore) {
    auto& device = *this->devices_[0];
    const CoreCoord grid = device.compute_with_storage_grid_size();
    const CoreRangeSet free_core{CoreRange(kFreeCore)};
    const CoreRangeSet full_cores = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})).subtract(free_core);

    // Lockstep buffers first, as in blaze: a lockstep address must be free on every core, so none fits
    // once the per-core fill below is in place.
    auto result = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);
    auto cb_backing = allocate(device, CoreRangeSet(kBothCores), kPageSize, /*per_core=*/false);

    // Fill the rest of L1 on every core but the free one, per core: a lockstep fill would take the free
    // core's addresses too.
    const auto free_bytes =
        static_cast<uint32_t>(device.allocator()->get_statistics(BufferType::L1).largest_free_block_bytes);
    auto filler = allocate(device, full_cores, free_bytes, /*per_core=*/true);

    // Without a window, the usual launch has nowhere to put the block.
    auto usual = make_workload(device, CoreRange(kFreeCore), *result, *cb_backing);
    expect_throw_containing(
        [&] { distributed::EnqueueMeshWorkload(device.mesh_command_queue(), usual, /*blocking=*/true); },
        "too large for kernel config buffer");

    // A full core has no room for a buffer either.
    auto on_full_core = make_workload(device, CoreRange(kBlockedCore), *result, *cb_backing);
    expect_throw_containing([&] { experimental::CreateKernelConfigBuffer(device, on_full_core); }, "Out of Memory");

    auto workload = make_workload(device, CoreRange(kFreeCore), *result, *cb_backing);
    auto buffer = experimental::CreateKernelConfigBuffer(device, workload);
    experimental::EnqueueMeshWorkload(device.mesh_command_queue(), workload, buffer, /*blocking=*/true);

    std::vector<uint32_t> readback;
    distributed::ReadShard(device.mesh_command_queue(), readback, result, kDevice);
    constexpr uint32_t kWordsPerPage = kPageSize / sizeof(uint32_t);
    ASSERT_EQ(readback.size(), 2 * kWordsPerPage);
    const uint32_t* page = readback.data() + kWordsPerPage;  // kFreeCore's page: the second core
    for (const auto& [kernel, slot] : {std::pair{kBrisc, kBriscSlot}, std::pair{kNcrisc, kNcriscSlot}}) {
        EXPECT_EQ(page[slot], arg_value(kernel, 0)) << "kernel " << kernel;
        EXPECT_EQ(page[slot + 1], kCommonValue) << "kernel " << kernel;
        EXPECT_EQ(page[slot + 2], cb_backing->address()) << "kernel " << kernel;
    }
    EXPECT_EQ(page[kComputeSlot], arg_value(kCompute, 0));
    EXPECT_EQ(page[kComputeSlot + 1], arg_value(kCompute, 0) + 1);
    EXPECT_EQ(page[kComputeSlot + 2], arg_value(kCompute, 0) + 2);
    EXPECT_EQ(page[kComputeSlot + 3], kCommonValue);
}

}  // namespace tt::tt_metal
