// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A globally allocated CB whose backing buffer is bound after the program is built: a CBDescriptor that sets
// globally_allocated but names no buffer. The program compiles and lays out its kernel config block first, and
// launches once a descriptor names the buffer.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {
namespace {

const CoreCoord kCore(0, 0);
const distributed::MeshCoordinate kDevice(0, 0);
constexpr uint32_t kPageSize = 1024;

// Writes CB 0's write pointer to the result page.
const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/write_cb_write_ptr.cpp";

std::shared_ptr<distributed::MeshBuffer> allocate_page(distributed::MeshDevice& device) {
    const CoreRangeSet cores{CoreRange(kCore)};
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kPageSize},
        distributed::DeviceLocalBufferConfig{
            .page_size = kPageSize,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(
                ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
                TensorMemoryLayout::HEIGHT_SHARDED)},
        &device);
}

// The kernel on BRISC of kCore, writing to `result`, with CB 0 globally allocated: backed by `cb_backing`, or by a
// buffer bound later when `cb_backing` is null.
ProgramDescriptor make_descriptor(const distributed::MeshBuffer& result, Buffer* cb_backing) {
    const CoreRangeSet cores{CoreRange(kCore)};
    KernelDescriptor kernel;
    kernel.kernel_source = kKernel;
    kernel.core_ranges = cores;
    kernel.common_runtime_args = {static_cast<uint32_t>(result.address())};
    kernel.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

    CBDescriptor cb;
    cb.total_size = kPageSize;
    cb.core_ranges = cores;
    cb.format_descriptors = {
        CBFormatDescriptor{.buffer_index = 0, .data_format = tt::DataFormat::Float16_b, .page_size = kPageSize}};
    cb.globally_allocated = true;
    cb.buffer = cb_backing;

    ProgramDescriptor descriptor;
    descriptor.kernels.push_back(std::move(kernel));
    descriptor.cbs.push_back(std::move(cb));
    return descriptor;
}

distributed::MeshWorkload make_workload(distributed::MeshDevice& device, const ProgramDescriptor& descriptor) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), Program(descriptor));
    return workload;
}

uint32_t read_result(distributed::MeshDevice& device, const std::shared_ptr<distributed::MeshBuffer>& result) {
    std::vector<uint32_t> readback;
    distributed::ReadShard(device.mesh_command_queue(), readback, result, kDevice);
    return readback.at(0);
}

template <typename F>
void expect_throw_containing(F&& call, const std::string& text) {
    try {
        call();
        ADD_FAILURE() << "expected a throw containing \"" << text << "\"";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
    }
}

}  // namespace

// The program is prepared (compiled and laid out) before the CB has a buffer; applying a descriptor that names the
// buffer binds it, and the kernel sees the bound buffer's address.
TEST_F(AnyDispatchMeshDeviceFixture, TensixCircularBufferBoundLaterRunsOnceBound) {
    for (const auto& mesh_device : this->devices_) {
        auto& device = *mesh_device;
        auto result = allocate_page(device);
        auto cb_backing = allocate_page(device);

        auto workload = make_workload(device, make_descriptor(*result, /*cb_backing=*/nullptr));
        experimental::program_preparation::prepare(device, workload);

        auto& program = workload.get_programs().begin()->second;
        apply_descriptor_runtime_args(program, make_descriptor(*result, cb_backing->get_reference_buffer()));
        distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, /*blocking=*/true);
        EXPECT_EQ(read_result(device, result), cb_backing->address());
    }
}

// Launching writes the CB's address to the device, so it fails while the buffer is unbound. The failure leaves the
// workload usable: once the buffer is bound, it launches.
TEST_F(AnyDispatchMeshDeviceFixture, TensixCircularBufferBoundLaterDoesNotLaunchWhileUnbound) {
    for (const auto& mesh_device : this->devices_) {
        auto& device = *mesh_device;
        auto result = allocate_page(device);
        auto cb_backing = allocate_page(device);

        auto workload = make_workload(device, make_descriptor(*result, /*cb_backing=*/nullptr));
        expect_throw_containing(
            [&] { distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, /*blocking=*/true); },
            "not bound");

        auto& program = workload.get_programs().begin()->second;
        apply_descriptor_runtime_args(program, make_descriptor(*result, cb_backing->get_reference_buffer()));
        distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, /*blocking=*/true);
        EXPECT_EQ(read_result(device, result), cb_backing->address());
    }
}

// Leaving the buffer to be bound later relaxes "exactly one backing" to "at most one", not further.
TEST_F(AnyDispatchMeshDeviceFixture, TensixCircularBufferBoundLaterRejectsTwoBackings) {
    for (const auto& mesh_device : this->devices_) {
        auto& device = *mesh_device;
        auto result = allocate_page(device);
        auto cb_backing = allocate_page(device);

        ProgramDescriptor two_backings = make_descriptor(*result, cb_backing->get_reference_buffer());
        // Rejected before either pointer is read, so the tensor needs no real object behind it.
        two_backings.cbs[0].tensor = reinterpret_cast<const MeshTensor*>(cb_backing.get());
        expect_throw_containing([&] { Program program(two_backings); }, "2 are set");
    }
}

}  // namespace tt::tt_metal
