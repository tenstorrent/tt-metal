// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <cstdlib>
#include <iostream>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

void warn_if_dprint_not_enabled() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see kernel output."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, DmHelloWorld) {
    warn_if_dprint_not_enabled();

    auto mesh_device = devices_[0];
    const CoreRange cluster_range = CoreRange(CoreCoord{0, 0}, CoreCoord{1, 0});

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_hello_world.cpp",
        cluster_range,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
}

TEST_F(MeshDeviceSingleCardFixture, DmPacketWalkCorrectnessAdjacentCore) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];

    constexpr CoreCoord src_core = {0, 0};
    constexpr CoreCoord dst_core = {0, 1};

    constexpr uint32_t packet_size_bytes = 3 * 64;
    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);
    constexpr uint32_t num_iterations = 4;
    constexpr uint32_t stride_bytes = packet_size_bytes;
    constexpr uint32_t total_words = packet_words * num_iterations;
    constexpr uint32_t total_bytes = total_words * sizeof(uint32_t);

    // Keep both buffers in a high, non-overlapping region of L1.
    constexpr uint32_t src_l1_address = 1000 * 1024;
    constexpr uint32_t dst_l1_address = 1200 * 1024;

    std::vector<uint32_t> src_init(total_words, 0);
    std::vector<uint32_t> dst_init(total_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, src_core, src_l1_address, src_init);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, dst_l1_address, dst_init);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    const CoreCoord physical_dst_core = device->worker_core_from_logical_core(dst_core);
    const uint32_t packed_physical_dst_core = (physical_dst_core.x << 16) | (physical_dst_core.y & 0xFFFF);

    std::vector<uint32_t> compile_args = {
        src_l1_address, dst_l1_address, num_iterations, packet_size_bytes, stride_bytes, packed_physical_dst_core};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_packet_walk.cpp",
        src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = compile_args,
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> observed(total_words, 0);
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, dst_l1_address, total_bytes, observed);

    std::vector<uint32_t> expected(total_words, 0);
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        for (uint32_t word = 0; word < packet_words; ++word) {
            expected[iter * packet_words + word] = (iter << 16) | word;
        }
    }

    ASSERT_EQ(observed, expected);
}
