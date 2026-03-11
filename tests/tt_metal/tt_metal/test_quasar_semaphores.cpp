// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarComputeKernelSemaphores) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    constexpr uint32_t base_src_l1_address = 1000 * 1024;
    constexpr uint32_t base_dst_l1_address = 1025 * 1024;
    std::vector<uint32_t> expected_values{0x0123, 0x4567, 0x89AB, 0xCDEF};
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], core, base_src_l1_address, expected_values);

    uint32_t sem_id = CreateSemaphore(program, core, 0);

    const KernelHandle kernel_handle = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_l1_read_write.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 4, .compile_args = {sem_id, 0}});
    SetRuntimeArgs(program, kernel_handle, core, {base_src_l1_address, base_dst_l1_address});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(4, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core, base_dst_l1_address, 4 * sizeof(uint32_t), actual_values);

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarDmAndComputeKernelSemaphores) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const uint32_t sem_id = CreateSemaphore(program, core, 0);

    uint32_t l1_address = 1000 * 1024;
    std::vector<uint32_t> expected_values{0x0123, 0x4567, 0x89AB, 0xCDEF};
    uint32_t dram_address = 30000 * 1024;
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_address, expected_values);

    std::vector<KernelHandle> dm_dram_to_l1_kernels;
    std::vector<KernelHandle> dm_l1_to_dram_kernels;
    dm_dram_to_l1_kernels.reserve(2);
    dm_l1_to_dram_kernels.reserve(2);
    for (uint32_t i = 0; i < 2; i++) {
        dm_dram_to_l1_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = {sem_id}}));

        dm_l1_to_dram_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = {sem_id}}));
    }

    SetRuntimeArgs(program, dm_dram_to_l1_kernels[0], core, {dram_address, l1_address, 4 * sizeof(uint32_t), 0, 0});
    SetRuntimeArgs(program, dm_l1_to_dram_kernels[0], core, {dram_address, l1_address, 4 * sizeof(uint32_t), 0, 1});
    l1_address += 4 * sizeof(uint32_t);
    SetRuntimeArgs(program, dm_dram_to_l1_kernels[1], core, {dram_address, l1_address, 4 * sizeof(uint32_t), 0, 2});

    const KernelHandle kernel_handle = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_l1_read_write.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 4, .compile_args = {sem_id, 3}});
    const uint32_t base_src_l1_address = l1_address;
    const uint32_t base_dst_l1_address = l1_address + 4 * sizeof(uint32_t);
    SetRuntimeArgs(program, kernel_handle, core, {base_src_l1_address, base_dst_l1_address});

    l1_address += 4 * sizeof(uint32_t);
    SetRuntimeArgs(program, dm_l1_to_dram_kernels[1], core, {dram_address, l1_address, 4 * sizeof(uint32_t), 0, 7});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(4, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core, l1_address, 4 * sizeof(uint32_t), actual_values);

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const uint32_t sem0 = CreateSemaphore(program, core, 0);
    const uint32_t sem1 = CreateSemaphore(program, core, 0);

    constexpr uint32_t num_elements = 10;
    constexpr uint32_t buf_a_addr = 1000 * 1024;
    constexpr uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    constexpr uint32_t dram_src_addr = 29000 * 1024;
    constexpr uint32_t dram_dst_addr = 30000 * 1024;

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_src_addr, initial_data);

    auto dm_reader = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = {sem0}});
    SetRuntimeArgs(program, dm_reader, core, {dram_src_addr, buf_a_addr, num_elements, 0});

    auto compute = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/transform_pipeline.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {num_elements, sem0, sem1},
            .defines = {{"OUTGOING_SEM", "1"}, {"INCOMING_SEM", "1"}}});
    SetRuntimeArgs(program, compute, core, {buf_a_addr, buf_b_addr});

    auto dm_writer = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = {sem1}});
    SetRuntimeArgs(program, dm_writer, core, {dram_dst_addr, buf_b_addr, num_elements, 0});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ASSERT_EQ(actual_data, expected_data);
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarMultipleClustersMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(1,0) to see the "
                     "output of the kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=(0,0),(1,0)" << std::endl;
    }

    constexpr CoreCoord core_0 = {0, 0};
    constexpr CoreCoord core_1 = {1, 0};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const uint32_t sem_core_0 = CreateSemaphore(program, core_0, 0);
    const uint32_t sem_cross = CreateSemaphore(program, CoreRange(core_0, core_1), 0);
    const uint32_t sem_core_1 = CreateSemaphore(program, core_1, 0);

    // const uint32_t dram_bank_size = MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::UNRESERVED);
    constexpr uint32_t num_elements = 10;
    constexpr uint32_t buf_a_addr = 1000 * 1024;
    constexpr uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_mid_addr = 31000 * 1024;

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], core_0, buf_a_addr, initial_data);

    const CoreCoord core_1_virtual = mesh_device->worker_core_from_logical_core(core_1);

    auto compute_0 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/transform_pipeline.cpp",
        core_0,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {num_elements, sem_core_0},
            .defines = {{"OUTGOING_SEM", "1"}}});
    SetRuntimeArgs(program, compute_0, core_0, {buf_a_addr, buf_b_addr});

    auto dm_writer_0 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        core_0,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {sem_core_0, sem_cross},
            .defines = {{"INCREMENT_REMOTE_SEM", "1"}}});
    SetRuntimeArgs(
        program, dm_writer_0, core_0, {dram_mid_addr, buf_b_addr, num_elements, 0, core_1_virtual.x, core_1_virtual.y});

    // --- Core(1,0) kernels: wait for cross-core signal, DRAM -> L1 -> transform -> DRAM ---
    auto dm_reader_1 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        core_1,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {sem_core_1, sem_cross},
            .defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}});
    SetRuntimeArgs(program, dm_reader_1, core_1, {dram_mid_addr, buf_a_addr, num_elements, 0});

    auto compute_1 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/transform_pipeline.cpp",
        core_1,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {num_elements, sem_core_1},
            .defines = {{"INCOMING_SEM", "1"}}});
    SetRuntimeArgs(program, compute_1, core_1, {buf_a_addr, buf_b_addr});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core_1, buf_b_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    ASSERT_EQ(actual_data, expected_data);
}
