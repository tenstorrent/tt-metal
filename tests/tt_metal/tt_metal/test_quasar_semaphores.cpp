// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hal.hpp"
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    const uint32_t buf_a_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_src_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_dst_addr = dram_src_addr + (1000 * 1024);

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_src_addr, initial_data);

    const experimental::KernelSpecName DM_READER{"dm_reader"};
    const experimental::KernelSpecName DM_TRANSFORM{"dm_transform"};
    const experimental::KernelSpecName DM_WRITER{"dm_writer"};

    experimental::SemaphoreSpec sem0_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem0"},
        .target_nodes = node,
    };
    experimental::SemaphoreSpec sem1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem1"},
        .target_nodes = node,
    };

    experimental::KernelSpec dm_reader_spec{
        .unique_id = DM_READER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_transform_spec{
        .unique_id = DM_TRANSFORM,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}, {"INCOMING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_spec{
        .unique_id = DM_WRITER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_READER, DM_TRANSFORM, DM_WRITER},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "multi_semaphore_pipeline",
        .kernels = {dm_reader_spec, dm_transform_spec, dm_writer_spec},
        .semaphores = {sem0_spec, sem1_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER,
            .runtime_arg_values =
                {{node,
                  {{"dram_addr", dram_src_addr},
                   {"l1_addr", buf_a_addr},
                   {"num_elements", num_elements},
                   {"dram_bank_id", 0u}}}}},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dram_addr", dram_dst_addr},
                   {"l1_addr", buf_b_addr},
                   {"num_elements", num_elements},
                   {"dram_bank_id", 0u}}}}},
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ASSERT_EQ(actual_data, expected_data);
}

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarMultipleClustersMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    if (mesh_device->compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }

    const experimental::NodeCoord node_0{0, 0};
    const experimental::NodeCoord node_1{1, 0};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    const uint32_t buf_a_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_mid_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_dst_addr = dram_mid_addr + (1000 * 1024);

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], node_0, buf_a_addr, initial_data);

    const CoreCoord core_1_virtual = mesh_device->worker_core_from_logical_core(node_1);

    experimental::SemaphoreSpec sem_core_0_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem_core_0"},
        .target_nodes = node_0,
    };
    experimental::SemaphoreSpec sem_cross_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem_cross"},
        .target_nodes = experimental::NodeRange{node_0, node_1},
    };
    experimental::SemaphoreSpec sem0_core_1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem0_core_1"},
        .target_nodes = node_1,
    };
    experimental::SemaphoreSpec sem1_core_1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem1_core_1"},
        .target_nodes = node_1,
    };

    const experimental::KernelSpecName DM_TRANSFORM_0{"dm_transform_0"};
    const experimental::KernelSpecName DM_WRITER_0{"dm_writer_0"};
    const experimental::KernelSpecName DM_READER_1{"dm_reader_1"};
    const experimental::KernelSpecName DM_TRANSFORM_1{"dm_transform_1"};
    const experimental::KernelSpecName DM_WRITER_1{"dm_writer_1"};

    experimental::KernelSpec dm_transform_0_spec{
        .unique_id = DM_TRANSFORM_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem_out"}},
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_0_spec{
        .unique_id = DM_WRITER_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCREMENT_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"dram_addr", "l1_addr", "num_elements", "dram_bank_id", "remote_noc_x", "remote_noc_y"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_reader_1_spec{
        .unique_id = DM_READER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_transform_1_spec{
        .unique_id = DM_TRANSFORM_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCOMING_SEM", "1"}, {"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_1_spec{
        .unique_id = DM_WRITER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec wu_core_0{
        .name = "wu_core_0",
        .kernels = {DM_TRANSFORM_0, DM_WRITER_0},
        .target_nodes = node_0,
    };
    experimental::WorkUnitSpec wu_core_1{
        .name = "wu_core_1",
        .kernels = {DM_READER_1, DM_TRANSFORM_1, DM_WRITER_1},
        .target_nodes = node_1,
    };

    experimental::ProgramSpec spec{
        .name = "multi_cluster_multi_semaphore_pipeline",
        .kernels = {dm_transform_0_spec, dm_writer_0_spec, dm_reader_1_spec, dm_transform_1_spec, dm_writer_1_spec},
        .semaphores = {sem_core_0_spec, sem_cross_spec, sem0_core_1_spec, sem1_core_1_spec},
        .work_units = {wu_core_0, wu_core_1},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_0},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_1},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_0,
            .runtime_arg_values =
                {{node_0,
                  {{"dram_addr", dram_mid_addr},
                   {"l1_addr", buf_b_addr},
                   {"num_elements", num_elements},
                   {"dram_bank_id", 0u},
                   {"remote_noc_x", static_cast<uint32_t>(core_1_virtual.x)},
                   {"remote_noc_y", static_cast<uint32_t>(core_1_virtual.y)}}}}},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER_1,
            .runtime_arg_values =
                {{node_1,
                  {{"dram_addr", dram_mid_addr},
                   {"l1_addr", buf_a_addr},
                   {"num_elements", num_elements},
                   {"dram_bank_id", 0u}}}}},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_1,
            .runtime_arg_values =
                {{node_1,
                  {{"dram_addr", dram_dst_addr},
                   {"l1_addr", buf_b_addr},
                   {"num_elements", num_elements},
                   {"dram_bank_id", 0u}}}}},
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    ASSERT_EQ(actual_data, expected_data);
}
