// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
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
    const experimental::metal2_host_api::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

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

    constexpr const char* DM_READER = "dm_reader";
    constexpr const char* DM_TRANSFORM = "dm_transform";
    constexpr const char* DM_WRITER = "dm_writer";

    experimental::metal2_host_api::SemaphoreSpec sem0_spec{
        .unique_id = "sem0",
        .target_nodes = node,
    };
    experimental::metal2_host_api::SemaphoreSpec sem1_spec{
        .unique_id = "sem1",
        .target_nodes = node,
    };

    experimental::metal2_host_api::KernelSpec dm_reader_spec{
        .unique_id = DM_READER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = "sem0", .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_transform_spec{
        .unique_id = DM_TRANSFORM,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}, {"INCOMING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = "sem0", .accessor_name = "sem_in"},
                {.semaphore_spec_name = "sem1", .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_writer_spec{
        .unique_id = DM_WRITER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = "sem1", .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_READER, DM_TRANSFORM, DM_WRITER},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .name = "multi_semaphore_pipeline",
        .kernels = {dm_reader_spec, dm_transform_spec, dm_writer_spec},
        .semaphores = {sem0_spec, sem1_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel_spec_name = DM_READER,
         .runtime_arg_values =
             {{.node = node,
               .args =
                   {{"dram_addr", dram_src_addr},
                    {"l1_addr", buf_a_addr},
                    {"num_elements", num_elements},
                    {"dram_bank_id", 0u}}}}},
        {.kernel_spec_name = DM_TRANSFORM},
        {.kernel_spec_name = DM_WRITER,
         .runtime_arg_values =
             {{.node = node,
               .args =
                   {{"dram_addr", dram_dst_addr},
                    {"l1_addr", buf_b_addr},
                    {"num_elements", num_elements},
                    {"dram_bank_id", 0u}}}}},
    };
    experimental::metal2_host_api::SetProgramRunArgs(program, params);

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

    const experimental::metal2_host_api::NodeCoord node_0{0, 0};
    const experimental::metal2_host_api::NodeCoord node_1{1, 0};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    constexpr uint32_t buf_a_addr = 1000 * 1024;
    constexpr uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_mid_addr = 31000 * 1024;
    const uint32_t dram_dst_addr = 32000 * 1024;

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], node_0, buf_a_addr, initial_data);

    const CoreCoord core_1_virtual = mesh_device->worker_core_from_logical_core(node_1);

    experimental::metal2_host_api::SemaphoreSpec sem_core_0_spec{
        .unique_id = "sem_core_0",
        .target_nodes = node_0,
    };
    experimental::metal2_host_api::SemaphoreSpec sem_cross_spec{
        .unique_id = "sem_cross",
        .target_nodes = experimental::metal2_host_api::NodeRange{node_0, node_1},
    };
    experimental::metal2_host_api::SemaphoreSpec sem0_core_1_spec{
        .unique_id = "sem0_core_1",
        .target_nodes = node_1,
    };
    experimental::metal2_host_api::SemaphoreSpec sem1_core_1_spec{
        .unique_id = "sem1_core_1",
        .target_nodes = node_1,
    };

    constexpr const char* DM_TRANSFORM_0 = "dm_transform_0";
    constexpr const char* DM_WRITER_0 = "dm_writer_0";
    constexpr const char* DM_READER_1 = "dm_reader_1";
    constexpr const char* DM_TRANSFORM_1 = "dm_transform_1";
    constexpr const char* DM_WRITER_1 = "dm_writer_1";

    experimental::metal2_host_api::KernelSpec dm_transform_0_spec{
        .unique_id = DM_TRANSFORM_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}}},
        .semaphore_bindings = {{.semaphore_spec_name = "sem_core_0", .accessor_name = "sem_out"}},
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_writer_0_spec{
        .unique_id = DM_WRITER_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCREMENT_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = "sem_core_0", .accessor_name = "sem"},
                {.semaphore_spec_name = "sem_cross", .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"dram_addr", "l1_addr", "num_elements", "dram_bank_id", "remote_noc_x", "remote_noc_y"},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_reader_1_spec{
        .unique_id = DM_READER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = "sem0_core_1", .accessor_name = "sem"},
                {.semaphore_spec_name = "sem_cross", .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_transform_1_spec{
        .unique_id = DM_TRANSFORM_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCOMING_SEM", "1"}, {"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = "sem0_core_1", .accessor_name = "sem_in"},
                {.semaphore_spec_name = "sem1_core_1", .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec dm_writer_1_spec{
        .unique_id = DM_WRITER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = "sem1_core_1", .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .config =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec wu_core_0{
        .name = "wu_core_0",
        .kernels = {DM_TRANSFORM_0, DM_WRITER_0},
        .target_nodes = node_0,
    };
    experimental::metal2_host_api::WorkUnitSpec wu_core_1{
        .name = "wu_core_1",
        .kernels = {DM_READER_1, DM_TRANSFORM_1, DM_WRITER_1},
        .target_nodes = node_1,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .name = "multi_cluster_multi_semaphore_pipeline",
        .kernels = {dm_transform_0_spec, dm_writer_0_spec, dm_reader_1_spec, dm_transform_1_spec, dm_writer_1_spec},
        .semaphores = {sem_core_0_spec, sem_cross_spec, sem0_core_1_spec, sem1_core_1_spec},
        .work_units = {wu_core_0, wu_core_1},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel_spec_name = DM_TRANSFORM_0},
        {.kernel_spec_name = DM_TRANSFORM_1},
        {.kernel_spec_name = DM_WRITER_0,
         .runtime_arg_values =
             {{.node = node_0,
               .args =
                   {{"dram_addr", dram_mid_addr},
                    {"l1_addr", buf_b_addr},
                    {"num_elements", num_elements},
                    {"dram_bank_id", 0u},
                    {"remote_noc_x", static_cast<uint32_t>(core_1_virtual.x)},
                    {"remote_noc_y", static_cast<uint32_t>(core_1_virtual.y)}}}}},
        {.kernel_spec_name = DM_READER_1,
         .runtime_arg_values =
             {{.node = node_1,
               .args =
                   {{"dram_addr", dram_mid_addr},
                    {"l1_addr", buf_a_addr},
                    {"num_elements", num_elements},
                    {"dram_bank_id", 0u}}}}},
        {.kernel_spec_name = DM_WRITER_1,
         .runtime_arg_values =
             {{.node = node_1,
               .args =
                   {{"dram_addr", dram_dst_addr},
                    {"l1_addr", buf_b_addr},
                    {"num_elements", num_elements},
                    {"dram_bank_id", 0u}}}}},
    };
    experimental::metal2_host_api::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    ASSERT_EQ(actual_data, expected_data);
}
