// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "command_queue_fixture.hpp"
#include "env_lib.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

void build_and_run_program(
    std::shared_ptr<distributed::MeshDevice> device,
    bool slow_dispatch,
    uint32_t NUM_PROGRAMS,
    uint32_t MAX_LOOP,
    uint32_t page_size,
    bool mix_noc_mode) {
    // Make random
    auto random_seed = 0; // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    CoreCoord worker_grid_size = {1,1};
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_PROGRAMS);

    Program program1;
    Program program2;

    CircularBufferConfig cb_config =
        CircularBufferConfig(page_size, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, page_size);
    CreateCircularBuffer(program1, cr_set, cb_config);
    CreateCircularBuffer(program2, cr_set, cb_config);

    vector<uint32_t> compile_args = {MAX_LOOP, page_size};
    tt_metal::TensorAccessorArgs().append_to(compile_args);

    auto brisc_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto ncrisc_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto brisc_kernel2 = CreateKernel(
        program2,
        mix_noc_mode ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dedicated_noc_writer.cpp"
                     : "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto ncrisc_kernel2 = CreateKernel(
        program2,
        mix_noc_mode ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dedicated_noc_writer.cpp"
                     : "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    for (int core_idx_y = 0; core_idx_y < worker_grid_size.y; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < worker_grid_size.x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            CoreCoord neighbour_core = {core_idx_x == worker_grid_size.x - 1 ? 0 : core_idx_x + 1, core_idx_y};
            CoreCoord neighbour_core_physical = device->worker_core_from_logical_core(neighbour_core);
            // mcast
            auto device_grid = device->compute_with_storage_grid_size();
            CoreCoord top_left_core = {0, 0};
            CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
            CoreCoord bottom_right_core = {device_grid.x - 1, device_grid.y - 1};
            CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
            std::vector<uint32_t> rt_args = {
                (std::uint32_t)neighbour_core_physical.x,
                (std::uint32_t)neighbour_core_physical.y,
                // mcast
                core_idx_x == 0 && core_idx_y == 0,
                top_left_core_physical.x,
                top_left_core_physical.y,
                bottom_right_core_physical.x,
                bottom_right_core_physical.y,
                device_grid.x * device_grid.y};
            tt::tt_metal::SetRuntimeArgs(program1, brisc_kernel1, core, rt_args);
            tt::tt_metal::SetRuntimeArgs(program1, ncrisc_kernel1, core, rt_args);
            tt::tt_metal::SetRuntimeArgs(program2, brisc_kernel2, core, rt_args);
            tt::tt_metal::SetRuntimeArgs(program2, ncrisc_kernel2, core, rt_args);
        }
    }
    distributed::MeshWorkload workload1;
    distributed::MeshWorkload workload2;
    distributed::AddProgramToMeshWorkload(
        workload1, std::move(program1), distributed::MeshCoordinateRange(device->shape()));
    distributed::AddProgramToMeshWorkload(
        workload2, std::move(program2), distributed::MeshCoordinateRange(device->shape()));

    // This loop caches program1 and runs
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        log_info(tt::LogTest, "Running program1 {} of {}", i + 1, NUM_PROGRAMS);
        if (i % 2 == 0) {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload1, false);
        } else {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload2, false);
        }
    }
    distributed::Finish(device->mesh_command_queue());
}

TEST_F(UnitMeshCQFixture, TestDynamicNoCOneProgram) {
    uint32_t NUM_PROGRAMS = 1;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(UnitMeshCQFixture, TestDynamicNoCMutlipleProgram) {
    uint32_t NUM_PROGRAMS = 3;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(UnitMeshCQFixture, TestDynamicNoCMutlipleProgramMixedMode) {
    uint32_t NUM_PROGRAMS = 5;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = true;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}
}  // namespace tt::tt_metal
