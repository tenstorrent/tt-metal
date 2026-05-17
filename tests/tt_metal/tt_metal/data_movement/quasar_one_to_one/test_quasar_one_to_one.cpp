// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar one-to-one DM perf scaffold. The kernel currently just DPRINTs
// "Hello, World!"; perf scenarios will be added on top.

#include <cstdint>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_one_to_one {

constexpr auto kHelloWorldKernel = "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/hello_world.cpp";
constexpr auto kAttWritePerfKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/att_write_perf.cpp";

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

bool run_hello_world(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr const char* DM_KERNEL = "hello_world";
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kHelloWorldKernel},
        .num_threads = 1,
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "quasar_one_to_one_hello",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = DM_KERNEL,
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}

bool run_att_write_perf(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr const char* DM_KERNEL = "att_write_perf";
    constexpr uint32_t kSrcAddr = 0x10000;
    constexpr uint32_t kDstAddr = 0x20000;
    constexpr uint32_t kMasterX = 0;
    constexpr uint32_t kMasterY = 0;
    constexpr uint32_t kDstX = 1;
    constexpr uint32_t kDstY = 0;
    constexpr uint32_t kPayloadBytes = 16;
    constexpr uint32_t kNumIters = 1000;
    const experimental::metal2_host_api::NodeCoord node{kMasterX, kMasterY};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kAttWritePerfKernel},
        .num_threads = 6,
        .compile_time_arg_bindings =
            {{"src_addr", kSrcAddr},
             {"dst_addr", kDstAddr},
             {"dst_x", kDstX},
             {"dst_y", kDstY},
             {"master_x", kMasterX},
             {"master_y", kMasterY},
             {"payload_bytes", kPayloadBytes},
             {"num_iters", kNumIters}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "quasar_one_to_one_att_write_perf",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = DM_KERNEL,
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}

}  // namespace unit_tests::dm::quasar_one_to_one

// =============================================================================
// Test Suite: Quasar One-to-One
// =============================================================================

class QuasarOneToOneOps : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarOneToOneOps, HelloWorld) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_hello_world(devices_[0]));
}

TEST_F(QuasarOneToOneOps, AttWritePerf) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Master core (1,1) issues posted NOC writes to (2,1) in two phases:
    //   Phase 1: ATT-translated address (endpoint id -> physical coord)
    //   Phase 2: physical (x,y) directly
    // Kernel DPRINTs per-issue cycles for each phase; compare to estimate the
    // DM stall introduced by ATT translation.
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_att_write_perf(devices_[0]));
}

}  // namespace tt::tt_metal
