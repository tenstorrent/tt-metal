// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Example test for im2col address generation using the Quasar hardware address generator.

#include <cstdint>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_im2col {

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

bool run_im2col_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::string& kernel_path,
    uint32_t num_of_addresses) {
    constexpr const char* DM_KERNEL = "addrgen";
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kernel_path},
        .num_threads = 1,
        .compile_time_arg_bindings = {{"num_of_addresses", num_of_addresses}},
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
        .program_id = "im2col",
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

constexpr auto kIm2Col =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_im2col/kernels/addrgen_im2col_example.cpp";
constexpr auto kIm2ColDilation1 =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_im2col/kernels/addrgen_im2col_dilation1_example.cpp";

}  // namespace unit_tests::dm::quasar_im2col

// =============================================================================
// Test Suite: Quasar Im2Col Address Generator
// =============================================================================

class QuasarIm2ColOps : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarIm2ColOps, Im2Col_FullImage) {
    if (unit_tests::dm::quasar_im2col::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Image 6x7, kernel 3x3, dil_w=2: 3 vert x 3 horz x 9 patch = 81 total addresses
    EXPECT_TRUE(
        unit_tests::dm::quasar_im2col::run_im2col_test(devices_[0], unit_tests::dm::quasar_im2col::kIm2Col, 81));
}

TEST_F(QuasarIm2ColOps, Im2Col_Dilation1) {
    if (unit_tests::dm::quasar_im2col::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Image 4x5, kernel 3x3: matrix_size(20) * kH*kW(9) = 180 total addresses
    EXPECT_TRUE(unit_tests::dm::quasar_im2col::run_im2col_test(
        devices_[0], unit_tests::dm::quasar_im2col::kIm2ColDilation1, 180));
}

}  // namespace tt::tt_metal
