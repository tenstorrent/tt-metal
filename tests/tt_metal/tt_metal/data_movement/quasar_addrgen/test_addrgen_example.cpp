// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Example tests for the Quasar hardware address generator API.

#include <cstdint>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/host_api.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_addrgen {

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

bool run_addrgen_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::string& kernel_path,
    uint32_t src_stride_en,
    uint32_t dst_stride_en,
    uint32_t num_of_addresses) {
    constexpr CoreCoord core = {0, 0};

    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {src_stride_en, dst_stride_en, num_of_addresses}});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}

constexpr auto k1D = "tests/tt_metal/tt_metal/data_movement/quasar_addrgen/kernels/addrgen_1d_example.cpp";
constexpr auto k2D = "tests/tt_metal/tt_metal/data_movement/quasar_addrgen/kernels/addrgen_2d_example.cpp";
constexpr auto kFace = "tests/tt_metal/tt_metal/data_movement/quasar_addrgen/kernels/addrgen_face_example.cpp";

}  // namespace unit_tests::dm::quasar_addrgen

// =============================================================================
// Test Suite: Quasar Address Generator
// =============================================================================

class QuasarAddrgenOps : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarAddrgenOps, Strided1D_SrcOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k1D, 1, 0, 10));
}

TEST_F(QuasarAddrgenOps, Strided1D_DstOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k1D, 0, 1, 10));
}

TEST_F(QuasarAddrgenOps, Strided1D_Both) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k1D, 1, 1, 10));
}

TEST_F(QuasarAddrgenOps, Strided2D_SrcOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k2D, 1, 0, 16));
}

TEST_F(QuasarAddrgenOps, Strided2D_DstOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k2D, 0, 1, 16));
}

TEST_F(QuasarAddrgenOps, Strided2D_Both) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::k2D, 1, 1, 16));
}

TEST_F(QuasarAddrgenOps, Face_SrcOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::kFace, 1, 0, 32));
}

TEST_F(QuasarAddrgenOps, Face_DstOnly) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::kFace, 0, 1, 32));
}

TEST_F(QuasarAddrgenOps, Face_Both) {
    if (unit_tests::dm::quasar_addrgen::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_addrgen::run_addrgen_test(devices_[0], unit_tests::dm::quasar_addrgen::kFace, 1, 1, 32));
}

}  // namespace tt::tt_metal
