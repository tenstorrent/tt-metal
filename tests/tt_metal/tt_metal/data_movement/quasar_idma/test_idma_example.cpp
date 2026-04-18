// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_idma {

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

constexpr auto kIdmaBasic = "tests/tt_metal/tt_metal/data_movement/quasar_idma/kernels/idma_basic_example.cpp";
constexpr auto kIdma1DStrided = "tests/tt_metal/tt_metal/data_movement/quasar_idma/kernels/idma_1d_strided_example.cpp";

static void run_kernel(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::string& kernel_path,
    CoreCoord core,
    std::vector<uint32_t> compile_args = {}) {
    Program program = CreateProgram();
    experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = std::move(compile_args)});
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);
}

// Basic: 16 elements * 8 B = 128 B linear copy from src to dst
bool run_idma_basic_test(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t src_base = 0x10000;
    constexpr uint32_t dst_base = 0x20000;
    constexpr uint32_t num_elements = 16;
    constexpr uint32_t elem_size = 8;
    constexpr uint32_t total_bytes = num_elements * elem_size;
    constexpr uint32_t num_words = total_bytes / sizeof(uint32_t);

    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> src_data(num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        src_data[i] = 0xA0000000 + i;
    }
    tt_metal::detail::WriteToDeviceL1(device, core, src_base, src_data);

    run_kernel(mesh_device, kIdmaBasic, core, {src_base, dst_base});

    std::vector<uint32_t> dst_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, dst_base, total_bytes, dst_data);

    bool pass = (dst_data == src_data);
    if (!pass) {
        for (uint32_t i = 0; i < num_words; i++) {
            if (dst_data[i] != src_data[i]) {
                log_error(
                    tt::LogTest,
                    "Basic mismatch word {}: expected 0x{:08x}, got 0x{:08x}",
                    i,
                    src_data[i],
                    dst_data[i]);
            }
        }
    }
    return pass;
}

// 1D strided: 10 elements, src_stride=16 B, dst linear.
// dst[i] = src[i * src_stride] (every other 8 B element from src)
bool run_idma_1d_strided_test(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t src_base = 0x10000;
    constexpr uint32_t dst_base = 0x20000;
    constexpr uint32_t num_elements = 10;
    constexpr uint32_t elem_size = 8;
    constexpr uint32_t src_stride = 2 * elem_size;  // 16 B

    // src region covers num_elements * src_stride = 160 B = 40 words
    constexpr uint32_t src_num_words = (num_elements * src_stride) / sizeof(uint32_t);
    constexpr uint32_t dst_num_words = (num_elements * elem_size) / sizeof(uint32_t);

    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> src_data(src_num_words);
    for (uint32_t i = 0; i < src_num_words; i++) {
        src_data[i] = 0xB0000000 + i;
    }
    tt_metal::detail::WriteToDeviceL1(device, core, src_base, src_data);

    // Build expected: for each element i, copy elem_size bytes from src_base + i*src_stride
    constexpr uint32_t words_per_elem = elem_size / sizeof(uint32_t);     // 2
    constexpr uint32_t src_stride_words = src_stride / sizeof(uint32_t);  // 4
    std::vector<uint32_t> expected(dst_num_words);
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t w = 0; w < words_per_elem; w++) {
            expected[i * words_per_elem + w] = src_data[i * src_stride_words + w];
        }
    }

    run_kernel(mesh_device, kIdma1DStrided, core, {src_base, dst_base});

    std::vector<uint32_t> dst_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, dst_base, num_elements * elem_size, dst_data);

    bool pass = (dst_data == expected);
    if (!pass) {
        for (uint32_t i = 0; i < dst_num_words; i++) {
            if (dst_data[i] != expected[i]) {
                log_error(
                    tt::LogTest,
                    "1D strided mismatch word {}: expected 0x{:08x}, got 0x{:08x}",
                    i,
                    expected[i],
                    dst_data[i]);
            }
        }
    }
    return pass;
}

}  // namespace unit_tests::dm::quasar_idma

// =============================================================================
// Test Suite: Quasar IDMA
// =============================================================================

class QuasarIdmaOps : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarIdmaOps, IDMA_Basic) {
    if (unit_tests::dm::quasar_idma::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Host writes pattern to src, kernel copies 16*8=128 B to dst, host verifies dst==src
    EXPECT_TRUE(unit_tests::dm::quasar_idma::run_idma_basic_test(devices_[0]));
}

TEST_F(QuasarIdmaOps, IDMA_1D_Strided) {
    if (unit_tests::dm::quasar_idma::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Host writes pattern to src, kernel copies 10 elements with src_stride=16 B to dst linearly
    EXPECT_TRUE(unit_tests::dm::quasar_idma::run_idma_1d_strided_test(devices_[0]));
}

}  // namespace tt::tt_metal
