// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

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

constexpr auto kIdmaBasic =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_idma/kernels/idma_basic_example.cpp";
constexpr auto kIdma1DStrided =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_idma/kernels/idma_1d_strided_example.cpp";
constexpr auto kNocVsIdmaCopyBench =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_idma/kernels/noc_vs_idma_copy_bench.cpp";
constexpr uint32_t kNocVsIdmaSweepSizes = 11;
constexpr uint32_t kNocVsIdmaConfigs = 2;
constexpr uint32_t kNocVsIdmaRepeats = 10;
constexpr uint32_t kNocVsIdmaResultWordsPerSample = 4;
constexpr uint32_t kNocVsIdmaResultSamples = kNocVsIdmaSweepSizes * kNocVsIdmaConfigs;
constexpr uint32_t kNocVsIdmaResultBytes = kNocVsIdmaResultSamples * kNocVsIdmaResultWordsPerSample * sizeof(uint32_t);

void run_kernel(
    distributed::MeshDevice& mesh_device,
    const std::string& kernel_path,
    experimental::KernelSpec::CompileTimeArgs compile_time_args) {
    const experimental::KernelSpecName DM_KERNEL{"idma"};
    const experimental::NodeCoord node{0, 0};

    // DataMovementGen2Config is Quasar-only -- ValidateProgramSpec hard-fails on Gen1 (WH/BH) with
    // "targets Gen1 but its DataMovementHardwareConfig holds a DataMovementGen2Config". Select per arch so
    // arch-generic kernels (e.g. clock_calibration_example.cpp) can run on both.
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    experimental::DataMovementHardwareConfig dm_hw_config = experimental::DataMovementGen2Config{};
    if (arch != tt::ARCH::QUASAR) {
        dm_hw_config = experimental::CreateReaderGen1DataMovementConfig();
    }

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = kernel_path,
        .num_threads = 1,
        .compile_time_args = std::move(compile_time_args),
        .hw_config = dm_hw_config,
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "idma",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_KERNEL}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);
}

// Allocates a pair of same-sized replicated L1 buffers on the mesh for use as IDMA src/dst.
// Buffers are returned (not just their addresses) so the caller keeps them alive for the
// lifetime of the test; letting the MeshBuffers be destroyed would free the allocation.
struct IdmaSrcDstBuffers {
    std::shared_ptr<distributed::MeshBuffer> src;
    std::shared_ptr<distributed::MeshBuffer> dst;
};

IdmaSrcDstBuffers make_idma_src_dst_buffers(distributed::MeshDevice& mesh_device, uint32_t total_bytes) {
    distributed::DeviceLocalBufferConfig local_buffer_config = {
        .page_size = total_bytes, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config = {.size = total_bytes};
    return {
        .src = distributed::MeshBuffer::create(buffer_config, local_buffer_config, &mesh_device),
        .dst = distributed::MeshBuffer::create(buffer_config, local_buffer_config, &mesh_device),
    };
}

// Basic: 16 elements * 8 B = 128 B linear copy from src to dst
bool run_idma_basic_test(distributed::MeshDevice& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t num_elements = 16;
    constexpr uint32_t elem_size = 8;
    constexpr uint32_t total_bytes = num_elements * elem_size;
    constexpr uint32_t num_words = total_bytes / sizeof(uint32_t);

    auto buffers = make_idma_src_dst_buffers(mesh_device, total_bytes);
    const uint32_t src_base = buffers.src->address();
    const uint32_t dst_base = buffers.dst->address();

    std::vector<uint32_t> src_data(num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        src_data[i] = 0xA0000000 + i;
    }
    slow_dispatch::WriteToL1(mesh_device, core, src_base, src_data);

    run_kernel(mesh_device, kIdmaBasic, {{"src_addr", src_base}, {"dst_addr", dst_base}});

    std::vector<uint32_t> dst_data;
    slow_dispatch::ReadFromL1(mesh_device, core, dst_base, total_bytes, dst_data);

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
bool run_idma_1d_strided_test(distributed::MeshDevice& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t num_elements = 10;
    constexpr uint32_t elem_size = 8;
    constexpr uint32_t src_stride = 2 * elem_size;  // 16 B
    constexpr uint32_t total_bytes = num_elements * src_stride;
    auto buffers = make_idma_src_dst_buffers(mesh_device, total_bytes);
    const uint32_t src_base = buffers.src->address();
    const uint32_t dst_base = buffers.dst->address();

    // src region covers num_elements * src_stride = 160 B = 40 words
    constexpr uint32_t src_num_words = (num_elements * src_stride) / sizeof(uint32_t);
    constexpr uint32_t dst_num_words = (num_elements * elem_size) / sizeof(uint32_t);

    std::vector<uint32_t> src_data(src_num_words);
    for (uint32_t i = 0; i < src_num_words; i++) {
        src_data[i] = 0xB0000000 + i;
    }
    slow_dispatch::WriteToL1(mesh_device, core, src_base, src_data);

    // Build expected: for each element i, copy elem_size bytes from src_base + i*src_stride
    constexpr uint32_t words_per_elem = elem_size / sizeof(uint32_t);     // 2
    constexpr uint32_t src_stride_words = src_stride / sizeof(uint32_t);  // 4
    std::vector<uint32_t> expected(dst_num_words);
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t w = 0; w < words_per_elem; w++) {
            expected[i * words_per_elem + w] = src_data[i * src_stride_words + w];
        }
    }

    run_kernel(mesh_device, kIdma1DStrided, {{"src_addr", src_base}, {"dst_addr", dst_base}});

    std::vector<uint32_t> dst_data;
    slow_dispatch::ReadFromL1(mesh_device, core, dst_base, num_elements * elem_size, dst_data);

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

const char* noc_vs_idma_benchmark_name(uint32_t config_id) {
    return config_id == 0 ? "iDMA xfer benchmark" : "NoC xfer benchmark";
}

// NoC vs iDMA local TL1 copy benchmark. Averaged kernel rows are written into an L1 result buffer and logged
// by the host; the destination check only validates the final largest sample's copied contents.
bool run_noc_vs_idma_copy_bench(distributed::MeshDevice& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t kLargestSweepSize = 131072;  // must match kSweepSizes.back() in the kernel
    constexpr uint32_t num_words = kLargestSweepSize / sizeof(uint32_t);

    IDevice* device = mesh_device.get_devices()[0];
    auto buffers = make_idma_src_dst_buffers(mesh_device, kLargestSweepSize);
    distributed::DeviceLocalBufferConfig local_buffer_config = {
        .page_size = kNocVsIdmaResultBytes, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config = {.size = kNocVsIdmaResultBytes};
    auto result_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, &mesh_device);
    const uint32_t src_base = buffers.src->address();
    const uint32_t dst_base = buffers.dst->address();
    const uint32_t result_base = result_buffer->address();

    std::vector<uint32_t> src_data(num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        src_data[i] = 0xC0000000 + i;
    }
    tt_metal::detail::WriteToDeviceL1(device, core, src_base, src_data);
    std::vector<uint32_t> zero_results(kNocVsIdmaResultBytes / sizeof(uint32_t), 0);
    tt_metal::detail::WriteToDeviceL1(device, core, result_base, zero_results);

    // A wrong loopback coordinate waits for acks instead of producing a data mismatch, so use the
    // host's physical mapping instead of deriving it from NOC_NODE_ID in the kernel.
    const CoreCoord phys_core = device->virtual_core_from_logical_core(core, CoreType::WORKER);

    run_kernel(
        mesh_device,
        kNocVsIdmaCopyBench,
        {{"src_addr", src_base},
         {"dst_addr", dst_base},
         {"result_addr", result_base},
         {"noc_x", static_cast<uint32_t>(phys_core.x)},
         {"noc_y", static_cast<uint32_t>(phys_core.y)}});

    std::vector<uint32_t> result_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, result_base, kNocVsIdmaResultBytes, result_data);
    for (uint32_t sample = 0; sample < kNocVsIdmaResultSamples; ++sample) {
        const uint32_t base = sample * kNocVsIdmaResultWordsPerSample;
        const uint32_t config_id = result_data[base + 0];
        const uint32_t bytes = result_data[base + 1];
        const uint32_t issue_cycles = result_data[base + 2];
        const uint32_t wait_cycles = result_data[base + 3];
        const uint32_t total_cycles = issue_cycles + wait_cycles;
        const double bytes_per_cycle = total_cycles == 0 ? 0.0 : static_cast<double>(bytes) / total_cycles;
        log_info(
            tt::LogTest,
            "{}: bytes={} avg_issue_cycles={} avg_wait_cycles={} bytes_per_cycle={} repeats={}",
            noc_vs_idma_benchmark_name(config_id),
            bytes,
            issue_cycles,
            wait_cycles,
            bytes_per_cycle,
            kNocVsIdmaRepeats);
    }

    // The final sample leaves dst containing the largest copy, which is this benchmark's only host check.
    std::vector<uint32_t> dst_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, dst_base, kLargestSweepSize, dst_data);
    if (dst_data == src_data) {
        return true;
    }
    for (uint32_t i = 0; i < num_words; i++) {
        if (dst_data[i] != src_data[i]) {
            log_error(
                tt::LogTest,
                "noc_vs_idma: first mismatch at word {} (byte {}): got {:#x} expected {:#x}",
                i,
                i * sizeof(uint32_t),
                dst_data[i],
                src_data[i]);
            break;
        }
    }
    return false;
}

}  // namespace unit_tests::dm::quasar_idma

// =============================================================================
// Test Suite: Quasar IDMA
// =============================================================================

class QuasarIdmaOps : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarIdmaOps, IDMA_Basic) {
    if (unit_tests::dm::quasar_idma::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Host writes pattern to src, kernel copies 16*8=128 B to dst, host verifies dst==src
    EXPECT_TRUE(unit_tests::dm::quasar_idma::run_idma_basic_test(this->device()));
}

TEST_F(QuasarIdmaOps, IDMA_1D_Strided) {
    if (unit_tests::dm::quasar_idma::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Host writes pattern to src, kernel copies 10 elements with src_stride=16 B to dst linearly
    EXPECT_TRUE(unit_tests::dm::quasar_idma::run_idma_1d_strided_test(this->device()));
}

TEST_F(QuasarIdmaOps, NocVsIdmaCopyBench) {
    if (unit_tests::dm::quasar_idma::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Measurement rows are read from L1 and logged by the host; the assertion checks final destination contents.
    EXPECT_TRUE(unit_tests::dm::quasar_idma::run_noc_vs_idma_copy_bench(this->device()));
}

}  // namespace tt::tt_metal
