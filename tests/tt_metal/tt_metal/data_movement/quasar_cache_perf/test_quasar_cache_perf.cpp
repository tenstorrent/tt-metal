// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM-core cache-write performance test (test id 912).

#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <cstdint>

namespace tt::tt_metal {

using namespace std;
using namespace tt::test_utils;

namespace unit_tests::dm::quasar_cache_perf {

constexpr std::uint32_t QUASAR_CACHE_WRITE_TEST_ID = 912;
constexpr std::uint32_t BASE_ADDR = 100 * 1024;

bool should_skip_test() { return std::getenv("TT_METAL_SIMULATOR") == nullptr; }

// Runs one write pass of `size_bytes` via `write_path` (0=uncached,1=cached+flush)
// on a single DM core, then reads back and verifies the byte pattern landed.
bool run_cache_write(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, std::uint32_t size_bytes, std::uint32_t write_path) {
    IDevice* device = mesh_device->get_devices()[0];
    constexpr CoreCoord core = {0, 0};
    const experimental::NodeCoord node{0, 0};

    // Pre-fill destination with a sentinel so a no-op run fails the read-back.
    std::vector<std::uint32_t> init_data((size_bytes + 3) / 4, 0xA5A5A5A5);
    tt_metal::detail::WriteToDeviceL1(device, core, BASE_ADDR, init_data);

    const experimental::KernelSpecName DM_KERNEL{"cache_write_perf"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_cache_perf/kernels/cache_write_perf.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"base_addr", "size_bytes", "write_path", "test_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "cache_write_perf", .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"base_addr", BASE_ADDR},
             {"size_bytes", size_bytes},
             {"write_path", write_path},
             {"test_id", QUASAR_CACHE_WRITE_TEST_ID}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<std::uint32_t> out;
    tt_metal::detail::ReadFromDeviceL1(device, core, BASE_ADDR, ((size_bytes + 3) / 4) * 4, out);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(out.data());
    for (std::uint32_t i = 0; i < size_bytes; i++) {
        if (bytes[i] != static_cast<std::uint8_t>(i & 0xFF)) {
            log_error(tt::LogTest, "path {} size {} mismatch at {}: got 0x{:02x}", write_path, size_bytes, i, bytes[i]);
            return false;
        }
    }
    return true;
}

}  // namespace unit_tests::dm::quasar_cache_perf

class QuasarCacheWrite : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarCacheWrite, SpikeSingleRun) {
    if (unit_tests::dm::quasar_cache_perf::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(unit_tests::dm::quasar_cache_perf::run_cache_write(devices_[0], 256, /*uncached*/ 0));
}

}  // namespace tt::tt_metal
