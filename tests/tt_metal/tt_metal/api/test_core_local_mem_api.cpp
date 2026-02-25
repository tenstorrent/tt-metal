// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <vector>
#include <random>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "command_queue_fixture.hpp"
#include "context/metal_context.hpp"
#include <tt-metalium/distributed.hpp>
#include "gtest/gtest.h"
#include "mesh_device.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace {

void RunTest(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const CoreCoord core = {0, 0};
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::distributed::MeshWorkload workload;

    auto zero_coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto& mc = tt::tt_metal::MetalContext::instance();

    uint32_t unreserved_addr = mc.hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t alignment = mc.hal().get_alignment(tt::tt_metal::HalMemType::L1);

    // Ensure no regressions with the new API
    uint32_t cycles_addr = unreserved_addr + 0;
    uint32_t src_addr = std::max(
        (uint32_t)(unreserved_addr + alignment),
        (uint32_t)(unreserved_addr + (2 * sizeof(uint64_t))));  // two 64 bit cycle counters
    uint32_t num_bytes = (128 * 1024 * sizeof(uint32_t));       // 512KB
    uint32_t num_iterations = 1000;

    // Try using the memory API with the NoC API to send random data to the neighbor core
    CoreCoord neighbor_core{core.x + 1, core.y};
    auto neighbor_virtual_core = mesh_device->worker_core_from_logical_core(neighbor_core);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    uint32_t pattern = dist(gen);

    std::vector<uint32_t> compile_args = {
        cycles_addr,
        src_addr,
        num_bytes,
        num_iterations,
        neighbor_virtual_core.x,
        neighbor_virtual_core.y,
        pattern,
    };

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/core_local_mem_api.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args});
    workload.add_program(device_range, std::move(program));

    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

    uint64_t cycles_elapsed = 0;
    auto* device = mesh_device->get_devices()[0];
    auto virtual_core = device->worker_core_from_logical_core(core);
    mc.get_cluster().read_core(&cycles_elapsed, sizeof(uint64_t), tt_cxy_pair(device->id(), virtual_core), cycles_addr);

    uint64_t cycles_elapsed_legacy_api = 0;
    mc.get_cluster().read_core(
        &cycles_elapsed_legacy_api,
        sizeof(uint64_t),
        tt_cxy_pair(device->id(), virtual_core),
        cycles_addr + sizeof(uint64_t));

    uint64_t total_bytes = (uint64_t)num_bytes * num_iterations;

    double bytes_per_cycle = (double)total_bytes / (double)cycles_elapsed;
    double bytes_per_cycle_legacy_api = (double)total_bytes / (double)cycles_elapsed_legacy_api;

    double speedup = bytes_per_cycle_legacy_api / bytes_per_cycle;
    // Ensure no differences greater than 0.05%
    ASSERT_LT(std::abs(speedup - 1.0), 0.0005);

    log_info(
        tt::LogTest,
        "Read Test: Num Bytes / Iteration: {}, Iterations: {}, Cycles: {}, Bytes/Cycle: {:.4f}, Bytes/Cycle (Legacy "
        "API): {:.4f}",
        num_bytes,
        num_iterations,
        cycles_elapsed,
        bytes_per_cycle,
        bytes_per_cycle_legacy_api);

    // Verify data is the pattern
    std::vector<uint32_t> data(num_bytes / sizeof(uint32_t), 0);
    std::vector<uint32_t> expected_data(num_bytes / sizeof(uint32_t));
    std::iota(expected_data.begin(), expected_data.end(), pattern);
    mc.get_cluster().read_core(data.data(), num_bytes, tt_cxy_pair(device->id(), virtual_core), src_addr);
    ASSERT_EQ(data, expected_data);

    // Verify neighbor core received the data
    mc.get_cluster().read_core(data.data(), num_bytes, tt_cxy_pair(device->id(), neighbor_virtual_core), src_addr);
    ASSERT_EQ(data, expected_data);
}

}  // namespace

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardProgramFixture, TestSimpleL1Read) {
    for (auto& mesh_device : devices_) {
        RunTest(mesh_device.get());
    }
}

}  // namespace tt::tt_metal
