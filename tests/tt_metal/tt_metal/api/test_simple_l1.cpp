// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
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
    uint32_t cycles_addr = unreserved_addr + 0;
    uint32_t src_addr = unreserved_addr + alignment;
    uint32_t num_bytes = (128 * 1024 * sizeof(uint32_t));  // 512KB
    uint32_t num_iterations = 5000;

    std::vector<uint32_t> compile_args = {
        cycles_addr,
        src_addr,
        num_bytes,
        num_iterations,
    };

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/local_l1_bandwidth.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args});
    workload.add_program(device_range, std::move(program));

    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

    uint64_t cycles_elapsed = 0;
    auto device = mesh_device->get_devices()[0];
    auto physical_core = device->worker_core_from_logical_core(core);
    mc.get_cluster().read_core(
        &cycles_elapsed, sizeof(uint64_t), tt_cxy_pair(device->id(), physical_core), cycles_addr);

    log_info(tt::LogTest, "Cycles elapsed: {} (0x{:016X})", cycles_elapsed, cycles_elapsed);

    uint64_t total_bytes = (uint64_t)num_bytes * num_iterations;
    double bytes_per_cycle = (double)total_bytes / (double)cycles_elapsed;

    log_info(
        tt::LogTest,
        "Read Test: Num Bytes / Iteration: {}, Iterations: {}, Cycles: {}, Bytes/Cycle: {:.4f}",
        num_bytes,
        num_iterations,
        cycles_elapsed,
        bytes_per_cycle);
}

}  // namespace

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardProgramFixture, TestSimpleL1Read) {
    for (auto& mesh_device : devices_) {
        RunTest(mesh_device.get());
    }
}

}  // namespace tt::tt_metal
