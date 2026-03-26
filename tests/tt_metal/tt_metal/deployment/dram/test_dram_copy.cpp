// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include "tt_metal/tt_metal/deployment/eth/common.hpp"  // TODO

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"

#define BANDWIDTH_DRAM_COPY 35.0
#define BANDWIDTH_DRAM_COPY_SELF 15.0

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test_dram_copy(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t src_bank,
    uint32_t dst_bank) {
    /* ================= */
    auto* const device = mesh_device->get_devices()[0];

    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;

    uint32_t dram_start_addr = 0x500000u;
    uint32_t dram_end_addr = 0xff000000u;
    // uint32_t dram_end_addr = dram_start_addr + (2 << 10);
    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    uint32_t transfer_size = 256 * 1024;
    uint64_t total_transferred = dram_end_addr - dram_start_addr;
    TT_FATAL(total_transferred % transfer_size == 0, "Total transfer must be a multiple of transfer size");

    uint32_t c = 123;
    size_t wordcount = total_transferred / sizeof(uint32_t);
    vector<uint32_t> inputs, zeros(wordcount);
    inputs.reserve(wordcount);
    for (long i = 0; i < wordcount; i++) {
        inputs.push_back(c++);
    }

    detail::WriteToDeviceDRAMChannel(device, src_bank, dram_start_addr, inputs);
    if (src_bank != dst_bank) {
        detail::WriteToDeviceDRAMChannel(device, dst_bank, dram_start_addr, zeros);
    }

    struct l1_allocator alloc = new_tensix_allocator();

    uint32_t buffer0 = l1_alloc(&alloc, transfer_size);
    uint32_t buffer1 = l1_alloc(&alloc, transfer_size);
    uint32_t delta_addr = l1_alloc(&alloc, sizeof(uint64_t));

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    CoreCoord core(5, 5);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
        .compile_args = {
            dram_start_addr,
            dram_end_addr,
            transfer_size,
            delta_addr,
            buffer0,
            buffer1,
            src_bank,
            dst_bank,
        }};

    auto kernel = tt_metal::CreateKernel(
        program, "tests/tt_metal/tt_metal/deployment/kernels/dram_copy.cpp", core, kernel_config);

    tt_metal::SetRuntimeArgs(program, kernel, core, {});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    double threshold = src_bank == dst_bank ? BANDWIDTH_DRAM_COPY_SELF : BANDWIDTH_DRAM_COPY;

    bool pass = true;
    pass &= bandwidth_check(device, core, delta_addr, total_transferred, threshold);
    pass &= dram_data_check(device, dram_start_addr, dram_end_addr, dst_bank, inputs);

    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentDramCopy) {
    bool pass = true;

    SignalGuard g(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        auto* const device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "device id: {}", device->id());

        const int num_banks = device->num_dram_channels();
        for (int i = 0; i < num_banks; i++) {
            for (int j = 0; j < num_banks; j++) {
                if (g_stop_requested.load()) {
                    GTEST_SKIP() << "Test interrupted by user after current test finished.";
                    return;
                }

                if (0 && i == j) {
                    continue;  // TODO
                }

                log_info(tt::LogTest, "  sending from bank {} to bank {}", i, j);
                pass &= run_test_dram_copy(this, mesh_device, i, j);
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
