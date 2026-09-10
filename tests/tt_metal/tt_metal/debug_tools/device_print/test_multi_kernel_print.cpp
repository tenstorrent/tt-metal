// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-kernel DEVICE_PRINT tests covering both the fast-dispatch DRAM
// aggregation path (dispatch_s) and the slow-dispatch L1-polling fallback.
//
// Primary motivation: regression coverage for the dispatch_s NOC_CTRL
// state-corruption fix. dispatch_s reuses NCRISC_RD_CMD_BUF for inline
// writes, leaving NOC_CTRL in write mode; the DRAM-aggregating DEVICE_PRINT
// path subsequently issues reads on that same cmd buf, which in
// DM_DEDICATED_NOC mode does not reprogram NOC_CTRL — so the read goes out
// but the NIU never marks a read response, and dispatch_s spins forever on
// the barrier. The fix in device_print_dispatch.h saves/resets/restores
// NOC_CTRL around every execute() and shutdown(). Under fast dispatch these
// tests hang without the fix and pass with it; under slow dispatch (no
// dispatch_s) they exercise the legacy per-core L1 polling path.

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include "tests/tt_metal/tt_metal/eth/eth_test_common.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* kWorkerKernel = "tests/tt_metal/tt_metal/test_kernels/device_print/print_simple_string.cpp";
constexpr const char* kEriscKernel = "tests/tt_metal/tt_metal/test_kernels/device_print/erisc_print.cpp";

const std::vector<std::string> kWorkerExpected = {"Hello world!", "First line.", "Second line."};
const std::vector<std::string> kEriscExpected = {"Test Debug Print: ERISC"};

distributed::MeshCoordinateRange single_device_range() {
    auto zero = distributed::MeshCoordinate(0, 0);
    return distributed::MeshCoordinateRange(zero, zero);
}

Program& add_empty_program(distributed::MeshWorkload& workload) {
    auto range = single_device_range();
    workload.add_program(range, Program());
    return workload.get_programs().at(range);
}

// One print kernel per node, each its own KernelSpec in its own WorkUnit. These tests are about a
// program carrying several distinct kernels, so this deliberately does not collapse them into a
// single kernel fanned out over a node range.
Program make_worker_print_program(
    distributed::MeshDevice& mesh_device, const std::vector<experimental::NodeCoord>& nodes) {
    experimental::ProgramSpec spec{.name = "dprint_multi_kernel"};
    for (size_t i = 0; i < nodes.size(); i++) {
        const experimental::KernelSpecName name{"worker_print_" + std::to_string(i)};
        spec.kernels.push_back(experimental::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{kWorkerKernel},
            .num_threads = 1,
            .hw_config = DevicePrintFixture::SingleThreadDmConfig(mesh_device.arch()),
        });
        spec.work_units.push_back(
            experimental::WorkUnitSpec{.name = "wu_" + std::to_string(i), .kernels = {name}, .target_nodes = nodes[i]});
    }
    return experimental::MakeProgramFromSpec(mesh_device, spec);
}

EthernetConfig make_active_eth_config() {
    constexpr DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    EthernetConfig config{.noc = static_cast<NOC>(processor), .processor = processor};
    config.eth_mode = Eth::SENDER;
    eth_test_common::set_arch_specific_eth_config(config);
    return config;
}

}  // namespace

// Single program, two BRISC kernels on two different worker cores.
TEST_F(DevicePrintFixture, TwoWorkerKernelsSameProgram) {
    for (auto& mesh_device : this->devices_) {
        // Kernel placement is bounds-checked against the compute grid, which is a single node on
        // some Quasar configurations.
        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        if (grid.x < 2) {
            log_info(tt::LogTest, "Skipping device (need a compute grid at least 2 wide, have {}x{})", grid.x, grid.y);
            continue;
        }

        distributed::MeshWorkload workload;
        workload.add_program(single_device_range(), make_worker_print_program(*mesh_device, {{0, 0}, {1, 0}}));

        this->RunProgram(mesh_device, workload);
        EXPECT_TRUE(FileContainsAllStrings(this->dprint_file_name, kWorkerExpected));
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

// Single program, two active ETH kernels (DM0) on two different ETH cores.
TEST_F(DevicePrintFixture, TwoActiveEthKernelsSameProgram) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const auto eth_cores = device->get_active_ethernet_cores(true);
        if (eth_cores.size() < 2) {
            log_info(
                tt::LogTest, "Skipping device {} (need >=2 active ETH cores, have {})", device->id(), eth_cores.size());
            continue;
        }

        distributed::MeshWorkload workload;
        Program& program = add_empty_program(workload);

        const CoreRangeSet crs(std::set<CoreRange>(eth_cores.begin(), eth_cores.end()));
        CreateKernel(program, kEriscKernel, crs, make_active_eth_config());

        this->RunProgram(mesh_device, workload);
        EXPECT_TRUE(FileContainsAllStrings(this->dprint_file_name, kEriscExpected));
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

// Two programs run back-to-back on the same worker core / RISC.
TEST_F(DevicePrintFixture, TwoWorkerProgramsBackToBack) {
    for (auto& mesh_device : this->devices_) {
        for (int i = 0; i < 2; i++) {
            distributed::MeshWorkload workload;
            workload.add_program(single_device_range(), make_worker_print_program(*mesh_device, {{0, 0}}));
            this->RunProgram(mesh_device, workload);
        }

        EXPECT_TRUE(FileContainsAllStrings(this->dprint_file_name, kWorkerExpected));
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

// Two programs run back-to-back on the same active ETH core / RISC.
TEST_F(DevicePrintFixture, TwoActiveEthProgramsBackToBack) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        const auto eth_cores = device->get_active_ethernet_cores(true);
        if (eth_cores.empty()) {
            log_info(tt::LogTest, "Skipping device {} (no active ETH cores)", device->id());
            continue;
        }

        const CoreCoord core = *eth_cores.begin();
        const EthernetConfig config = make_active_eth_config();
        for (int i = 0; i < 2; i++) {
            distributed::MeshWorkload workload;
            Program& program = add_empty_program(workload);
            CreateKernel(program, kEriscKernel, core, config);
            this->RunProgram(mesh_device, workload);
        }

        EXPECT_TRUE(FileContainsAllStrings(this->dprint_file_name, kEriscExpected));
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}
