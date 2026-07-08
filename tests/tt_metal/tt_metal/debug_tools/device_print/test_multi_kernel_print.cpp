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
        distributed::MeshWorkload workload;
        Program& program = add_empty_program(workload);

        const std::vector<CoreCoord> cores = {{0, 0}, {1, 0}};
        for (const auto& core : cores) {
            CreateKernel(
                program,
                kWorkerKernel,
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        }

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
        constexpr CoreCoord core = {0, 0};
        for (int i = 0; i < 2; i++) {
            distributed::MeshWorkload workload;
            Program& program = add_empty_program(workload);
            CreateKernel(
                program,
                kWorkerKernel,
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
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
