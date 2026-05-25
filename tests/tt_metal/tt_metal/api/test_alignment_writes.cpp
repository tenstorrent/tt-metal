// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.NocRead_L1_Misaligned_SanityCheck:MeshDeviceFixture.NocWrite_L1_Misaligned_SanityCheck:MeshDeviceFixture.NocRead_DRAM_Misaligned_SanityCheck_WH:MeshDeviceFixture.NocWrite_DRAM_Misaligned_SanityCheck"

#include <gtest/gtest.h>
#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// L1->L1 read: src_noc lower 4 bits (0) != dst_l1 lower 4 bits (1) -> abort
TEST_F(MeshDeviceFixture, NocRead_L1_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src NOC addr offset = 0x30000, lower 4 bits = 0
            uint64_t src = get_noc_addr(0x30000);
            // dst lower 4 bits = 1 -- mismatches src
            uint32_t dst = 0x30001;
            noc_async_read(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*NOC Transfer Alignment.*L1.*lower 4 bits must match.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->L1 write: src_l1 lower 4 bits (0) != dst_noc lower 4 bits (1) -> abort
TEST_F(MeshDeviceFixture, NocWrite_L1_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src lower 4 bits = 0
            uint32_t src = 0x30000;
            // dst NOC addr offset = 0x30001, lower 4 bits = 1 -- mismatches src
            uint64_t dst = get_noc_addr(0x30001);
            noc_async_write(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*NOC Transfer Alignment.*L1.*lower 4 bits must match.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// DRAM->L1 read (WH): DRAM lower 8 bits (0x10) != L1 lower 8 bits (0x20) -> abort
// Constructs DRAM NOC address from the host-side NOC XY of DRAM bank 0.
TEST_F(MeshDeviceFixture, NocRead_DRAM_Misaligned_SanityCheck_WH) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Get DRAM bank 0 NOC coordinates and build a NOC address with offset 0x10.
    // NOC encoding: (y << 6) | x, shifted by 36 for the local-address field.
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) |
                       static_cast<uint32_t>(dram_noc_coord.x);
    uint64_t dram_src = (static_cast<uint64_t>(noc_xy) << 36) | 0x0010ULL;
    uint32_t dram_lo = static_cast<uint32_t>(dram_src & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_src >> 32);

    // L1 dst: lower 8 bits = 0x20 -- mismatches DRAM lower 8 bits (0x10)
    uint32_t l1_dst = 0x30020;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t dst = get_arg_val<uint32_t>(2);
            uint64_t src = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_read(src, dst, 64);
        }
    )";

    auto kernel = CreateKernelFromString(
        program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_dst});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*NOC Transfer Alignment.*DRAM.*lower bits must match.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->DRAM write (WH/BH): L1 lower 4 bits (0) != DRAM lower 4 bits (1) -> abort
TEST_F(MeshDeviceFixture, NocWrite_DRAM_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Build DRAM NOC address with offset 0x01 so lower 4 bits = 1.
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) |
                       static_cast<uint32_t>(dram_noc_coord.x);
    uint64_t dram_dst = (static_cast<uint64_t>(noc_xy) << 36) | 0x0001ULL;
    uint32_t dram_lo = static_cast<uint32_t>(dram_dst & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_dst >> 32);

    // L1 src: lower 4 bits = 0 -- mismatches DRAM lower 4 bits (1)
    uint32_t l1_src = 0x30000;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t src = get_arg_val<uint32_t>(2);
            uint64_t dst = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_write(src, dst, 16);
        }
    )";

    auto kernel = CreateKernelFromString(
        program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_src});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*NOC Transfer Alignment.*DRAM.*lower 4 bits must match.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
