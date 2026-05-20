// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Fabric_Access_Violation_SanityCheck"

#include <gtest/gtest.h>
#include <cstdlib>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, Fabric_Access_Violation_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Allocate a real L1 buffer as the read destination so the tensor-area
    // sanitizer doesn't fire before the fabric check sees (50, 50).
    auto dst_buf = Buffer::create(device, 64, 64, BufferType::L1);

    // Coordinate (50, 50) is definitely not on a standard N150/N300 worker grid
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst_addr = get_arg_val<uint32_t>(0);
            // Target a core that doesn't exist in our 1x1 or 8x8 mesh
            uint64_t ghost_noc_addr = get_noc_addr(50, 50, 0x1000);
            noc_async_read(ghost_noc_addr, dst_addr, 16);
        }
    )";

    auto kernel = CreateKernelFromString(program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Fabric Access Violation: Attempted to access unallocated Core at NOC coordinates \\(50, 50\\).*"
    );
}

}