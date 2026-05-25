// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.NoC_Barrier_Missing_SanityCheck"

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, NoC_Barrier_Missing_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 1. Create a CB
    uint32_t cb_id = 0;
    CircularBufferConfig cb_config = CircularBufferConfig(2048, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Allocate a real L1 buffer as the noc_async_read destination so the
    // tensor-area sanitizer doesn't fire before cb_push_back triggers the
    // barrier check.
    auto dst_buf = Buffer::create(device, 1024, 1024, BufferType::L1);

    // 2. Kernel that reads then pushes WITHOUT a barrier
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst = get_arg_val<uint32_t>(0);
            uint64_t src_addr = get_noc_addr(0x20000);

            noc_async_read(src_addr, dst, 1024);
            // MISSING: noc_async_read_barrier();
            cb_push_back(0, 1);
        }
    )";

    auto kernel = CreateKernelFromString(program, kernel_src, logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dst_buf->address()});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Race Condition: cb_push_back.*called while a NoC read is still pending.*"
    );
}

}