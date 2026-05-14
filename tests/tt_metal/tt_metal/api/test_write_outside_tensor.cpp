// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.OOB_Tensor_Gap_SanityCheck"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// Sanity check for the L1 out-of-bounds-tensor sanitizer in
// __emule_local_l1_to_ptr. Allocates one small L1 buffer, then has the kernel
// translate an L1 address that is comfortably below the buffer (still well
// within the L1 mmap and well above l1_unreserved_base, so it is not inside
// any system region and not inside any allocated tensor). The sanitizer is
// expected to abort with an Out-of-Bounds Write ASAN message.
TEST_F(MeshDeviceFixture, OOB_Tensor_Gap_SanityCheck) {
    ::setenv("TT_EMULE_STRICT_TENSOR", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // One small L1 buffer near the top of L1 (allocator is top-down for L1).
    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);

    // Pick a target that sits 64 KB below the buffer's start. That lands in
    // user-allocatable L1 (above l1_unreserved_base on a fresh device with a
    // single small buffer), but no allocation covers it.
    constexpr uint32_t gap_distance = 64 * 1024;
    uint32_t oob_addr = static_cast<uint32_t>(buf->address()) - gap_distance;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* bad_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *bad_ptr = 0x666;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {oob_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access address.*not part of any allocated tensor.*");
}

}  // namespace tt::tt_metal
