// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Tensor_Padding_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"
#include "impl/emulation/host_sanitizers.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, Tensor_Padding_Violation_SanityCheck) {
    GTEST_SKIP() << "Temporarily disabled. See SANITIZER_CHECKS.md for details.";
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // 1. Create a buffer mimicking a padded tensor layout:
    // 1024 bytes of logical data, but padded to a 2048-byte physical block size.
    // Buffer::create only takes a single size — emule::register_logical_size()
    // declares that bytes [logical_size, physical_size) are hardware padding and
    // registers them with LiveL1PaddingRanges so the kernel-side sanitizer
    // in __emule_local_l1_to_ptr will fire on accesses into that region.
    uint32_t logical_size = 1024;
    uint32_t physical_size = 2048;
    auto buffer = Buffer::create(device, physical_size, physical_size, BufferType::L1);
    tt::tt_metal::emule::register_logical_size(*buffer, logical_size);

    // 2. Add an inline kernel that attempts to modify a padded datum at byte
    // 1028 — inside the 2048-byte allocation (so it passes the OOB-tensor
    // check) but past the 1024-byte logical extent (so it must trip the
    // padding check).
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t base_addr = get_arg_val<uint32_t>(0);
            uint32_t logical_limit = get_arg_val<uint32_t>(1);

            // __emule_local_l1_to_ptr runs the padding check internally on
            // the address we hand it. Translate the padded address itself
            // (not the buffer base) so the check inspects the violating
            // offset.
            volatile uint8_t* padded_ptr =
                (volatile uint8_t*)__emule_local_l1_to_ptr(base_addr + logical_limit + 4);
            *padded_ptr = 0xAA;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Pass the base address and the logical data cutoff boundary to the kernel
    SetRuntimeArgs(program, kernel, logical_core, {buffer->address(), logical_size});

    // 3. The emulator should intercept the illegal write inside l1_ptr
    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Tensor Padding Violation: Attempted to write to a padded memory region at address 0x.*");
}

}  // namespace tt::tt_metal
