// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, Semaphore_Direct_Write_SanityCheck) {
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // EMULE_SEM_BASE is a JIT-time define injected by the runner: the
    // firmware-style L1 offset of the reserved Semaphore region. Any scalar
    // pointer access into that range must trip the ASAN guard inside
    // __emule_local_l1_to_ptr and abort the kernel thread.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t sem_addr = EMULE_SEM_BASE;
            volatile uint32_t* illegal_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(sem_addr);
            *illegal_ptr = 0xABCD;
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Illegal Semaphore Access: Offset 0x.*is inside the reserved Semaphore region.*");
}

}  // namespace tt::tt_metal
