// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.CB_Reservation_Overflow_SanityCheck"

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

TEST_F(MeshDeviceFixture, CB_Reservation_Overflow_SanityCheck) {
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    
    Program program = CreateProgram();
    uint32_t cb_id = 0; 
    uint32_t num_pages = 2;
    uint32_t page_size = 1024;
    CircularBufferConfig cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, tt::DataFormat::Float16_b}})
        .set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // Attempt to reserve 3 pages on a CB that only has 2
            cb_reserve_back(0, 3);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*EMULE BUG: cb_reserve_back.*requests more than capacity.*"
    );
}

}  // namespace tt::tt_metal