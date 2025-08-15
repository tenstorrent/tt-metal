// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::detail;

namespace basic_tests::circular_buffer {

struct CBConfig {
    const uint32_t num_pages = 1;
    const uint32_t page_size = TileSize(tt::DataFormat::Float16_b);
    const tt::DataFormat data_format = tt::DataFormat::Float16_b;
};

inline void initialize_program(Program& program, const CoreRangeSet& cr_set) {
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

}  // end namespace basic_tests::circular_buffer
