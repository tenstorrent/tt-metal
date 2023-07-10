#pragma once

#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

struct CBConfig {
    const u32 num_pages = 1;
    const u32 page_size = TileSize(tt::DataFormat::Float16_b);
    const tt::DataFormat data_format = tt::DataFormat::Float16_b;
};

inline void initialize_program(Program& program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    auto dummy_writer_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto dummy_compute_kernel = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        cr_set,
        {},
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);
}

}   // end namespace basic_tests::circular_buffer
