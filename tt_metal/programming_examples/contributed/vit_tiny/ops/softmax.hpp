// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"

namespace vit {

// Multicore row-wise softmax: distributes tile-rows across cores.
// input [Mt*32, Wt*32] -> output [Mt*32, Wt*32]
inline void softmax_op(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    uint32_t Mt, uint32_t Wt) {
    Program program = CreateProgram();

    uint32_t num_cores = choose_num_cores(Mt);
    uint32_t rows_per_core = Mt / num_cores;
    CoreRange cores({0, 0}, {num_cores - 1, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tile_size(cb_data_format);

    // cb_in: input tiles (Wt tiles per row)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, tile_size));
    // cb_scaler: 1.0 tile for reduce
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, tile_size));
    // cb_exps: exp intermediate (Wt tiles)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, tile_size));
    // cb_recip: 1/sum (1 tile)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_3, cb_data_format}})
            .set_page_size(CBIndex::c_3, tile_size));
    // cb_max: row max (1 tile)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_4, cb_data_format}})
            .set_page_size(CBIndex::c_4, tile_size));
    // cb_sub: x - max intermediate (Wt tiles)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_5, cb_data_format}})
            .set_page_size(CBIndex::c_5, tile_size));
    // cb_out: output tiles
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);
    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/reader_softmax_multicore.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_buf).append_to(writer_ct_args);
    auto writer_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/writer_unary_multicore.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/compute/softmax_compute.cpp",
        cores,
        ComputeConfig{.compile_args = {rows_per_core, Wt}});

    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t start_row = c * rows_per_core;
        CoreCoord core(c, 0);
        SetRuntimeArgs(program, reader_id, core,
            {src_buf->address(), start_row, rows_per_core, Wt});
        SetRuntimeArgs(program, writer_id, core,
            {dst_buf->address(), start_row * Wt, rows_per_core * Wt});
    }

    run_program(ctx, program);
}

}  // namespace vit
