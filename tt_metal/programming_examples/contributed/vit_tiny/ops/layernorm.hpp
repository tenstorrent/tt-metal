// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include <cstring>

namespace vit {

// Multicore row-wise LayerNorm: distributes tile-rows across cores.
// gamma/beta are [1, Wt*32] broadcast-row format, loaded by each core.
inline void layernorm_op(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& gamma_buf,
    const std::shared_ptr<distributed::MeshBuffer>& beta_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    uint32_t Mt, uint32_t Wt,
    float eps = 1e-6f) {
    Program program = CreateProgram();

    uint32_t num_cores = choose_num_cores(Mt);
    uint32_t rows_per_core = Mt / num_cores;
    CoreRange cores({0, 0}, {num_cores - 1, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tile_size(cb_data_format);

    // cb_in (c_0): input tiles, Wt per row
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, tile_size));
    // cb_scaler (c_1): 1/N tile
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, tile_size));
    // cb_gamma (c_2): gamma weights, Wt tiles
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, tile_size));
    // cb_beta (c_3): beta weights, Wt tiles
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_3, cb_data_format}})
            .set_page_size(CBIndex::c_3, tile_size));
    // cb_eps (c_4): eps tile
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_4, cb_data_format}})
            .set_page_size(CBIndex::c_4, tile_size));
    // cb_mean (c_5): mean intermediate
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_5, cb_data_format}})
            .set_page_size(CBIndex::c_5, tile_size));
    // cb_xmm (c_6): x-mean intermediate (Wt tiles)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_6, cb_data_format}})
            .set_page_size(CBIndex::c_6, tile_size));
    // cb_xmm2 (c_7): (x-mean)^2 intermediate (Wt tiles)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_7, cb_data_format}})
            .set_page_size(CBIndex::c_7, tile_size));
    // cb_var (c_8): variance intermediate
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(tile_size, {{CBIndex::c_8, cb_data_format}})
            .set_page_size(CBIndex::c_8, tile_size));
    // cb_norm (c_9): normalized intermediate (Wt tiles)
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_9, cb_data_format}})
            .set_page_size(CBIndex::c_9, tile_size));
    // cb_out (c_16): output tiles
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(Wt * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size));

    uint32_t eps_float_bits;
    std::memcpy(&eps_float_bits, &eps, sizeof(eps));
    uint32_t eps_bits = eps_float_bits >> 16;

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);
    TensorAccessorArgs(*gamma_buf).append_to(reader_ct_args);
    TensorAccessorArgs(*beta_buf).append_to(reader_ct_args);
    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/reader_layernorm_multicore.cpp",
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
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/compute/layernorm_compute.cpp",
        cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {rows_per_core, Wt}});

    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t start_row = c * rows_per_core;
        CoreCoord core(c, 0);
        SetRuntimeArgs(program, reader_id, core,
            {src_buf->address(), gamma_buf->address(), beta_buf->address(),
             start_row, rows_per_core, Wt, eps_bits});
        SetRuntimeArgs(program, writer_id, core,
            {dst_buf->address(), start_row * Wt, rows_per_core * Wt});
    }

    run_program(ctx, program);
}

}  // namespace vit
