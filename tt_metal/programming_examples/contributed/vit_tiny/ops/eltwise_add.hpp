// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"

namespace vit {

// Multicore element-wise add: distributes tiles across cores.
inline void eltwise_add_op(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& src0_buf,
    const std::shared_ptr<distributed::MeshBuffer>& src1_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    uint32_t n_tiles) {
    Program program = CreateProgram();

    uint32_t num_cores = choose_num_cores(n_tiles);
    uint32_t tiles_per_core = n_tiles / num_cores;
    CoreRange cores({0, 0}, {num_cores - 1, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tile_size(cb_data_format);

    CreateCircularBuffer(program, cores,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, tile_size));
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, tile_size));
    CreateCircularBuffer(program, cores,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src0_buf).append_to(reader_ct_args);
    TensorAccessorArgs(*src1_buf).append_to(reader_ct_args);
    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/reader_binary_multicore.cpp",
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
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/compute/eltwise_add_compute.cpp",
        cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {tiles_per_core}});

    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t tile_start = c * tiles_per_core;
        CoreCoord core(c, 0);
        SetRuntimeArgs(program, reader_id, core,
            {src0_buf->address(), src1_buf->address(), tile_start, tiles_per_core});
        SetRuntimeArgs(program, writer_id, core,
            {dst_buf->address(), tile_start, tiles_per_core});
    }

    run_program(ctx, program);
}

}  // namespace vit
