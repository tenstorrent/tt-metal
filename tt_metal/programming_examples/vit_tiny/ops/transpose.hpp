// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"

namespace vit {

// 2D matrix transpose: [Mt, Nt] tiles -> [Nt, Mt] tiles
// Combines tile-layout transposition (reader) with within-tile WH transpose (compute).
inline void transpose_2d_op(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    uint32_t Mt, uint32_t Nt) {
    Program program = CreateProgram();
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tile_size(cb_data_format);
    uint32_t n_tiles = Mt * Nt;

    CreateCircularBuffer(program, core,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, tile_size));
    CreateCircularBuffer(program, core,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);
    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "vit_tiny/kernels/dataflow/reader_transpose.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_buf).append_to(writer_ct_args);
    auto writer_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "vit_tiny/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "vit_tiny/kernels/compute/transpose_compute.cpp",
        core,
        ComputeConfig{.compile_args = {n_tiles}});

    SetRuntimeArgs(program, reader_id, core, {src_buf->address(), Mt, Nt});
    SetRuntimeArgs(program, writer_id, core, {dst_buf->address(), n_tiles});

    run_program(ctx, program);
}

}  // namespace vit
