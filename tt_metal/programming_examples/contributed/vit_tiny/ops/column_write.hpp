// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"

namespace vit {

// Write tiles from src (contiguous [Mt, slice_Wt]) to specific columns in dst [Mt, total_Wt].
// Used for head concatenation: write each head's output to the correct column offset.
inline void column_write_op(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    uint32_t Mt, uint32_t total_Wt, uint32_t start_col, uint32_t slice_Wt) {
    Program program = CreateProgram();
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tile_size(cb_data_format);

    CreateCircularBuffer(program, core,
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, tile_size));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);
    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/reader_unary_to_out.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_buf).append_to(writer_ct_args);
    auto writer_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/vit_tiny/kernels/dataflow/writer_column_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    SetRuntimeArgs(program, reader_id, core, {src_buf->address(), Mt * slice_Wt});
    SetRuntimeArgs(program, writer_id, core, {dst_buf->address(), Mt, total_Wt, start_col, slice_Wt});

    run_program(ctx, program);
}

}  // namespace vit
