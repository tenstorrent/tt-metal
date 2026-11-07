// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Qwen context-parallel SDPA writer: both owned ring slices of a mesh column in one program.
// Derived from the stock writer_interleaved.cpp (stock_writer.cpp here), streaming + lightweight causal mask
// path only. The setup (reduce scaler, column identity, lightweight mask palette) is the stock one; the output
// loop walks this core's item list and writes item (p, nq, q chunk) at local rows p * valid_Sqt + q chunk rows,
// i.e. the column's local order [slice c ; slice 7 - c]. With out_concat_heads the heads are written side by
// side ([1, 1, 2 * Sq, NQH * DH], what concat + nlp_concat_heads produce), else [1, NQH, 2 * Sq, DH].
//
// Runtime args: out_addr, n_items, item... (item = (p << 16) | (nq << 8) | q_chunk).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(2);
    constexpr uint32_t out_Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr bool out_concat_heads = get_compile_time_arg_val(9) == 1;

    constexpr auto out_args = TensorAccessorArgs<10>();

    constexpr uint32_t cb_arg_offset = out_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 3);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    constexpr uint32_t item_arg_base = 2;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto out_tile_shape =
        out_concat_heads ? TensorTileShape(B, 1, out_Sqt, NQH * vDHt) : TensorTileShape(B, NQH, out_Sqt, vDHt);
    constexpr uint32_t out_row_stride = out_concat_heads ? NQH * vDHt : vDHt;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    // Lightweight causal mask palette [neginf, causal_diag], left permanently fronted.
    generate_lightweight_mask_tiles<0u, 0u, cb_mask_in, true, 0u>(noc);

    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = get_arg_val<uint32_t>(item_arg_base + it);
        const uint32_t phase = item >> 16;
        const uint32_t nq = (item >> 8) & 0xFF;
        const uint32_t q_chunk = item & 0xFF;
        const uint32_t write_offset = phase * valid_Sqt;

        const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
        const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
        const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
        uint32_t out_tile_id;
        if constexpr (out_concat_heads) {
            out_tile_id = out_tile_shape.id_of(0, 0, write_offset + out_row_start_tile, nq * vDHt);
        } else {
            out_tile_id = out_tile_shape.id_of(0, nq, write_offset + out_row_start_tile, 0);
        }
        write_block_row_grouped(
            noc,
            out_writer,
            cb_out,
            Sq_chunk_t,
            out_row_tile_count,
            vDHt,
            out_tile_id,
            tile_bytes,
            out_subblock_h,
            barrier_threshold,
            out_row_stride);
    }
}
