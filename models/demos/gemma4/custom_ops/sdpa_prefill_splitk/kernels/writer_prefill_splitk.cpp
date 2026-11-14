// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Writer: the stock reduce scaler, column identity and lightweight causal mask tiles once,
// then per work piece the normalised partial output (slot ``slot``) and its statistics (m in tile column 0, l in
// tile column 1 of the stats tensor).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t num_slots = get_compile_time_arg_val(1);
    constexpr uint32_t Sqt = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(8);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(11);

    constexpr auto out_args = TensorAccessorArgs<12>();
    constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_pieces = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t out_tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_stats);
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<out_tile_bytes, num_cores>();

    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr);
    const auto out_tile_shape = TensorTileShape(1, num_slots * NQH, Sqt, vDHt);
    const auto stats_tile_shape = TensorTileShape(1, num_slots * NQH, Sqt, 2);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    generate_lightweight_mask_tiles<0u, 0u, cb_mask_in, true, 0>(noc);

    CircularBuffer cb_st(cb_stats);
    for (uint32_t piece = 0; piece < num_pieces; ++piece) {
        const uint32_t head = get_arg_val<uint32_t>(argidx++);
        const uint32_t q_chunk = get_arg_val<uint32_t>(argidx++);
        const uint32_t slot = get_arg_val<uint32_t>(argidx++);
        const uint32_t plane = slot * NQH + head;
        const uint32_t row0 = q_chunk * Sq_chunk_t;

        // statistics first: compute packs them before the normalised output
        cb_st.wait_front(2 * Sq_chunk_t);
        uint32_t offset = 0;
        for (uint32_t col = 0; col < 2; ++col) {
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                noc.async_write(
                    cb_st,
                    stats_writer,
                    stats_tile_bytes,
                    {.offset_bytes = offset},
                    {.page_id = stats_tile_shape.id_of(0, plane, row0 + r, col)});
                offset += stats_tile_bytes;
            }
        }
        noc.async_write_barrier();
        cb_st.pop_front(2 * Sq_chunk_t);

        write_block<0>(
            noc,
            out_writer,
            cb_out,
            out_chunk_tiles,
            Sq_chunk_t,
            vDHt,
            out_tile_shape.id_of(0, plane, row0, 0),
            out_tile_bytes,
            barrier_threshold);
    }
}
