// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, writer: the top-k of one 32-row tile as ROW_MAJOR rows of top_k bf16 scores and top_k uint16
// indices (the untilize + uint16 typecast of the composed chain).  The scores tile is [token, k]; the index tile
// is the sort's transposed [k, token] uint32 tile.  Rows are staged at a 64-byte pitch before the NoC write.
// Runtime arg 4 (token_mask) names the rows this core writes (the lane form gives each core eight tokens).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_row = get_arg_val<uint32_t>(2);
    const uint32_t rows_in_tile = get_arg_val<uint32_t>(3);
    const uint32_t token_mask =
        get_arg_val<uint32_t>(4);  // bit r: this core writes row r (all ones: one core per tile)

    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t row_bytes = top_k * 2;
    constexpr uint32_t stage_pitch = 64;

    constexpr auto scores_args = TensorAccessorArgs<0, 0>();
    constexpr auto indices_args = TensorAccessorArgs<
        scores_args.next_compile_time_args_offset(),
        scores_args.next_common_runtime_args_offset()>();
    const auto scores_out = TensorAccessor(scores_args, scores_addr);
    const auto indices_out = TensorAccessor(indices_args, indices_addr);

    DataflowBuffer idx_t(cb_idx_t);
    DataflowBuffer scores(cb_scores);
    DataflowBuffer stage(cb_stage);
    constexpr uint32_t stage_pages = get_named_compile_time_arg_val("stage_pages");
    stage.reserve_back(stage_pages);
    const uint32_t stage_base = (stage.get_write_ptr() + stage_pitch - 1) & ~(stage_pitch - 1);
    const uint32_t stage_scores = stage_base;
    const uint32_t stage_indices = stage_base + 32 * stage_pitch;

    {
        FUSED_ZONE("fz_rt_w_main");
        idx_t.wait_front(1);
        scores.wait_front(1);
        volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_t.get_read_ptr());
        volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores.get_read_ptr());
        for (uint32_t row = 0; row < rows_in_tile; ++row) {
            if (((token_mask >> row) & 1u) == 0) {
                continue;  // another core's token
            }
            volatile tt_l1_ptr uint16_t* srow =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_scores + row * stage_pitch);
            volatile tt_l1_ptr uint16_t* irow =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_indices + row * stage_pitch);
            const uint32_t face_row = row >> 4;
            const uint32_t in_face = row & 15;
            for (uint32_t k = 0; k < top_k; ++k) {
                srow[k] = sc[(face_row * 2) * 256 + in_face * 16 + k];                    // [token, k] bf16, k < 16
                irow[k] = static_cast<uint16_t>(idx[face_row * 256 + k * 16 + in_face]);  // [k, token] uint32
            }
            const uint32_t page = tile_row * 32 + row;
            noc_async_write(stage_scores + row * stage_pitch, scores_out.get_noc_addr(page), row_bytes);
            noc_async_write(stage_indices + row * stage_pitch, indices_out.get_noc_addr(page), row_bytes);
        }
        noc_async_write_barrier();
        idx_t.pop_front(1);
        scores.pop_front(1);
        stage.push_back(stage_pages);
    }
}
