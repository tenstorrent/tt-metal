// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form writer (rows == 1; indices first): the top-k of one 32-row tile as ROW_MAJOR rows of
// top_k bf16 scores and top_k uint16 indices (the untilize + uint16 typecast of the composed chain).  The scores tile
// is [token, k]; the index tile is the sort's transposed [k, token] uint32 tile.  Rows are staged at a 64-byte pitch
// before the NoC write. Runtime arg 4 (token_mask) names the rows this core writes (the lane form gives each core eight
// tokens).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

#ifndef RTF_IDXF
#define RTF_IDXF 0
#endif

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
        // rows == 1 (token 0).  The indices are known when core A's reader pushes cb_idx_t, long before the scores:
        // their rows go out first (RTF_WRITER_EARLY), the scores' behind the division.
        idx_t.wait_front(1);
        volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_t.get_read_ptr());
        volatile tt_l1_ptr uint16_t* srow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_scores);
        volatile tt_l1_ptr uint16_t* irow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_indices);
        for (uint32_t k = 0; k < top_k; ++k) {
#if RTF_IDXF
            const uint32_t bits = idx[k * 16];  // fp32 integer 0..511
            irow[k] =
                static_cast<uint16_t>(bits == 0u ? 0u : (((bits & 0x7FFFFFu) | 0x800000u) >> (150u - (bits >> 23))));
#else
            irow[k] = static_cast<uint16_t>(idx[k * 16]);  // [k, token] uint32, token 0
#endif
        }
        const uint32_t page = tile_row * 32;
        noc_async_write(stage_indices, indices_out.get_noc_addr(page), row_bytes);
#ifdef MOE_FUSE_SEM
        // QWEN38_MOE_FUSE=1 only (the define is set by the moe_experts fused form alone): the routed-expert cores of
        // the same program wait for the indices; once the rows are in the drain shard (acknowledged) raise their
        // semaphore.
        noc_async_write_barrier();
        {
            const uint32_t fuse_cores = get_arg_val<uint32_t>(5);
            const uint32_t fuse_sem = get_semaphore(MOE_FUSE_SEM);
            for (uint32_t c = 0; c < fuse_cores; ++c) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(6 + 2 * c), get_arg_val<uint32_t>(7 + 2 * c), fuse_sem), 1);
            }
        }
#endif
        scores.wait_front(1);
        volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores.get_read_ptr());
        for (uint32_t k = 0; k < top_k; ++k) {
            srow[k] = sc[k];  // [token, k] bf16, row 0 of face 0
        }
        noc_async_write(stage_scores, scores_out.get_noc_addr(page), row_bytes);
        noc_async_write_barrier();
        idx_t.pop_front(1);
        scores.pop_front(1);
        stage.push_back(stage_pages);
    }
}
