// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA attention (compute, on every attention core; core (b, g) is head group g of S-block b).
//
// Latent attention of 8 heads (one [8, 32] tile row of absorbed queries, DHt tiles wide) over the K chunks of this
// S-block, with the flash-attention LLKs of models/demos/deepseek_v3_b1 (api/compute/experimental/sdpa.h):
// compute_sdpa_chunk keeps the running (O, max, sum) in DEST over a user's chunks, sdpa_tail merges two partials.
// K = the whole row (latent | RoPE), V = its first vDHt tiles; kv_b is applied elsewhere, after the merges.
//
// Phase 1, users ascending: attend over the user's chunks here and pack the partial (O unnormalised, max / sum)
//   to cb_final (this block is the user's root and receives nothing), cb_send (it sends and receives nothing) or
//   cb_part (it receives first).
// Phase 2, users ascending, steps ascending: merge the partial received in slot (u, step) into the user's own
//   partial; the last merge goes to cb_final or cb_send. Slots nothing is due in are popped unused.
// Nothing is normalised here and the sink is not added: the cross-chip merge does both (layout.py).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/eltwise_unary/exp.h"

#include "sm_common.hpp"

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_q = get_named_compile_time_arg_val("cb_q");
    constexpr uint32_t cb_k = get_named_compile_time_arg_val("cb_k");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("cb_ctrl_a");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_send_o = get_named_compile_time_arg_val("cb_send_o");
    constexpr uint32_t cb_send_ms = get_named_compile_time_arg_val("cb_send_ms");
    constexpr uint32_t cb_final_o = get_named_compile_time_arg_val("cb_final_o");
    constexpr uint32_t cb_final_ms = get_named_compile_time_arg_val("cb_final_ms");
    constexpr uint32_t cb_part_o = get_named_compile_time_arg_val("cb_part_o");
    constexpr uint32_t cb_part_ms = get_named_compile_time_arg_val("cb_part_ms");
    constexpr uint32_t cb_tmp_o = get_named_compile_time_arg_val("cb_tmp_o");
    constexpr uint32_t cb_tmp_ms = get_named_compile_time_arg_val("cb_tmp_ms");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("dht");
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vdht");
    constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale_fp32");
    constexpr uint32_t dst_size = get_named_compile_time_arg_val("dst_size");
    constexpr uint32_t Sk_chunk_t = CHUNK / sm::TILE_ROWS;
    static_assert(vDHt % dst_size == 0 && vDHt % 2 == 0 && USERS <= sm::MAX_USERS);

    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;
    constexpr bool exp_approx_mode = false;
    // DEST layout in rows of 8x32 tiles (16 DEST rows each): O | max, sum | correction | scores.
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
    constexpr uint32_t output_granularity = vDHt;
    constexpr uint32_t num_blocks = vDHt / dst_size;
    constexpr uint32_t block_size = vDHt / num_blocks;

    const uint32_t block =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>()
            .block;

    deepseek_compute_kernel_init();

    cb_wait_front(cb_ctrl, 1);
    uint32_t counts[sm::MAX_USERS];
    for (uint32_t w = 0; w < sm::COUNT_WORDS; ++w) {  // two counts per word
        const uint32_t word = ckernel::read_tile_value(cb_ctrl, 0, w);
        if (2 * w < USERS) {
            counts[2 * w] = word & 0xFFFF;
        }
        if (2 * w + 1 < USERS) {
            counts[2 * w + 1] = word >> 16;
        }
    }
    cb_pop_front(cb_ctrl, 1);

    // ---- phase 1: attend ----
    for (uint32_t u = 0; u < USERS; ++u) {
        const uint32_t here = sm::chunks_here<GROUPS, CHUNK>(counts[u], block, u);
        if (here == 0) {
            continue;
        }
        const uint32_t r = sm::rel<GROUPS>(block, u);
        const uint32_t n_act = sm::active_blocks<GROUPS, CHUNK>(counts[u]);
        const bool merges = sm::num_receives(r, n_act) != 0;
        const uint32_t o_cb = merges ? cb_part_o : (r == 0 ? cb_final_o : cb_send_o);
        const uint32_t ms_cb = merges ? cb_part_ms : (r == 0 ? cb_final_ms : cb_send_ms);
        const uint32_t last_here = sm::first_chunk<GROUPS>(block, u) + (here - 1) * GROUPS;
        const bool mask_last = last_here == sm::ceil_div(counts[u], CHUNK) - 1 && (counts[u] % CHUNK) != 0;

        reconfig_full_operand(cb_k, cb_q);
        pack_reconfig_data_format<true>(cb_send_o);
        PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
        PACK(SFPU_UNARY_INIT_FN(
            exponential,
            sfpu::exp_init,
            (true /*APPROXIMATION_MODE*/, scale_fp32, true /*CLAMP_NEGATIVE*/, DST_ACCUM_MODE)));
        sdpa_custom_mm_block_init_pack_short();
        pack_block_contiguous_init(o_cb);

        if (mask_last) {
            cb_wait_front(cb_mask, 1);
        }
        cb_wait_front(cb_q, DHt);
        cb_reserve_back(o_cb, vDHt);
        cb_reserve_back(ms_cb, 1);
        tile_regs_acquire();
        for (uint32_t chunk = 0; chunk < here; ++chunk) {
            const bool last_chunk = chunk == here - 1;
            compute_sdpa_chunk<
                Sk_chunk_t,
                DHt,
                vDHt,
                scale_fp32,
                transpose_k,
                transpose_v,
                packed_tile_size,
                exp_approx_mode,
                /*qk_signal_granularity=*/1,
                /*exp_signal_granularity=*/1,
                output_granularity,
                false>(
                cb_q,
                cb_k,
                cb_k,  // V is the first vDHt tiles of every K row
                cb_mask,
                o_cb,
                mm1_dst_offset,
                mm2_dst_offset,
                max_dst_offset,
                sum_dst_offset,
                corr_exp_dst_offset,
                chunk == 0,
                last_chunk,
                mask_last && last_chunk);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_block_contiguous(max_dst_tile_offset, ms_cb, 1);
        cb_push_back(ms_cb, 1);
        for (uint32_t i = 0; i < vDHt; i += output_granularity) {
            PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
            pack_block_contiguous(mm2_dst_tile_offset + i, o_cb, output_granularity);
            PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
        }
        cb_push_back(o_cb, vDHt);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        sdpa_custom_mm_block_uninit();
        MATH(t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU));

        cb_pop_front(cb_q, DHt);
        if (mask_last) {
            cb_pop_front(cb_mask, 1);
        }
    }

    // ---- phase 2: merge ----
    if (!sm::any_receive<USERS, GROUPS, CHUNK>(counts, block)) {
        return;  // nothing is due on this core: the I/O kernel publishes no receive slot
    }
    reconfig_full_operand_srca(cb_ms_in);
    exp_tile_init<exp_approx_mode, scale_fp32>();
    for (uint32_t u = 0; u < USERS; ++u) {
        const uint32_t r = sm::rel<GROUPS>(block, u);
        const uint32_t n_act = sm::active_blocks<GROUPS, CHUNK>(counts[u]);
        const uint32_t total = sm::num_receives(r, n_act);
        uint32_t done = 0;
        for (uint32_t s = 0; s < sm::MERGE_STEPS; ++s) {
            if (!sm::receives(r, n_act, s)) {
                cb_wait_front(cb_ms_in, 1);
                cb_pop_front(cb_ms_in, 1);
                cb_wait_front(cb_out_in, vDHt);
                cb_pop_front(cb_out_in, vDHt);
                continue;
            }
            const bool first = done == 0;
            const bool last = done == total - 1;
            const uint32_t prev_o = first ? cb_part_o : cb_tmp_o;
            const uint32_t prev_ms = first ? cb_part_ms : cb_tmp_ms;
            const uint32_t cur_o = last ? (r == 0 ? cb_final_o : cb_send_o) : cb_tmp_o;
            const uint32_t cur_ms = last ? (r == 0 ? cb_final_ms : cb_send_ms) : cb_tmp_ms;
            sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                cb_ms_in, prev_ms, cur_ms, cb_out_in, prev_o, cur_o);
            ++done;
        }
    }
}
