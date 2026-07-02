// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

namespace {

constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tiles_per_sender = get_compile_time_arg_val(1);
constexpr uint32_t tile_scratch_bytes = get_compile_time_arg_val(2);
constexpr uint32_t winner_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t num_senders = get_compile_time_arg_val(4);
constexpr uint32_t expected_remote_incs = get_compile_time_arg_val(5);
constexpr uint32_t receiver_semaphore_addr = get_compile_time_arg_val(6);
constexpr uint32_t tile_cb = get_compile_time_arg_val(7);
constexpr uint32_t gather_cb = get_compile_time_arg_val(8);
constexpr uint32_t sender_idx = get_compile_time_arg_val(9);
constexpr bool is_final_core = get_compile_time_arg_val(10) == 1;
constexpr uint32_t active_batch_size = get_compile_time_arg_val(11);
constexpr uint32_t output_pair_page_bytes = get_compile_time_arg_val(12);

FORCE_INLINE bool is_better_candidate(
    uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
    return bfloat16_greater(candidate_score, best_score) ||
           ((candidate_score == best_score) && (candidate_index < best_index));
}

FORCE_INLINE void write_winner_slot(uint32_t slot_addr, uint16_t score, uint32_t index) {
    auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
    auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
    slot_u16_ptr[0] = score;
    slot_u32_ptr[1] = index;
}

FORCE_INLINE uint32_t tile_offset(uint32_t row, uint32_t col) {
    const uint32_t row_face = row >> 4;
    const uint32_t col_face = col >> 4;
    const uint32_t face = row_face * 2 + col_face;
    return face * 256 + (row & 0xF) * 16 + (col & 0xF);
}

}  // namespace

void kernel_main() {
    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_pair_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t final_noc_x = get_common_arg_val<uint32_t>(2);
    const uint32_t final_noc_y = get_common_arg_val<uint32_t>(3);
    const uint32_t global_vocab_offset = get_common_arg_val<uint32_t>(4);

    constexpr auto scores_args = TensorAccessorArgs<13>();
    constexpr uint32_t score_page_bytes = scores_args.get_aligned_page_size();
    static_assert(score_page_bytes <= tile_scratch_bytes);
    const auto scores_accessor = TensorAccessor(scores_args, scores_addr);

    const uint32_t gather_addr = get_write_ptr(gather_cb);
    const uint32_t tile_begin = sender_idx * tiles_per_sender;
    const uint32_t tile_end = (tile_begin + tiles_per_sender < num_tiles) ? tile_begin + tiles_per_sender : num_tiles;

    for (uint32_t batch = 0; batch < active_batch_size; ++batch) {
        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;
        for (uint32_t tile = tile_begin; tile < tile_end; ++tile) {
            cb_reserve_back(tile_cb, 1);
            const uint32_t tile_addr = get_write_ptr(tile_cb);
            noc_async_read(scores_accessor.get_noc_addr(tile), tile_addr, score_page_bytes);
            noc_async_read_barrier();

            auto tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t token_index = global_vocab_offset + tile * 32 + col;
                const uint16_t score = tile_ptr[tile_offset(batch, col)];
                if (is_better_candidate(score, token_index, best_score, best_index)) {
                    best_score = score;
                    best_index = token_index;
                }
            }
            cb_pop_front(tile_cb, 1);
        }

        const uint32_t slot_addr = gather_addr + (batch * num_senders + sender_idx) * winner_page_bytes;
        write_winner_slot(slot_addr, best_score, best_index);
    }
    if constexpr (!is_final_core) {
        const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
        const uint64_t dst_sem_noc_addr = final_noc_base | static_cast<uint64_t>(receiver_semaphore_addr);
        for (uint32_t batch = 0; batch < active_batch_size; ++batch) {
            const uint32_t src_addr = gather_addr + (batch * num_senders + sender_idx) * winner_page_bytes;
            const uint32_t dst_addr = gather_addr + (batch * num_senders + sender_idx) * winner_page_bytes;
            const uint64_t dst_noc_addr = final_noc_base | static_cast<uint64_t>(dst_addr);
            noc_async_write_one_packet<true, true>(src_addr, dst_noc_addr, winner_page_bytes);
        }
        noc_semaphore_inc(dst_sem_noc_addr, 1);
        noc_async_posted_writes_flushed();
        noc_async_atomic_barrier();
    } else {
        auto recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
        noc_semaphore_wait(recv_sem_ptr, expected_remote_incs);
        noc_semaphore_set(recv_sem_ptr, 0);

        auto out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_pair_addr);
        for (uint32_t batch = 0; batch < active_batch_size; ++batch) {
            uint16_t global_best_score = NEG_INF_BFLOAT16;
            uint32_t global_best_index = 0xFFFFFFFF;
            for (uint32_t slot = 0; slot < num_senders; ++slot) {
                const uint32_t candidate_slot_addr = gather_addr + (batch * num_senders + slot) * winner_page_bytes;
                auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_slot_addr);
                auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(candidate_slot_addr);
                const uint16_t score = slot_u16_ptr[0];
                const uint32_t index = slot_u32_ptr[1];
                if (is_better_candidate(score, index, global_best_score, global_best_index)) {
                    global_best_score = score;
                    global_best_index = index;
                }
            }
            write_winner_slot(output_pair_addr + batch * output_pair_page_bytes, global_best_score, global_best_index);
        }
    }
}
