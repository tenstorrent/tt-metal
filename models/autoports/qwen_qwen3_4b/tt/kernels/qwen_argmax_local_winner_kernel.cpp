// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

namespace {

constexpr uint32_t num_values = get_named_compile_time_arg_val("num_values");
constexpr uint32_t winner_page_bytes = get_named_compile_time_arg_val("winner_page_bytes");
constexpr uint32_t num_senders = get_named_compile_time_arg_val("num_senders");
constexpr uint32_t expected_remote_incs = get_named_compile_time_arg_val("expected_remote_incs");
constexpr uint32_t receiver_semaphore_addr = get_named_compile_time_arg_val("receiver_semaphore_addr");
constexpr uint32_t gather_cb = get_named_compile_time_arg_val("gather_cb");
constexpr uint32_t sender_idx = get_named_compile_time_arg_val("sender_idx");
constexpr bool is_final_core = get_named_compile_time_arg_val("is_final_core") == 1;

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

}  // namespace

void kernel_main() {
    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t output_pair_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t final_noc_x = get_common_arg_val<uint32_t>(3);
    const uint32_t final_noc_y = get_common_arg_val<uint32_t>(4);

    const uint32_t gather_addr = get_write_ptr(gather_cb);
    const uint32_t slot_addr = gather_addr + sender_idx * winner_page_bytes;

    auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
    auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_addr);

    uint16_t best_score = NEG_INF_BFLOAT16;
    uint32_t best_index = 0xFFFFFFFF;
    for (uint32_t i = 0; i < num_values; ++i) {
        const uint16_t score = scores_ptr[i];
        const uint32_t index = indices_ptr[i];
        if (is_better_candidate(score, index, best_score, best_index)) {
            best_score = score;
            best_index = index;
        }
    }

    write_winner_slot(slot_addr, best_score, best_index);
    if constexpr (!is_final_core) {
        const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
        const uint64_t dst_data_noc_addr = final_noc_base | static_cast<uint64_t>(slot_addr);
        const uint64_t dst_sem_noc_addr = final_noc_base | static_cast<uint64_t>(receiver_semaphore_addr);
        noc_async_write_one_packet<true, true>(slot_addr, dst_data_noc_addr, winner_page_bytes);
        noc_semaphore_inc(dst_sem_noc_addr, 1);
        noc_async_posted_writes_flushed();
        noc_async_atomic_barrier();
    } else {
        auto recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
        noc_semaphore_wait(recv_sem_ptr, expected_remote_incs);
        noc_semaphore_set(recv_sem_ptr, 0);

        uint16_t global_best_score = NEG_INF_BFLOAT16;
        uint32_t global_best_index = 0xFFFFFFFF;
        for (uint32_t slot = 0; slot < num_senders; ++slot) {
            const uint32_t candidate_slot_addr = gather_addr + slot * winner_page_bytes;
            auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_slot_addr);
            auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(candidate_slot_addr);
            const uint16_t score = slot_u16_ptr[0];
            const uint32_t index = slot_u32_ptr[1];
            if (is_better_candidate(score, index, global_best_score, global_best_index)) {
                global_best_score = score;
                global_best_index = index;
            }
        }

        auto out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_pair_addr);
        out_ptr[0] = static_cast<uint32_t>(global_best_score);
        out_ptr[1] = global_best_index;
    }
}
