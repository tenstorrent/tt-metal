// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sampling unified kernel (k=1 argmax fast path)
//
// Multi-core single-device scope:
// - Each active core computes local argmax over its 160 values.
// - Local winner is packed into one 16B page:
//   [bf16 score, uint32 index, garbage, garbage]
// - Non-final active cores posted-write their page to final core slot and
//   increment final-core semaphore.
// - Final core waits for all remote semaphore increments, then reduces all
//   gathered slots to one final index (tie-break: lowest index).

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "api/numeric/bfloat16.h"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
    static constexpr bool is_final_core = get_named_compile_time_arg_val("sampling_is_final_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t num_values = get_named_compile_time_arg_val("sampling_num_values");
    constexpr uint32_t winner_page_bytes = get_named_compile_time_arg_val("sampling_winner_page_bytes");
    constexpr uint32_t num_senders = get_named_compile_time_arg_val("sampling_num_senders");
    constexpr uint32_t expected_remote_incs = get_named_compile_time_arg_val("sampling_expected_remote_incs");
    constexpr uint32_t winner_cb = get_named_compile_time_arg_val("sampling_winner_cb");
    constexpr uint32_t gather_cb = get_named_compile_time_arg_val("sampling_gather_cb");
    constexpr uint32_t semaphore_id = get_named_compile_time_arg_val("sampling_receiver_semaphore_id");

    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t final_noc_x = get_common_arg_val<uint32_t>(3);
    const uint32_t final_noc_y = get_common_arg_val<uint32_t>(4);

    const uint32_t sender_idx = get_named_compile_time_arg_val("sampling_sender_idx");
    const uint32_t slot_offset = sender_idx * winner_page_bytes;
    const uint32_t gather_addr = get_write_ptr(gather_cb);

    if constexpr (Core::is_active_core) {
        auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
        auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_addr);

        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;

        for (uint32_t i = 0; i < num_values; ++i) {
            const uint16_t score = scores_ptr[i];

            if (bfloat16_greater(score, best_score)) {
                best_score = score;
                best_index = indices_ptr[i];
            }
        }

        if constexpr (Core::is_final_core) {
            auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gather_addr + slot_offset);
            auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr + slot_offset);
            slot_u16_ptr[0] = best_score;
            slot_u32_ptr[1] = best_index;
        } else {
            cb_reserve_back(winner_cb, 1);
            auto local_slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(winner_cb));
            auto local_slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(winner_cb));
            local_slot_u16_ptr[0] = best_score;
            local_slot_u32_ptr[1] = best_index;
            cb_push_back(winner_cb, 1);

            cb_wait_front(winner_cb, 1);
            const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
            const uint64_t dst_data_noc_addr = final_noc_base | (uint64_t)(gather_addr + slot_offset);
            const uint64_t dst_sem_noc_addr = final_noc_base | (uint64_t)(get_semaphore(semaphore_id));
            const uint32_t src_data_addr = get_read_ptr(winner_cb);

            noc_async_write_one_packet<true, true>(src_data_addr, dst_data_noc_addr, winner_page_bytes);
            noc_semaphore_inc(dst_sem_noc_addr, 1);
            noc_async_posted_writes_flushed();
            noc_async_atomic_barrier();
            cb_pop_front(winner_cb, 1);
        }
    }

    if constexpr (Core::is_final_core) {
        auto recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
        noc_semaphore_wait(recv_sem_ptr, expected_remote_incs);
        noc_semaphore_set(recv_sem_ptr, 0);

        auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_addr);
        uint16_t global_best_score = NEG_INF_BFLOAT16;
        uint32_t global_best_index = 0xFFFFFFFF;

        for (uint32_t slot = 0; slot < num_senders; ++slot) {
            const uint32_t slot_addr = gather_addr + slot * winner_page_bytes;
            auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
            auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
            const uint16_t score = slot_u16_ptr[0];
            if (bfloat16_greater(score, global_best_score)) {
                global_best_score = score;
                global_best_index = slot_u32_ptr[1];
            }
        }

        output_ptr[0] = global_best_index;
    }

#elif defined(COMPILE_FOR_BRISC)
    // No-op for k=1 argmax fast path.

#elif defined(COMPILE_FOR_TRISC)
    // No-op for k=1 argmax fast path.
#endif
}
