// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

namespace {

constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
constexpr uint32_t pair_payload_bytes = get_compile_time_arg_val(1);
constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
constexpr uint32_t scratch_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t active_batch_size = get_compile_time_arg_val(4);

FORCE_INLINE bool is_better_candidate(
    uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
    return bfloat16_greater(candidate_score, best_score) ||
           ((candidate_score == best_score) && (candidate_index < best_index));
}

}  // namespace

void kernel_main() {
    const uint32_t gathered_pairs_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_token_addr = get_common_arg_val<uint32_t>(1);

    constexpr auto pairs_args = TensorAccessorArgs<5>();
    constexpr auto output_args = TensorAccessorArgs<pairs_args.next_compile_time_args_offset()>();
    constexpr uint32_t pair_page_bytes = pairs_args.get_aligned_page_size();
    static_assert(pair_payload_bytes <= pair_page_bytes);
    static_assert(pair_page_bytes <= scratch_page_bytes);
    const auto pairs_accessor = TensorAccessor(pairs_args, gathered_pairs_addr);
    const auto output_accessor = TensorAccessor(output_args, output_token_addr);

    cb_reserve_back(scratch_cb, num_candidates);
    const uint32_t scratch_addr = get_write_ptr(scratch_cb);

    for (uint32_t batch = 0; batch < active_batch_size; ++batch) {
        for (uint32_t i = 0; i < num_candidates; ++i) {
            const uint64_t src_noc_addr = pairs_accessor.get_noc_addr(i * active_batch_size + batch);
            noc_async_read(src_noc_addr, scratch_addr + i * scratch_page_bytes, pair_page_bytes);
        }
        noc_async_read_barrier();

        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;
        for (uint32_t i = 0; i < num_candidates; ++i) {
            const uint32_t pair_addr = scratch_addr + i * scratch_page_bytes;
            auto pair_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pair_addr);
            const uint16_t score = static_cast<uint16_t>(pair_u32[0] & 0xFFFF);
            const uint32_t index = pair_u32[1];
            if (is_better_candidate(score, index, best_score, best_index)) {
                best_score = score;
                best_index = index;
            }
        }

        auto out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
        out_ptr[0] = best_index;
        noc_async_write(scratch_addr, output_accessor.get_noc_addr(0) + batch * sizeof(uint32_t), sizeof(uint32_t));
        noc_async_write_barrier();
    }
    cb_pop_front(scratch_cb, num_candidates);
}
