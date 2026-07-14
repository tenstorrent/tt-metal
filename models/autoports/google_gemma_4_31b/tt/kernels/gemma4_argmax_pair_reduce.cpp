// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

namespace {
constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
constexpr uint32_t pair_payload_bytes = get_compile_time_arg_val(1);
constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
constexpr uint32_t scratch_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t active_batch_size = get_compile_time_arg_val(4);
constexpr uint32_t max_batch_size = get_compile_time_arg_val(5);

FORCE_INLINE bool is_better_candidate(
    uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
    return bfloat16_greater(candidate_score, best_score) ||
           ((candidate_score == best_score) && (candidate_index < best_index));
}
}  // namespace

void kernel_main() {
    const uint32_t gathered_pairs_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_token_addr = get_common_arg_val<uint32_t>(1);
    constexpr auto pairs_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<pairs_args.next_compile_time_args_offset()>();
    constexpr uint32_t pair_page_bytes = pairs_args.get_aligned_page_size();
    constexpr uint32_t output_write_bytes =
        ((active_batch_size * sizeof(uint32_t) + NOC_DRAM_WRITE_ALIGNMENT_BYTES - 1) / NOC_DRAM_WRITE_ALIGNMENT_BYTES) *
        NOC_DRAM_WRITE_ALIGNMENT_BYTES;
    static_assert(pair_payload_bytes == 2 * sizeof(uint32_t));
    static_assert(output_write_bytes <= scratch_page_bytes * num_candidates);
    const auto output_accessor = TensorAccessor(output_args, output_token_addr);

    cb_reserve_back(scratch_cb, num_candidates);
    const uint32_t scratch_addr = get_write_ptr(scratch_cb);
    auto out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    for (uint32_t word = 0; word < output_write_bytes / sizeof(uint32_t); ++word) {
        out_ptr[word] = 0;
    }
    for (uint32_t batch = 0; batch < active_batch_size; ++batch) {
        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;
        for (uint32_t i = 0; i < num_candidates; ++i) {
            const uint32_t pair_addr = gathered_pairs_addr + (i * max_batch_size + batch) * pair_page_bytes;
            auto pair_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pair_addr);
            const uint16_t score = static_cast<uint16_t>(pair_u32[0] & 0xFFFF);
            const uint32_t index = pair_u32[1];
            if (is_better_candidate(score, index, best_score, best_index)) {
                best_score = score;
                best_index = index;
            }
        }
        out_ptr[batch] = best_index;
    }
    // Blackhole DRAM writes require matching 16-byte alignment between the
    // L1 source and DRAM destination. A per-token write to base+4 violates
    // that contract for batch row 1. Materialize and write the whole aligned
    // row-major page so fixed-slot token feedback remains watcher-clean.
    noc_async_write(scratch_addr, output_accessor.get_noc_addr(0), output_write_bytes);
    noc_async_write_barrier();
    cb_pop_front(scratch_cb, num_candidates);
}
