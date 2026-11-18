// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Greedy acceptance of one verify pass (tt/verify_sampler.py GreedyVerify; host mirror reference/sample_verify.py
// verify_pass on an all-greedy control page), one core: the accept step of a verify whose slots the device sampler
// does not serve. Each request of the pass ("group") holds its last committed token and its n drafts in consecutive
// slots s0 .. s0 + n; it accepts draft j while it equals argmax[s0 + j] and commits the argmax chain, as
// sample_verify.cpp's greedy groups (the group's greedy word is not read: every group is greedy).
//
// Inputs (one-page interleaved DRAM):
// - argmax: uint32 row-major [1, max_batch], each slot's greedy token (GreedyHead's).
// - ctrl: uint32 row-major [1, groups * group_words], per group: first slot (0xFFFFFFFF: unused), drafts n, greedy,
//   unused, drafts [D], then the sampled groups' uniforms (not read).
// Output: uint32 row-major [1, groups * out_words], per group: committed count (0: unused group), the committed tokens
// [D + 1] (0xFFFFFFFF past the count), truncated (0), unused.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint64_t dram_addr(uint32_t base, uint32_t page_size) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(0, g);
}

constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63u; }

constexpr uint32_t kNone = 0xFFFFFFFFu;

}  // namespace

void kernel_main() {
    const uint32_t argmax_addr = get_arg_val<uint32_t>(0);
    const uint32_t ctrl_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t max_batch = get_compile_time_arg_val(0);
    constexpr uint32_t groups = get_compile_time_arg_val(1);
    constexpr uint32_t D = get_compile_time_arg_val(2);  // drafts per group at most
    constexpr uint32_t group_words = get_compile_time_arg_val(3);
    constexpr uint32_t out_words = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(5);
    static_assert(groups <= max_batch && D + 1 <= max_batch && 4 + D <= group_words && 3 + D <= out_words);
    constexpr uint32_t argmax_bytes = max_batch * 4;
    constexpr uint32_t ctrl_bytes = groups * group_words * 4;
    constexpr uint32_t out_bytes = groups * out_words * 4;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t argmax_l1 = get_write_ptr(cb_scratch);
    const uint32_t ctrl_l1 = argmax_l1 + align64(argmax_bytes);
    const uint32_t out_l1 = ctrl_l1 + align64(ctrl_bytes);
    noc_async_read(dram_addr(argmax_addr, argmax_bytes), argmax_l1, argmax_bytes);
    noc_async_read(dram_addr(ctrl_addr, ctrl_bytes), ctrl_l1, ctrl_bytes);
    noc_async_read_barrier();

    const uint32_t* argmax = reinterpret_cast<const uint32_t*>(argmax_l1);
    const uint32_t* ctrl = reinterpret_cast<const uint32_t*>(ctrl_l1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t* c = ctrl + g * group_words;
        volatile tt_l1_ptr uint32_t* o = out + g * out_words;
        o[0] = 0;
        for (uint32_t j = 0; j <= D; ++j) {
            o[1 + j] = kNone;
        }
        for (uint32_t j = 2 + D; j < out_words; ++j) {
            o[j] = 0;
        }
        const uint32_t first = c[0];
        const uint32_t n = c[1];
        if (first == kNone || first + n >= max_batch || n > D) {
            continue;
        }
        const uint32_t* drafts = c + 4;
        uint32_t m = n;
        for (uint32_t j = 0; j < n; ++j) {
            o[1 + j] = argmax[first + j];
            if (argmax[first + j] != drafts[j]) {
                m = j;
                break;
            }
        }
        if (m == n) {
            o[1 + n] = argmax[first + n];
        }
        o[0] = m + 1;
    }
    noc_async_write(out_l1, dram_addr(out_addr, out_bytes), out_bytes);
    noc_async_write_barrier();
}
