// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 2 (tt/device_sampler.py): one user's top-K logits of this chip's vocab shard and its
// normalizer, from the scan cores' partials. One core per user.
//
// Inputs: the top-K partials (sample_scan.cpp, sample_scan_writer.cpp): core c's K (key, local index) pairs of user u,
// best first, in its block of `slots` user slots (block = slots * K * 8 bytes, 4096 / block per 4096-byte page) at
// byte offset u * K * 8; the sums partials (sample_scan_writer.cpp): core c's sum of exp((logit - its largest) / T) of
// user u, uint64 Q30 at byte offset (c * slots + u) * 8; params (uint32 row-major [1, max_batch * 8], row u's word 0 =
// log2(e) / T as fp32 bits).
// Output, in the chip candidates page (uint32, 4096 bytes): user u's K pairs sorted by key descending, lower index
// first among equal keys, at byte offset u * K * 8; and the chip's sum of exp((logit - the chip's largest) / T),
// uint64 Q30, at byte offset 2048 + u * 16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sample_fixed.hpp"

void kernel_main() {
    const uint32_t partials_addr = get_arg_val<uint32_t>(0);
    const uint32_t sums_addr = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t cands_addr = get_arg_val<uint32_t>(3);
    const uint32_t u = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_cores = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t max_batch = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);  // user slots of a scan core's block: 4, 8 or 16
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t block_bytes = slots * slot_bytes;
    constexpr uint32_t blocks_per_page = 4096 / block_bytes;
    constexpr uint32_t sums_bytes = num_cores * slots * 8;
    constexpr uint32_t params_bytes = max_batch * 32;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // slots, sums, params, the sorted list, the sum chunk
    const uint32_t sums_l1 = scratch + num_cores * slot_bytes;
    const uint32_t params_l1 = sums_l1 + sums_bytes;
    const uint32_t list_l1 = params_l1 + params_bytes;
    const uint32_t sum_l1 = list_l1 + slot_bytes;

    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t offset = (c % blocks_per_page) * block_bytes + u * slot_bytes;
        noc_async_read(get_noc_addr(c / blocks_per_page, part_gen, offset), scratch + c * slot_bytes, slot_bytes);
    }
    const InterleavedAddrGen<false> sums_gen = {.bank_base_address = sums_addr, .page_size = 4096};
    noc_async_read(get_noc_addr(0, sums_gen), sums_l1, sums_bytes);
    const InterleavedAddrGen<true> params_gen = {.bank_base_address = params_addr, .page_size = params_bytes};
    noc_async_read(get_noc_addr(0, params_gen), params_l1, params_bytes);
    noc_async_read_barrier();

    // Each core's list is sorted: a pair below the current K-th key ends that list's useful part.
    const uint32_t* in = reinterpret_cast<const uint32_t*>(scratch);
    uint32_t keys[K];
    uint32_t ids[K];
    uint32_t n = 0;
    for (uint32_t c = 0; c < num_cores; ++c) {
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t key = in[2 * (c * K + j)];
            const uint32_t idx = in[2 * (c * K + j) + 1];
            if (n == K && !sample::better(key, idx, keys[K - 1], ids[K - 1])) {
                break;
            }
            uint32_t p = n < K ? n++ : K - 1;
            while (p > 0 && sample::better(key, idx, keys[p - 1], ids[p - 1])) {
                keys[p] = keys[p - 1];
                ids[p] = ids[p - 1];
                --p;
            }
            keys[p] = key;
            ids[p] = idx;
        }
    }

    // The chip's normalizer relative to its largest logit: each core's sum is relative to the core's largest.
    const uint32_t* params = reinterpret_cast<const uint32_t*>(params_l1);
    const sample::Rel rel(params[u * 8], sample::key_bits(keys[0]));
    const uint64_t* sums = reinterpret_cast<const uint64_t*>(sums_l1);
    uint64_t total = 0;
    for (uint32_t c = 0; c < num_cores; ++c) {
        total += sample::mul_q30(sums[c * slots + u], rel(sample::key_bits(in[2 * c * K])));
    }

    uint32_t* list = reinterpret_cast<uint32_t*>(list_l1);
    for (uint32_t j = 0; j < K; ++j) {
        list[2 * j] = keys[j];
        list[2 * j + 1] = ids[j];
    }
    uint64_t* sum = reinterpret_cast<uint64_t*>(sum_l1);
    sum[0] = total;
    sum[1] = 0;

    const InterleavedAddrGen<false> cands_gen = {.bank_base_address = cands_addr, .page_size = 4096};
    noc_async_write(list_l1, get_noc_addr(0, cands_gen, u * slot_bytes), slot_bytes);
    // 16 slots: the lists fill the first page, the normalizers go to the second
    constexpr bool two_pages = slots * slot_bytes > 2048;
    noc_async_write(sum_l1, get_noc_addr(two_pages ? 1 : 0, cands_gen, (two_pages ? 0 : 2048) + u * 16), 16);
    noc_async_write_barrier();
}
