// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read micro-benchmark (test_dram_read_bench.py): stream N_CHUNKS chunks of CHUNK_BYTES from one DRAM bank, chunk
// c at offset + c * STRIDE_BYTES, BATCH chunks per read barrier, into an L1 ring of SLOTS chunks (or, PUSH, into CB
// pages for bench_fwd.cpp on the other RISC). The bytes are never consumed: delivery bandwidth only.
//
// ROTATE: chunk c comes from bank (bank id + c) % 8 at offset + (c / 8) * STRIDE_BYTES (the core's region is spread
// over all 8 banks, like unified_routed_expert_ffn's one-tile-row ND shards), so every core rotates over the banks.
//
// CT: 0 CB, 1 CHUNK_BYTES, 2 N_CHUNKS, 3 BATCH, 4 SLOTS (multiple of BATCH), 5 STRIDE_BYTES, 6 PUSH, 7 ROTATE
// RT: 0 bank base address, 1 bank id, 2 byte offset in the bank

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);
    constexpr uint32_t stride = get_compile_time_arg_val(5);
    constexpr bool push = get_compile_time_arg_val(6) != 0;
    constexpr bool rotate = get_compile_time_arg_val(7) != 0;
    static_assert(slots % batch == 0, "slots must be a multiple of batch");

    const uint32_t bank0 = get_arg_val<uint32_t>(1), addr0 = get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank0, addr0);
    const uint32_t ring = get_write_ptr(cb);
    for (uint32_t c = 0, b = 0; c < n; c += batch, ++b) {
        const uint32_t nb = n - c < batch ? n - c : batch;
        if constexpr (push) {
            cb_reserve_back(cb, nb);
        }
        const uint32_t dst = push ? get_write_ptr(cb) : ring + (b % (slots / batch)) * batch * chunk;
        for (uint32_t i = 0; i < nb; ++i) {
            if constexpr (rotate) {
                const uint32_t k = c + i;
                noc_async_read(get_noc_addr_from_bank_id<true>((bank0 + k) % 8, addr0 + (k / 8) * stride), dst + i * chunk, chunk);
            } else {
                noc_async_read(src + uint64_t(c + i) * stride, dst + i * chunk, chunk);
            }
        }
        noc_async_read_barrier();
        if constexpr (push) {
            cb_push_back(cb, nb);
        }
    }
}
