// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 P3 (tt/prefill_tp.py rows_at, the sparse_sdpa DSA prefill): copy a chunk's ROW_MAJOR rows into a ROW_MAJOR
// store at the chunk's position: dst row (start + r) <- src row r for r in [first, first + count) of this core, where
// start is read from a one-element int32 device tensor (the chunk start), so one captured program serves every chunk
// position. One page per row (ROW_BYTES) in both tensors (interleaved DRAM).
//
// Compile-time: ROW_BYTES, DEPTH (rows in flight). Runtime per core: src, dst, start addresses, first row, row count.
// CB 0: 64 B (the start value) + DEPTH x ROW_BYTES.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    if (count == 0) {
        return;
    }

    const uint32_t base = get_write_ptr(0);
    const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = 4};
    noc_async_read(start_g.get_noc_addr(0), base, 4);
    noc_async_read_barrier();
    const int32_t s = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base)[0];
    const uint32_t start = s < 0 ? 0u : static_cast<uint32_t>(s);
    const uint32_t buf = base + 64;
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = ROW_BYTES};

    const uint32_t end = first + count;
    for (uint32_t r = first; r < end;) {
        const uint32_t n = (end - r) < DEPTH ? (end - r) : DEPTH;
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read(src.get_noc_addr(r + k), buf + k * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_write(buf + k * ROW_BYTES, dst.get_noc_addr(start + r + k), ROW_BYTES);
        }
        noc_async_write_barrier();
        r += n;
    }
}
