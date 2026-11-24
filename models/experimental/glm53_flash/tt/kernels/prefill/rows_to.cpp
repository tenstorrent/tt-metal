// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 P4 (tt/prefill_tp.py rows_to, the batched prefill's head rows): scatter ROW_MAJOR rows by a device index
// table: dst row idx[r] <- src row r for r in [first, first + count) of this core (idx[r] == 0xFFFFFFFF: skipped), so
// one captured program serves any slot assignment (the host writes the table before every replay). One page per row
// (ROW_BYTES) in both tensors (interleaved DRAM); idx = one page of uint32 (IDX_BYTES).
//
// Compile-time: ROW_BYTES, DEPTH (rows in flight), IDX_BYTES. Runtime per core: src, dst, idx addresses, first row,
// row count. CB 0: IDX_BYTES rounded up to 64 B + DEPTH x ROW_BYTES -- the table and the row buffer each start at a
// 64-B multiple of the CB base (a Blackhole DRAM read needs NOC_DRAM_READ_ALIGNMENT_BYTES = 64: expand_rows, 10-21).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(1);
    constexpr uint32_t IDX_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t IDX_ALIGNED = (IDX_BYTES + 63) / 64 * 64;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    if (count == 0) {
        return;
    }

    const uint32_t base = get_write_ptr(0);
    const InterleavedAddrGen<true> idx_g{.bank_base_address = idx_addr, .page_size = IDX_BYTES};
    noc_async_read(idx_g.get_noc_addr(0), base, IDX_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t buf = base + IDX_ALIGNED;
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = ROW_BYTES};

    const uint32_t end = first + count;
    for (uint32_t r = first; r < end;) {
        const uint32_t n = (end - r) < DEPTH ? (end - r) : DEPTH;
        for (uint32_t k = 0; k < n; ++k) {
            if (idx[r + k] != SKIP) {
                noc_async_read(src.get_noc_addr(r + k), buf + k * ROW_BYTES, ROW_BYTES);
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t d = idx[r + k];
            if (d != SKIP) {
                noc_async_write(buf + k * ROW_BYTES, dst.get_noc_addr(d), ROW_BYTES);
            }
        }
        noc_async_write_barrier();
        r += n;
    }
}
