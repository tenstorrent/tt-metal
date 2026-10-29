// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill -> decode conv-history handoff (BRISC, one core per GDN copy range, every chip). The host packs
// this chip's history columns compactly: page ((g * 3 + slot) * K + k) holds conv tile k (its 32 columns as
// a rows x 32 tile) for GDN copy g and ring slot `slot`. The decode's history tensor (row-major, one page per
// (streamer, copy) row of `slots` slots of slot_elems elements) takes it at row rows[k] * copies + g, element
// slot * slot_elems + tiles[k] * 32 * rows. Batch-1 decode: 1 row, 3 slots; the verify step: 2 rows (its
// pair slots), 4 slots of which the first 3 are written.
//
// Compile-time args: 0 K, 1 copies, 2 rows per tile, 3 slots per row, 4 slot_elems. Runtime args: 0 compact address,
// 1 history address, 2 first copy, 3 copies of this core, then K x (row, tile).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t copies = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2), slots = get_compile_time_arg_val(3);
    constexpr uint32_t slot_elems = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = 64 * rows, row_bytes = slots * slot_elems * 2, cb_buf = 0;
    const uint32_t compact = get_arg_val<uint32_t>(0), hist = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<true> src{.bank_base_address = compact, .page_size = chunk};
    const InterleavedAddrGen<true> dst{.bank_base_address = hist, .page_size = row_bytes};
    const uint32_t buf = get_write_ptr(cb_buf);

    for (uint32_t g = first; g < first + count; g++) {
        for (uint32_t slot = 0; slot < 3; slot++) {
            for (uint32_t k = 0; k < K; k++) {
                noc_async_read(get_noc_addr((g * 3 + slot) * K + k, src), buf + k * chunk, chunk);
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t row = get_arg_val<uint32_t>(4 + 2 * k), tile = get_arg_val<uint32_t>(5 + 2 * k);
                const uint32_t off = (slot * slot_elems + tile * 32 * rows) * 2;
                noc_async_write(buf + k * chunk, get_noc_addr(row * copies + g, dst, off), chunk);
            }
            noc_async_write_barrier();
        }
    }
}
