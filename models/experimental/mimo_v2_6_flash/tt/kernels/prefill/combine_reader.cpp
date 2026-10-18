// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC of the prefill MoE combine (CombineK in tt/moe_prefill.py). Item i is token tile i / (nt / block) and column
// block i % (nt / block) (block tile columns, block * 64 B of a row): for each k, the 32 tokens' rows k * T + t of the
// k-major row-major expert output, that column block each, land as 32 rows (block tile-sized pages of cb_in).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t nt = get_compile_time_arg_val(3);  // hidden / 32
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t seg_bytes = block * 64;

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> y_gen = {.bank_base_address = y_addr, .page_size = nt * 64};
    for (uint32_t i = first; i < first + count; i++) {
        const uint32_t tt = i / (nt / block);
        const uint32_t col_bytes = (i % (nt / block)) * seg_bytes;
        for (uint32_t k = 0; k < K; k++) {
            cb_reserve_back(cb_in, block);
            uint32_t dst = get_write_ptr(cb_in);
            for (uint32_t r = 0; r < 32; r++) {
                noc_async_read(y_gen.get_noc_addr(k * T + tt * 32 + r, col_bytes), dst, seg_bytes);
                dst += seg_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block);
        }
    }
}
