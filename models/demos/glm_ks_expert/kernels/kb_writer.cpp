// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 1 bench writer (NCRISC): drains NSB sub-blocks of partials (CB 17, P tiles each; 0: none) and h (CB 16, H
// tiles each), and writes the last sub-block's tiles to DRAM (h, partials).
// RT: 0 h addr, 1 partials addr, 2 NSB, 3 H, 4 P. CT: TensorAccessorArgs(h), TensorAccessorArgs(partials).
#include "api/dataflow/dataflow_api.h"

template <typename S>
FORCE_INLINE void drain(uint32_t cb, const S& s, uint32_t n, bool write) {
    cb_wait_front(cb, n);
    if (write) {
        uint32_t l1 = get_read_ptr(cb);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_page(i, s, l1);
            l1 += get_tile_size(cb);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb, n);
}

void kernel_main() {
    constexpr auto ah = TensorAccessorArgs<0>();
    constexpr auto ap = TensorAccessorArgs<ah.next_compile_time_args_offset()>();
    const auto sh = TensorAccessor(ah, get_arg_val<uint32_t>(0), get_tile_size(16));
    const auto sp = TensorAccessor(ap, get_arg_val<uint32_t>(1), get_tile_size(17));
    const uint32_t nsb = get_arg_val<uint32_t>(2);
    const uint32_t nh = get_arg_val<uint32_t>(3);
    const uint32_t np = get_arg_val<uint32_t>(4);
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        const bool last = sb + 1 == nsb;
        if (np) {
            drain(17, sp, np, last);
        }
        drain(16, sh, nh, last);
    }
}
