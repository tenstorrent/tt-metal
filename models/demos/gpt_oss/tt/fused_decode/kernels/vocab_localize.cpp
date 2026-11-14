// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill token ids -> local ids of this device's vocab shard (fused_decode/embed.py VocabLocalize), one core (BRISC):
// out[i] = tok[i] - off when that is < vd (the id is in this device's rows), else vd (the zero row of the local table).
// tok / out: [1, 1, 1, S] UINT32 row-major (one page of S ids); off: [1, 1, 1, 32] UINT32, entry 0 = d * vd.
//
// runtime args: [tok_addr, off_addr, out_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t tok_addr = get_arg_val<uint32_t>(0);
    const uint32_t off_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t vd = get_compile_time_arg_val(3);
    constexpr auto tok_args = TensorAccessorArgs<4>();
    constexpr auto off_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();

    const auto s_tok = TensorAccessor(tok_args, tok_addr, S * 4);
    const auto s_off = TensorAccessor(off_args, off_addr, 128);
    const auto s_out = TensorAccessor(out_args, out_addr, S * 4);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_off.get_noc_addr(0), scr, 128);
    noc_async_read_barrier();
    const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[0];
    const uint32_t buf = scr + 128;
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t i = 0; i < S; i += chunk) {
        const uint32_t n = S - i < chunk ? S - i : chunk;
        noc_async_read(s_tok.get_noc_addr(0, i * 4), buf, n * 4);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t local = ids[k] - off;  // wraps for ids below this shard: then >= vd
            ids[k] = local < vd ? local : vd;
        }
        noc_async_write(buf, s_out.get_noc_addr(0, i * 4), n * 4);
        noc_async_write_barrier();
    }
}
