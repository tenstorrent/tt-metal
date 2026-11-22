// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, reader (one core of an unembed chunk's L1 width-sharded fp32 output).
// Packs row 0 of each of the core's ntiles local tiles into dense pages (tile j -> elements 32j..32j+31, the rest
// of the last page zeroed) and reads the two softcap scalar tiles (1/cap, cap) from DRAM.
// Runtime args: shard L1 address, scalar tensor address.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_in = 0, cb_s = 1;

    const uint32_t shard = get_arg_val<uint32_t>(0);
    const uint32_t scal = get_arg_val<uint32_t>(1);

    cb_reserve_back(cb_s, 2);
    rowio::read_pages(scal, 4096, 0, 2, get_write_ptr(cb_s));

    cb_reserve_back(cb_in, T);
    const uint32_t dst = get_write_ptr(cb_in);
    rowio::zero_words(dst + ntiles * 128, T * 4096 - ntiles * 128);
    for (uint32_t j = 0; j < ntiles; ++j) {  // fp32 row 0: 16 values in face 0, 16 in face 1 (+1024 B)
        noc_async_read(get_noc_addr(shard + j * 4096), dst + j * 128, 64);
        noc_async_read(get_noc_addr(shard + j * 4096 + 1024), dst + j * 128 + 64, 64);
    }
    noc_async_read_barrier();
    cb_push_back(cb_s, 2);
    cb_push_back(cb_in, T);
}
