// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Loop-pass message split, one data-movement kernel per output tile column j: the [1, 1, 48, C] fp32
// messages (row 4 l + i = message i of loop layer l: q, k, v, mlp) -> per layer a qkv row tensor [1, 1, 3, C] and
// an mlp row tensor [1, 1, 1, C], each with its dense twin. Replaces 24 selector matmuls per pass and lets the
// message gates read 3 dense pages instead of gathering each message row.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");  // C / 32
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");  // dense pages per vector
    constexpr uint32_t n_layers = get_named_compile_time_arg_val("n_layers");
    constexpr uint32_t cb_in = 0;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);  // tile column
    // common args, per layer l: qkv row, qkv dense, mlp row, mlp dense addresses
    const InterleavedAddrGen<true> src = {.bank_base_address = in_addr, .page_size = 4096};
    const uint32_t l1 = get_write_ptr(cb_in);
    constexpr uint32_t n_tr = (4 * n_layers + 31) / 32;  // tile rows of the input
    for (uint32_t tr = 0; tr < n_tr; ++tr) {
        noc_async_read(src.get_noc_addr(tr * ntc + j), l1 + tr * 4096, 4096);
    }
    noc_async_read_barrier();
    const uint32_t e = 32 * j;  // first element of this tile column
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t v = 4 * l + i;
            const uint32_t s = l1 + (v / 32) * 4096 + rowio::row_offset(v % 32, 4);
            const bool mlp = i == 3;
            const uint32_t row = mlp ? 0 : i;
            const InterleavedAddrGen<true> rd = {
                .bank_base_address = get_common_arg_val<uint32_t>(4 * l + (mlp ? 2 : 0)), .page_size = 4096};
            const InterleavedAddrGen<true> dd = {
                .bank_base_address = get_common_arg_val<uint32_t>(4 * l + (mlp ? 3 : 1)), .page_size = 4096};
            const uint32_t ro = rowio::row_offset(row, 4);
            noc_async_write(s, rd.get_noc_addr(j, ro), 64);
            noc_async_write(s + 1024, rd.get_noc_addr(j, ro + 1024), 64);
            const uint64_t d = dd.get_noc_addr(row * T + e / 1024, (e % 1024) * 4);
            noc_async_write(s, d, 64);
            noc_async_write(s + 1024, d + 64, 64);
        }
    }
    noc_async_write_barrier();
}
