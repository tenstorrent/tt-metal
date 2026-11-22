// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE combine, reader. Unit = (tile row R, column block j of CW tiles). Per unit: the zero tile (once),
// the K weight tiles wk (k, R) (bf16 [1, K, T, 1] TILE: column 0 = the token weights), then per k the 32 expert
// output rows y[back[k, 32 R + m]] (bf16 ROW_MAJOR, HIDDEN wide), columns [32 CW j, 32 CW (j + 1)), as one
// row-major 32 x 32 CW block.
// Runtime args: y, back, wk, zero, n_units, (R, j) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("t");
    constexpr uint32_t K = get_named_compile_time_arg_val("k");
    constexpr uint32_t CW = get_named_compile_time_arg_val("cw");
    constexpr uint32_t ROW = get_named_compile_time_arg_val("row");  // y row bytes
    constexpr uint32_t SEG = CW * 64;                                // bytes of one row segment
    constexpr uint32_t cb_rm = 0, cb_w = 1, cb_z = 2;
    const uint32_t y = get_arg_val<uint32_t>(0), back = get_arg_val<uint32_t>(1), wk = get_arg_val<uint32_t>(2);
    const uint32_t zero = get_arg_val<uint32_t>(3), n_units = get_arg_val<uint32_t>(4);
    const InterleavedAddrGen<true> ya = {.bank_base_address = y, .page_size = ROW};
    const InterleavedAddrGen<true> ba = {.bank_base_address = back, .page_size = 64};
    const InterleavedAddrGen<true> wa = {.bank_base_address = wk, .page_size = 2048};
    const InterleavedAddrGen<true> za = {.bank_base_address = zero, .page_size = 2048};
    cb_reserve_back(cb_z, 1);
    noc_async_read(za.get_noc_addr(0), get_write_ptr(cb_z), 2048);
    noc_async_read_barrier();
    cb_push_back(cb_z, 1);
    // back rows of this unit's tokens, per k (128 bytes each), staged in the top of the rm CB's spare page
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(5 + 2 * u), j = get_arg_val<uint32_t>(6 + 2 * u);
        cb_reserve_back(cb_w, K);
        const uint32_t lw = get_write_ptr(cb_w);
        for (uint32_t k = 0; k < K; ++k) {
            noc_async_read(wa.get_noc_addr(k * (T / 32) + R), lw + k * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, K);
        for (uint32_t k = 0; k < K; ++k) {
            cb_reserve_back(cb_rm, CW);
            const uint32_t l = get_write_ptr(cb_rm);
            // the 32 buffer rows of (k, tokens 32 R ..): 128 bytes of back, read into the block's last row slot
            // first (overwritten by that row's data below, after the ids are taken)
            volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l + 31 * SEG);
            noc_async_read(ba.get_noc_addr(0, (k * T + 32 * R) * 4), l + 31 * SEG, 128);
            noc_async_read_barrier();
            uint32_t id[32];
            for (uint32_t m = 0; m < 32; ++m) {
                id[m] = ids[m];
            }
            for (uint32_t m = 0; m < 32; ++m) {
                noc_async_read(ya.get_noc_addr(id[m], j * SEG), l + m * SEG, SEG);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, CW);
        }
    }
}
