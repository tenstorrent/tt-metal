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
    // wt8: wk is the router's [1, 1, T, K] bf16 TILE weights (tile R, column k = the weights of k); the K column-0
    // weight tiles are built here from that one tile (the entries outside column 0 are not read by compute)
    constexpr uint32_t WT8 = get_named_compile_time_arg_val("wt8");
    constexpr uint32_t cb_rm = 0, cb_w = 1, cb_z = 2, cb_id = 5, cb_wt = 6;
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
    // per unit: the K weight tiles and the K x 32 buffer row ids (back, 128 bytes per k) in one round trip, then
    // per k the 32 row segments, pushed when they land (cb_rm holds NB blocks, so reads run ahead of compute)
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(5 + 2 * u), j = get_arg_val<uint32_t>(6 + 2 * u);
        cb_reserve_back(cb_w, K);
        const uint32_t lw = get_write_ptr(cb_w);
        if (WT8) {
            noc_async_read(wa.get_noc_addr(R), get_write_ptr(cb_wt), 2048);
        }
        for (uint32_t k = 0; k < K; ++k) {
            if (!WT8) {
                noc_async_read(wa.get_noc_addr(k * (T / 32) + R), lw + k * 2048, 2048);
            }
            noc_async_read(ba.get_noc_addr(0, (k * T + 32 * R) * 4), get_write_ptr(cb_id) + k * 128, 128);
        }
        noc_async_read_barrier();
        if (WT8) {  // element (r, c) of a bf16 tile: face 2 (r / 16) + c / 16, index 16 (r % 16) + c % 16
            const volatile tt_l1_ptr uint16_t* src =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_wt));
            for (uint32_t k = 0; k < K; ++k) {
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lw + k * 2048);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t at = (r >> 4) * 512 + (r & 15) * 16;
                    dst[at] = src[at + k];
                }
            }
        }
        cb_push_back(cb_w, K);
        for (uint32_t k = 0; k < K; ++k) {
            cb_reserve_back(cb_rm, CW);
            const uint32_t l = get_write_ptr(cb_rm);
            for (uint32_t m = 0; m < 32; ++m) {
                noc_async_read(ya.get_noc_addr(ids[32 * k + m], j * SEG), l + m * SEG, SEG);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, CW);
        }
    }
}
