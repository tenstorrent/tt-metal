// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 GLM53_PF_LAT_BOUND (tt/prefill_tp.py SparseCtx.local_latents_rm): the BQ attention's key store bounded to
// the VISIBLE local blocks. Reader: the device word (this chip's visible local shard units, ChunkInputs.pool_units_u;
// x BPU blocks per unit) clamps this core's static block range; the count goes to compute through CB_META; then every
// visible block's TROWS tile rows (TPR tiles each) are read from the paged TILE cache through the slot's LOCAL table.
// Blocks are STRIDED over the cores (core i: blocks i, i + NC, ...) so the visible prefix spreads over all of them.
// Compile-time: TILE_BYTES, TABLE_BYTES, TABLE_ALIGNED, TPR, TROWS, TPP, BPU. Runtime: cache, table, bound, core i,
// total local blocks NB, core count NC.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t TABLE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t TABLE_ALIGNED = get_compile_time_arg_val(2);
    constexpr uint32_t TPR = get_compile_time_arg_val(3);
    constexpr uint32_t TROWS = get_compile_time_arg_val(4);
    constexpr uint32_t TPP = get_compile_time_arg_val(5);
    constexpr uint32_t BPU = get_compile_time_arg_val(6);
    constexpr uint32_t CB_IN = 0, CB_META = 2, CB_SCR = 3;
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t bound_addr = get_arg_val<uint32_t>(2);
    const uint32_t ci = get_arg_val<uint32_t>(3);
    const uint32_t nb = get_arg_val<uint32_t>(4);
    const uint32_t nc = get_arg_val<uint32_t>(5);

    const uint32_t scr = get_write_ptr(CB_SCR);
    const InterleavedAddrGen<true> bound_g{.bank_base_address = bound_addr, .page_size = 4};
    noc_async_read(bound_g.get_noc_addr(0), scr, 64);  // its own 64-byte slot (BH DRAM read alignment)
    noc_async_read_barrier();
    uint32_t lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[0] * BPU;
    lim = lim < nb ? lim : nb;
    const uint32_t n = lim <= ci ? 0u : (lim - ci + nc - 1) / nc;  // this core's blocks ci, ci + nc, .. < lim
    cb_reserve_back(CB_META, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META))[0] = n;
    cb_push_back(CB_META, 1);
    if (n == 0) {
        return;
    }
    const InterleavedAddrGen<true> table_g{.bank_base_address = table_addr, .page_size = TABLE_BYTES};
    noc_async_read(table_g.get_noc_addr(0), scr + 64, TABLE_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + 64);
    const InterleavedAddrGen<true> src{.bank_base_address = cache_addr, .page_size = TILE_BYTES};
    for (uint32_t lb = ci; lb < lim; lb += nc) {
        const int32_t p_ = table[lb];
        const uint32_t p0 = (p_ < 0 ? 0u : static_cast<uint32_t>(p_)) * TPP;
        for (uint32_t tr = 0; tr < TROWS; ++tr) {
            cb_reserve_back(CB_IN, TPR);
            const uint32_t l1 = get_write_ptr(CB_IN);
            for (uint32_t j = 0; j < TPR; ++j) {
                noc_async_read(src.get_noc_addr(p0 + tr * TPR + j), l1 + j * TILE_BYTES, TILE_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_IN, TPR);
        }
    }
}
