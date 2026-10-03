// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE tail reader: replaces tilize_with_val_padding + deepseek_moe_fast_reduce_nc_fused reader for T < 32 token rows.
// combine [K, T, H] RM bf16 (one page = one (slot, token) row of H elements).  This core owns TPC consecutive 32-wide
// column tiles starting at tile j0.  For every column tile it builds K tiles (one per slot k; rows 0..T-1 = tokens,
// rows T..31 = 0) directly in tile layout, plus one score tile per slot (column 0 = score[t, k] if the slot's expert
// lives in this device's mesh column, else 0; padding rows 0) -- the same masking as the fused fast_reduce reader.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t TPC = get_compile_time_arg_val(5);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(6);
    constexpr uint32_t sc_page = get_compile_time_arg_val(7);    // bytes of one scores page (K bf16)
    constexpr uint32_t idx_page = get_compile_time_arg_val(8);   // bytes of one indices page (K u16)
    constexpr uint32_t map_page = get_compile_time_arg_val(9);   // bytes of expert-mapping row page
    constexpr uint32_t row_page = get_compile_time_arg_val(10);  // bytes of one combine row (H bf16)
    constexpr uint32_t STRIDE = 64;  // L1 stride of the small scratch pages (T of them each)
    constexpr auto cmb_args = TensorAccessorArgs<11>();
    constexpr auto sc_args = TensorAccessorArgs<cmb_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto map_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t cmb_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sc_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t map_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t col_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto cmb = TensorAccessor(cmb_args, cmb_addr, row_page);
    const auto scA = TensorAccessor(sc_args, sc_addr, sc_page);
    const auto idxA = TensorAccessor(idx_args, idx_addr, idx_page);
    const auto mapA = TensorAccessor(map_args, map_addr, map_page);
    const auto colA = TensorAccessor(col_args, col_addr, 256);
    experimental::CB act(cb_act), scb(cb_sc), tmp(cb_tmp);

    // scratch layout in cb_tmp: [rowseg: K*T slabs of TPC*64B][scores T*32][indices T*32][mapping][col 64]
    constexpr uint32_t SEG = TPC * 64;
    constexpr uint32_t OFF_SC = K * T * SEG;
    constexpr uint32_t OFF_IDX = OFF_SC + T * STRIDE;
    constexpr uint32_t OFF_COL = OFF_IDX + T * STRIDE;
    constexpr uint32_t OFF_MAP = OFF_COL + 256;
    tmp.reserve_back(1);
    const uint32_t tbase = tmp.get_write_ptr();

    act.reserve_back(K * TPC);
    scb.reserve_back(K);
    noc.async_write_zeros(act, K * TPC * 2048, {.offset_bytes = 0});
    noc.async_write_zeros(scb, K * 2048, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(scA, tmp, sc_page, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = OFF_SC + t * STRIDE});
        noc.async_read(idxA, tmp, idx_page, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = OFF_IDX + t * STRIDE});
    }
    noc.async_read(colA, tmp, 256, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = OFF_COL});
    noc.async_read(mapA, tmp, map_page, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = OFF_MAP});
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_read(
                cmb, tmp, SEG, {.page_id = k * T + t, .offset_bytes = j0 * 64}, {.offset_bytes = (k * T + t) * SEG});
        }
    }
    noc.async_read_barrier();

    // activation tiles: tile (jj, k) at page jj*K + k; row t -> face (t>>4)*2 + {0,1}
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbase);
    volatile tt_l1_ptr uint32_t* ap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act.get_write_ptr());
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t jj = 0; jj < TPC; ++jj) {
                volatile tt_l1_ptr uint32_t* s = seg + ((k * T + t) * SEG + jj * 64) / 4;
                // tile words: 512 per tile; face f at f*128 words; row r of a face = 8 words
                volatile tt_l1_ptr uint32_t* d = ap + (jj * K + k) * 512 + ((t >> 4) << 8) + ((t & 15) << 3);
                for (uint32_t w = 0; w < 8; ++w) {
                    d[w] = s[w];
                    d[128 + w] = s[8 + w];
                }
            }
        }
    }

    // score tiles: tile k, column 0 of row t
    volatile tt_l1_ptr uint16_t* sc16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_SC);
    volatile tt_l1_ptr uint16_t* ix16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_IDX);
    volatile tt_l1_ptr uint16_t* mp16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_MAP);
    const uint32_t my_col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbase + OFF_COL)[0];
    volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scb.get_write_ptr());
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            const uint16_t e = ix16[t * (STRIDE / 2) + k];
            const bool on = (mp16[e] % mesh_cols) == my_col;
            const uint32_t pos = ((t >> 4) << 9) + ((t & 15) << 4);  // column 0 of row t (bf16 index)
            sp[k * 1024 + pos] = on ? sc16[t * (STRIDE / 2) + k] : (uint16_t)0;
        }
    }
#ifdef TAIL_DBG
    if (j0 == 0) {
        volatile tt_l1_ptr uint16_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_COL);
        // words(u16) from index 2: for t<4: idx[t][0..5] (24), sc[t][0..5] (24) ; then mapping of idx[0..]
        uint32_t n = 2;
        for (uint32_t t = 0; t < T && t < 4; ++t) {
            for (uint32_t k = 0; k < 6; ++k) {
                dbg[n++] = ix16[t * 16 + k];
            }
        }
        for (uint32_t t = 0; t < T && t < 4; ++t) {
            for (uint32_t k = 0; k < 6; ++k) {
                dbg[n++] = sc16[t * 16 + k];
            }
        }
        noc.async_write(tmp, colA, 128, {.offset_bytes = OFF_COL}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
#endif
    act.push_back(K * TPC);
    scb.push_back(K);
}
