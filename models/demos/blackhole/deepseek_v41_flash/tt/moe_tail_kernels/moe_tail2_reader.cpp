// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE tail reader, tilize variant (DSV41_TAIL_TILIZE=1): the combine row segments are read straight into a row-major CB
// (page = one segment of TPC * 64 B, 32 pages per slot k, rows T..31 zero) and the COMPUTE kernel tilizes them
// (hardware tilize instead of the 3.8 us RISC-V / NoC scatter of moe_tail_reader.cpp). Same scores / mask tiles as the
// plain reader.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t TPC = get_compile_time_arg_val(5);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(6);
    constexpr uint32_t sc_page = get_compile_time_arg_val(7);
    constexpr uint32_t idx_page = get_compile_time_arg_val(8);
    constexpr uint32_t map_page = get_compile_time_arg_val(9);
    constexpr uint32_t row_page = get_compile_time_arg_val(10);
    constexpr uint32_t STRIDE = 64;
    constexpr auto cmb_args = TensorAccessorArgs<11>();
    constexpr auto sc_args = TensorAccessorArgs<cmb_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto map_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    static_assert((mesh_cols & (mesh_cols - 1)) == 0, "mesh_cols must be a power of two");

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
    experimental::CB rm(cb_rm), scb(cb_sc), tmp(cb_tmp);

    constexpr uint32_t SEG = TPC * 64;
    constexpr uint32_t OFF_SC = 0;
    constexpr uint32_t OFF_IDX = OFF_SC + T * STRIDE;
    constexpr uint32_t OFF_COL = OFF_IDX + T * STRIDE;
    constexpr uint32_t OFF_MAP = OFF_COL + 256;
    tmp.reserve_back(1);
    const uint32_t tbase = tmp.get_write_ptr();

    rm.reserve_back(32 * K);
    scb.reserve_back(K);
    for (uint32_t k = 0; k < K; ++k) {
        noc.async_write_zeros(rm, (32 - T) * SEG, {.offset_bytes = (k * 32 + T) * SEG});
    }
    noc.async_write_zeros(scb, K * 2048, {.offset_bytes = 0});
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(scA, tmp, sc_page, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = OFF_SC + t * STRIDE});
        noc.async_read(idxA, tmp, idx_page, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = OFF_IDX + t * STRIDE});
    }
    noc.async_read(colA, tmp, 256, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = OFF_COL});
    noc.async_read(mapA, tmp, map_page, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = OFF_MAP});
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_read(
                cmb, rm, SEG, {.page_id = k * T + t, .offset_bytes = j0 * 64}, {.offset_bytes = (k * 32 + t) * SEG});
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
    rm.push_back(32 * K);

    // score tiles: tile k, column 0 of row t
    volatile tt_l1_ptr uint16_t* sc16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_SC);
    volatile tt_l1_ptr uint16_t* ix16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_IDX);
    volatile tt_l1_ptr uint16_t* mp16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tbase + OFF_MAP);
    const uint32_t my_col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbase + OFF_COL)[0];
    volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scb.get_write_ptr());
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            const uint16_t e = ix16[t * (STRIDE / 2) + k];
            const bool on = (mp16[e] & (mesh_cols - 1)) == my_col;
            const uint32_t pos = ((t >> 4) << 9) + ((t & 15) << 4);  // column 0 of row t (bf16 index)
            sp[k * 1024 + pos] = on ? sc16[t * (STRIDE / 2) + k] : (uint16_t)0;
        }
    }
    scb.push_back(K);
#ifdef TAIL_FOLD
    {  // DSV41_TAIL_FOLD=1: the shared-expert partial (bf16 TILE [1,1,T,H], same tile ids as the output) rides along:
       // TPC tiles into cb CB_FOLD for the compute kernel's final add
        constexpr auto fold_args = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
        const auto foldA = TensorAccessor(fold_args, get_common_arg_val<uint32_t>(5), 2048);
        experimental::CB fb(CB_FOLD);
        fb.reserve_back(TPC);
        for (uint32_t j = 0; j < TPC; ++j) {
            noc.async_read(foldA, fb, 2048, {.page_id = j0 + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
        }
        noc.async_read_barrier();
        fb.push_back(TPC);
    }
#endif
}
