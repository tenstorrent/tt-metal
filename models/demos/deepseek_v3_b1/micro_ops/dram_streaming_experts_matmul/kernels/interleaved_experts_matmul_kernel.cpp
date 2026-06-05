// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Interleaved-DRAM selective-experts matmul (tt-xla Path A), multi-core.
//
// Streams only the router-selected experts from DRAM and matmuls them against a
// per-row activation. Every operand is a plain INTERLEAVED DRAM tensor (the layout
// the tt-xla compiler hands an opaque ttnn.tt_lang_op operand), read/written via
// InterleavedAddrGenFast. The N output columns are split across compute cores:
// core `core_id` owns column tiles [core_id*per_core_n, (core_id+1)*per_core_n),
// reads only those weight columns, and writes only that output slice -> the
// selected experts stream in parallel across cores. All CBs are working L1.
//
// Logical op:  out[r, 0, :] = in0[r, 0, :] @ in1[index[r], :, :]
//   in0   : [R, 1, K]   in1 : [E, K, N]   index : [1, R] int32   out : [R, 1, N]
// M=1 per (token,expert) row (activation in tile row 0) so the matmul output also
// lands in row 0; the [R,1,*] shaping keeps each output row in its own tile-group.
//
// CT args POSITIONAL (generic_op passes KERNEL_COMPILE_TIME_ARGS in artifact order;
// no KERNEL_COMPILE_TIME_ARG_MAP, so the named accessor is unavailable). Order must
// match stream_experts_kernel.build_*_artifact. Per-core RT arg 0 = core_id; common
// RT args 0..3 = in0/in1/index/out buffer base addresses.

#include "api/compile_time_args.h"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#endif

namespace {
constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
constexpr uint32_t cb_index = get_compile_time_arg_val(2);
constexpr uint32_t cb_out = get_compile_time_arg_val(3);
constexpr uint32_t Kt = get_compile_time_arg_val(4);
constexpr uint32_t Nt = get_compile_time_arg_val(5);  // total N tiles (full row)
constexpr uint32_t kNumRows = get_compile_time_arg_val(6);
// 7 = num_experts (unused). 8..11 = page sizes. 12..15 = data formats.
constexpr uint32_t per_core_n = get_compile_time_arg_val(16);  // N tiles per core
constexpr uint32_t k_reuse = get_compile_time_arg_val(17);     // rows/token sharing in0
}  // namespace

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // ---- reader: index, then per row {in0 row, this core's expert columns} -----
    constexpr uint32_t in0_page = get_compile_time_arg_val(8);
    constexpr uint32_t in1_page = get_compile_time_arg_val(9);
    constexpr uint32_t idx_page = get_compile_time_arg_val(10);
    constexpr uint32_t in0_df = get_compile_time_arg_val(12);
    constexpr uint32_t in1_df = get_compile_time_arg_val(13);
    constexpr uint32_t idx_df = get_compile_time_arg_val(15);

    const uint32_t col_base = get_arg_val<uint32_t>(0) * per_core_n;  // core_id * per_core_n
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t index_addr = get_common_arg_val<uint32_t>(2);

    const InterleavedAddrGenFast<true> in0_gen = {
        .bank_base_address = in0_addr, .page_size = in0_page, .data_format = static_cast<DataFormat>(in0_df)};
    const InterleavedAddrGenFast<true> in1_gen = {
        .bank_base_address = in1_addr, .page_size = in1_page, .data_format = static_cast<DataFormat>(in1_df)};
    const InterleavedAddrGenFast<true> idx_gen = {
        .bank_base_address = index_addr, .page_size = idx_page, .data_format = static_cast<DataFormat>(idx_df)};

    // Read all index tiles into L1 ([1,R] -> ceil(R/32) tiles).
    constexpr uint32_t idx_tiles = (kNumRows + 31) / 32;
    cb_reserve_back(cb_index, idx_tiles);
    uint32_t idx_l1 = get_write_ptr(cb_index);
    for (uint32_t g = 0; g < idx_tiles; ++g) {
        noc_async_read_tile(g, idx_gen, idx_l1 + g * idx_page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_index, idx_tiles);
    volatile tt_l1_ptr int32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(idx_l1);

    // Per row: read the in0 activation row, then stream this core's expert-weight
    // columns. The Kt tile reads of a column are issued back-to-back (async) and
    // overlap under one barrier; b1-style trid multi-buffering was measured ~neutral
    // (slightly worse) here because the interleaved per-tile reads already overlap,
    // so we keep the simple form.
    for (uint32_t r = 0; r < kNumRows; ++r) {
        uint32_t local = r % 32;
        uint32_t e = static_cast<uint32_t>(idx_ptr[(r / 32) * 1024 + (local / 16) * 256 + (local % 16)]);

        // in0 is shared across a token's k_reuse experts (gate_up): read it once per
        // token (when in0_row changes) instead of redundantly per output row.
        if (r % k_reuse == 0) {
            const uint32_t in0_row = r / k_reuse;
            cb_reserve_back(cb_in0, Kt);
            uint32_t in0_l1 = get_write_ptr(cb_in0);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                noc_async_read_tile(in0_row * Kt + kt, in0_gen, in0_l1 + kt * in0_page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in0, Kt);
        }

        const uint32_t e_base = e * Kt * Nt;
        for (uint32_t nl = 0; nl < per_core_n; ++nl) {
            const uint32_t nt = col_base + nl;
            cb_reserve_back(cb_in1, Kt);
            uint32_t in1_l1 = get_write_ptr(cb_in1);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                noc_async_read_tile(e_base + kt * Nt + nt, in1_gen, in1_l1 + kt * in1_page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in1, Kt);
        }
    }

#elif defined(COMPILE_FOR_TRISC)
    // ---- compute: per row, per output column (this core's slice) -------------
    mm_init(cb_in0, cb_in1, cb_out, /*transpose=*/0);
    for (uint32_t r = 0; r < kNumRows; ++r) {
        if (r % k_reuse == 0) {
            cb_wait_front(cb_in0, Kt);  // new token's activation; reused across k_reuse rows
        }
        for (uint32_t nl = 0; nl < per_core_n; ++nl) {
            cb_wait_front(cb_in1, Kt);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(cb_in0, cb_in1, kt, kt, /*dst=*/0);
            }
            tile_regs_commit();
            cb_pop_front(cb_in1, Kt);
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out, 0);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        if (r % k_reuse == (k_reuse - 1)) {
            cb_pop_front(cb_in0, Kt);
        }
    }

#elif defined(COMPILE_FOR_BRISC)
    // ---- writer: drain cb_out to out[R,1,N] tile-group r, this core's columns --
    constexpr uint32_t out_page = get_compile_time_arg_val(11);
    constexpr uint32_t out_df = get_compile_time_arg_val(14);
    const uint32_t col_base = get_arg_val<uint32_t>(0) * per_core_n;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);
    const InterleavedAddrGenFast<true> out_gen = {
        .bank_base_address = out_addr, .page_size = out_page, .data_format = static_cast<DataFormat>(out_df)};
    for (uint32_t r = 0; r < kNumRows; ++r) {
        for (uint32_t nl = 0; nl < per_core_n; ++nl) {
            cb_wait_front(cb_out, 1);
            uint32_t out_l1 = get_read_ptr(cb_out);
            noc_async_write_tile(r * Nt + (col_base + nl), out_gen, out_l1);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
#endif
}
