// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 MTP input projection (tt/spec_mtp.py MkEh; doc/ttnn/17-mtp-spec-decode/design.md §3), owner BRISC (NoC0):
// the head's owner kernels (kernels/head/head_c.cpp, head_dm1.cpp: RMSNorm of the stream "mean" * w over the 64 owners,
// owner k = column tiles c0 = 2k, 2k + 1) on ONE input vector per row instead of the 4-stream residual: the input goes
// into stream slot 0 of the flat x tiles (slots 1..3 stay zero; head_c runs with QUARTER = 1.0, so m = x exactly), and
// the normalised bf16 h is written to column tiles out_page0 + c0 + j of hcat [1, 1, T, 8192] (out_page0 = 0: enorm(e),
// 128: hnorm(h)). The second program (hnorm) signals the eh_proj DS readers (HEAD_HN_SEM, count 64), like MkHead.
// Input (EH_IN):
//   1 EMB      the token of the source row (word 0 of a [32, 64] uint32 page) -> its embedding row's 64 columns (the
//              bf16 ROW_MAJOR [V, 4096] table) -> fp32 (bits << 16, exact)
//   2 BF16     row (source row) of the bf16 TILE tensor [1, 1, T', 4096] (the main head's h = model.norm(mean X))
//   3 STACKED  stream 0 of row (source row) of a stacked fp32 residual [T', 1, 4, 4096] (the MTP block's raw output)
// Source row of output row t: t, or with EH_SEL the device word 0 of the [1, 32] uint32 selector page (T = 1).
// EH_SEL_ROWS (with EH_SEL; GLM53_SPEC_MULTI, doc/perf/mtp_multi_user_design.md section 6 item 5): output row t's
// source row = selector word t (the first chained MTP launch: row u = user u, its catch-up row u (K + 1) + n_u =
// SpecCtl's selector word u; T <= 32).
// EH_FLAT (EH_IN 3 only; GLM53_SPEC_MULTI): the source is a FLAT residual [1, 1, T', 16384] (an MTP launch of more than
// 4 rows): stream 0 of row r = row r of column tiles 0..127 (one tile row), instead of the stacked [T', 1, 4, 4096].
// EH_ROW0 (BF16 only; tt/spec_mtp.py MkEh.norm_h_proj(row0=): the MTP-cache prefill over the TP prefill's final-norm
// tap [1, 1, C, 4096]): source row of output row t = row0 + t, all in one tile row (row0 % 32 + T <= 32); row0 = word 0
// of a [1, 32] uint32 DRAM page (its address: the runtime arg after the readers' NoC pairs) -- device-resident, so one
// captured launch serves every row offset. Without the define the instructions are unchanged.
// Compile-time args: TensorAccessorArgs of in, w, out, table (EMB; else `in` again).
// Runtime args: in, w, out, c0, T, out_page0, table (EMB, else 0), selector (EH_SEL, else 0), n readers, their NoC
// (x, y) (HEAD_HN_SEM incremented on each after hcat's columns landed; n = 0: no signal)[, the row0 page (EH_ROW0)].
// CBs: kernels/head/head_common.hpp (X, W, OUT; EMB_CB = staging 4 KB + 512 B).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace head;

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(2));
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);
    const uint32_t out_page0 = get_arg_val<uint32_t>(5);
    const uint32_t n_readers = get_arg_val<uint32_t>(8);

    const uint32_t stage = (get_write_ptr(EMB_CB) + 63u) & ~63u;  // 4 KB staging + 256 B token / selector rows
    const uint32_t trow = stage + 4096;

    uint32_t src_row[32];
    for (uint32_t t = 0; t < T; ++t) {
        src_row[t] = t;
    }
#if defined(EH_SEL) && defined(EH_SEL_ROWS)
    {
        const InterleavedAddrGen<true> sg = {.bank_base_address = get_arg_val<uint32_t>(7), .page_size = 128};
        noc_async_read(sg.get_noc_addr(0), trow, 128);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            src_row[t] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[t];
        }
    }
#elif defined(EH_SEL)
    {
        const InterleavedAddrGen<true> sg = {.bank_base_address = get_arg_val<uint32_t>(7), .page_size = 128};
        noc_async_read(sg.get_noc_addr(0), trow, 64);
        noc_async_read_barrier();
        src_row[0] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[0];
    }
#endif
#if defined(EH_SEL_ROWS) && !defined(EH_SEL)
#error "EH_SEL_ROWS needs EH_SEL (the selector page)"
#endif
#if defined(EH_ROW0)
    static_assert(EH_IN == 2, "EH_ROW0: the bf16 tile source only");
    uint32_t row0;
    {
        const InterleavedAddrGen<true> rg = {
            .bank_base_address = get_arg_val<uint32_t>(9 + 2 * n_readers), .page_size = 128};
        noc_async_read(rg.get_noc_addr(0), trow, 64);
        noc_async_read_barrier();
        row0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[0];
    }
    for (uint32_t t = 0; t < T; ++t) {
        src_row[t] = (row0 & (TILE - 1)) + t;  // within the source tile row row0 / 32
    }
    const uint32_t in_page0 = (row0 / TILE) * COL_TILES;
#endif

    // flat x tiles zeroed (slots 1..3 stay zero: m = x0 with QUARTER = 1.0), the norm weight's 2 row tiles
    cb_reserve_back(CB_X, N_STREAMS * COLS);
    const uint32_t x_l1 = get_write_ptr(CB_X);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < N_STREAMS * COLS * FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
    }
    cb_reserve_back(CB_W, COLS);
    const uint32_t w_l1 = get_write_ptr(CB_W);
    for (uint32_t j = 0; j < COLS; ++j) {
        noc_async_read_page(c0 + j, w, w_l1 + j * FP32_TILE_BYTES);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_l1);

#if EH_IN == 1
    const auto table = TensorAccessor(t_args, get_arg_val<uint32_t>(6));
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read_page(src_row[t], in, trow);
        noc_async_read_barrier();
        uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trow)[0];
        if (id >= EH_VOCAB) {  // an out-of-range id (a host write gone wrong) must not read outside the table
            id = 0;
        }
        noc_async_read(table.get_noc_addr(id, c0 * 64), stage, 128);  // columns 32 c0 .. 32 c0 + 63 (bf16)
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage);
        for (uint32_t j = 0; j < COLS; ++j) {
            for (uint32_t c = 0; c < TILE; ++c) {
                x[j * (FP32_TILE_BYTES / 4) + tile_off(t, c)] = static_cast<uint32_t>(e[j * TILE + c]) << 16;
            }
        }
    }
#elif EH_IN == 2
    // the 2 bf16 tiles of columns c0, c0 + 1 (one row of tiles: T' <= 32), every output row from them
    for (uint32_t j = 0; j < COLS; ++j) {
#if defined(EH_ROW0)
        noc_async_read_page(in_page0 + c0 + j, in, stage + j * 2048);  // tile row row0 / 32 of the [1, 1, C, 4096] tap
#else
        noc_async_read_page(c0 + j, in, stage + j * 2048);
#endif
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* hb = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage);
    for (uint32_t j = 0; j < COLS; ++j) {
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < TILE; ++c) {
                x[j * (FP32_TILE_BYTES / 4) + tile_off(t, c)] =
                    static_cast<uint32_t>(hb[j * 1024 + tile_off(src_row[t], c)]) << 16;
            }
        }
    }
#elif EH_IN == 3 && defined(EH_FLAT)
    // the flat residual: stream 0 = column tiles 0..127 of its one tile row, row src_row of each (2 face-row reads)
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < COLS; ++j) {
            const uint32_t page = c0 + j;
            const uint32_t dst = x_l1 + j * FP32_TILE_BYTES;
            noc_async_read(in.get_noc_addr(page, tile_off(src_row[t], 0) * 4), dst + tile_off(t, 0) * 4, 64);
            noc_async_read(in.get_noc_addr(page, tile_off(src_row[t], 16) * 4), dst + tile_off(t, 16) * 4, 64);
        }
    }
    noc_async_read_barrier();
#elif EH_IN == 3
    // stream 0 (tile row 0) of the source row's column tiles: 2 face-row reads of 64 B per tile, as head_dm0
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < COLS; ++j) {
            const uint32_t page = src_row[t] * COL_TILES + c0 + j;
            const uint32_t dst = x_l1 + j * FP32_TILE_BYTES;
            noc_async_read(in.get_noc_addr(page, tile_off(0, 0) * 4), dst + tile_off(t, 0) * 4, 64);
            noc_async_read(in.get_noc_addr(page, tile_off(0, 16) * 4), dst + tile_off(t, 16) * 4, 64);
        }
    }
    noc_async_read_barrier();
#else
#error "EH_IN: 1 EMB, 2 BF16, 3 STACKED"
#endif
    cb_push_back(CB_X, N_STREAMS * COLS);
    cb_push_back(CB_W, COLS);

    // h out: column tiles out_page0 + c0 + j of hcat
    cb_wait_front(CB_OUT, COLS);
    const uint32_t o_l1 = get_read_ptr(CB_OUT);
    const uint32_t opage = get_tile_size(CB_OUT);
    for (uint32_t j = 0; j < COLS; ++j) {
        noc_async_write_page(out_page0 + c0 + j, out, o_l1 + j * opage);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, COLS);
    if (n_readers > 0) {
        mk::signal_handoff(HEAD_HN_SEM, 9, n_readers);  // the eh_proj readers' wait (count 64 owners)
    }
}
