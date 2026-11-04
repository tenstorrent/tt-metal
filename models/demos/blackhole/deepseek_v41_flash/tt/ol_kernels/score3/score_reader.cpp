// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: fused verify-indexer score, reader, PER-USER split (DSV41_OL_SCORE3=1): the cores are split in U groups
// of CPU cores, group u scores the key tile columns of user u only (rows u * NROW .. u * NROW + NROW - 1 of the verify
// block). Per core: q = NROW x 4 tiles (not R x 4: half / a third of the q traffic that was the long pole of the
// startup), NROW gate tiles, 4 key tiles per column; the compute kernel is the stock one with R = NROW rows and U = 1
// (tt/ol_kernels/score2/score_compute.cpp). Numerics identical to the stock kernel. No per-core runtime arguments (core
// index from the logical coordinates, grid width = common arg 4); everything of the startup (gate tile, q, first key
// column) is in flight together before the first barrier.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "../ol_zones.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t NROW = get_compile_time_arg_val(4);  // rows per user
    constexpr uint32_t U = get_compile_time_arg_val(5);     // users
    constexpr uint32_t K_TILE_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t K_TILES_PER_USER = get_compile_time_arg_val(7);  // n_alloc / 32
    constexpr uint32_t cb_mail = get_compile_time_arg_val(8);
    constexpr uint32_t CPU = get_compile_time_arg_val(9);  // cores per user
    constexpr uint32_t NCT_TOTAL = get_compile_time_arg_val(10);
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core_idx = ol_core_idx(get_common_arg_val<uint32_t>(4));
    const uint32_t user = core_idx / CPU, li = core_idx % CPU;

    Noc noc;
    const auto qa = TensorAccessor(q_args, q_addr, 2048);
    const auto ka = TensorAccessor(k_args, k_addr, K_TILE_BYTES);
    const auto wa = TensorAccessor(w_args, w_addr, 2048);
    experimental::CB cbq(cb_q), cbk(cb_k), cbwt(cb_wt), cbw(cb_w);

    experimental::CB cbm(cb_mail);
    cbm.reserve_back(1);
    uint32_t valid;
    {
        OLZ("Z_VALID");
        valid = trace_metadata::read_metadata_scalar_u32(noc, v_args, v_addr, cbm.get_write_ptr());
    }
    uint32_t live = (valid + 31) / 32;
    if (live > NCT_TOTAL) {
        live = NCT_TOTAL;
    }
    const uint32_t per = live / CPU, extra = live % CPU;
    const uint32_t ct0 = li * per + (li < extra ? li : extra);
    const uint32_t nct = per + (li < extra ? 1 : 0);
    volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbm.get_write_ptr());
    mp[0] = nct;
    cbm.push_back(1);

    cbw.reserve_back(1);
    noc.async_read(wa, cbw, 2048, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    cbq.reserve_back(NROW * 4);
    for (uint32_t t = 0; t < NROW * 4; ++t) {
        noc.async_read(qa, cbq, 2048, {.page_id = user * NROW * 4 + t, .offset_bytes = 0}, {.offset_bytes = t * 2048});
    }
    uint32_t ct = ct0;
    if (nct > 0) {
        cbk.reserve_back(4);
        for (uint32_t d = 0; d < 4; ++d) {
            noc.async_read(
                ka,
                cbk,
                K_TILE_BYTES,
                {.page_id = (user * K_TILES_PER_USER + ct) * 4 + d, .offset_bytes = 0},
                {.offset_bytes = d * K_TILE_BYTES});
        }
    }
    cbwt.reserve_back(NROW);
    for (uint32_t t = 0; t < NROW; ++t) {
        noc.async_write_zeros(cbwt, 2048, {.offset_bytes = t * 2048});
    }
    {
        OLZ("Z_BAR");
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
    }
    {
        OLZ("Z_GCOPY");
        // gate tile rl: row rl = the gates of global row g = user * NROW + rl (16-bit values moved as 32-bit words: 8
        // words per face half)
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbw.get_write_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbwt.get_write_ptr());
        for (uint32_t rl = 0; rl < NROW; ++rl) {
            const uint32_t g = user * NROW + rl;
            const uint32_t fs = (((g >> 4) << 1) * 256 + (g & 15) * 16) >> 1;
            const uint32_t fd = (((rl >> 4) << 1) * 256 + (rl & 15) * 16) >> 1;
            for (uint32_t c = 0; c < 8; ++c) {
                dst[rl * 512 + fd + c] = src[fs + c];
                dst[rl * 512 + fd + 128 + c] = src[fs + 128 + c];
            }
        }
    }
    cbwt.push_back(NROW);
    cbq.push_back(NROW * 4);
    if (nct > 0) {
        cbk.push_back(4);
    }
    for (ct = ct0 + 1; ct < ct0 + nct; ++ct) {
        OLZ("Z_K");
        cbk.reserve_back(4);
        for (uint32_t d = 0; d < 4; ++d) {
            noc.async_read(
                ka,
                cbk,
                K_TILE_BYTES,
                {.page_id = (user * K_TILES_PER_USER + ct) * 4 + d, .offset_bytes = 0},
                {.offset_bytes = d * K_TILE_BYTES});
        }
        noc.async_read_barrier();
        cbk.push_back(4);
    }
}
