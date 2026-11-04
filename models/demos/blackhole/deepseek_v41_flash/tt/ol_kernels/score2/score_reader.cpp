// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused verify-indexer score, reader. Per core: (1) builds the R gate tiles (row r of tile r = w[r, :], zeros
// elsewhere), (2) loads the R x 4 q tiles (resident), (3) streams the 4 key tiles of every user for the core's key tile
// columns [ct0, ct0 + nct).
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
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr uint32_t U = get_compile_time_arg_val(5);
    constexpr uint32_t K_TILE_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t K_TILES_PER_USER =
        get_compile_time_arg_val(7);  // n_alloc / 32 (tile columns of the slab per user)
    constexpr uint32_t NROW = R / U;
    constexpr uint32_t cb_mail = get_compile_time_arg_val(8);
    constexpr uint32_t NCORES = get_compile_time_arg_val(9);
    constexpr uint32_t NCT_TOTAL = get_compile_time_arg_val(10);
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(3);
    // otherlayers: the core index from the logical coordinates (row-major over the grid): no per-core runtime arguments
    // (every unique arg set is a separate dispatcher write)
    const uint32_t core_idx = get_absolute_logical_y() * get_common_arg_val<uint32_t>(4) + get_absolute_logical_x();

    Noc noc;
    const auto qa = TensorAccessor(q_args, q_addr, 2048);
    const auto ka = TensorAccessor(k_args, k_addr, K_TILE_BYTES);
    const auto wa = TensorAccessor(w_args, w_addr, 2048);
    experimental::CB cbq(cb_q), cbk(cb_k), cbwt(cb_wt), cbw(cb_w);

    // live key tile columns from the valid length (device tensor): this core's slice [ct0, ct0 + nct)
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
    const uint32_t per = live / NCORES, extra = live % NCORES;
    const uint32_t ct0 = core_idx * per + (core_idx < extra ? core_idx : extra);
    const uint32_t nct = per + (core_idx < extra ? 1 : 0);
    volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbm.get_write_ptr());
    mp[0] = nct;
    cbm.push_back(1);

    // otherlayers: everything below is issued before the first barrier (the stock reader serialised gate read -> RISC
    // zeroing / copy -> q read -> first key read, about 15 us before the first matmul): gate tile read, the q tiles,
    // the first key column in flight together, the gate tiles zeroed by the NOC.
    cbw.reserve_back(1);
    noc.async_read(wa, cbw, 2048, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    cbq.reserve_back(R * 4);
    for (uint32_t t = 0; t < R * 4; ++t) {
        noc.async_read(qa, cbq, 2048, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * 2048});
    }
    uint32_t ct = ct0;
    if (nct > 0) {
        cbk.reserve_back(U * 4);
        for (uint32_t u = 0; u < U; ++u) {
            for (uint32_t d = 0; d < 4; ++d) {
                noc.async_read(
                    ka,
                    cbk,
                    K_TILE_BYTES,
                    {.page_id = (u * K_TILES_PER_USER + ct) * 4 + d, .offset_bytes = 0},
                    {.offset_bytes = (u * 4 + d) * K_TILE_BYTES});
            }
        }
    }
    cbwt.reserve_back(R);
    for (uint32_t t = 0; t < R; ++t) {
        noc.async_write_zeros(cbwt, 2048, {.offset_bytes = t * 2048});
    }
    {
        OLZ("Z_BAR");
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
    }
    // 32-bit word copies of the 16 gate values (8 words) of each face half of row r (the stock loop copied 16-bit
    // values one by one: 13 us)
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbw.get_write_ptr());
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbwt.get_write_ptr());
    {
        OLZ("Z_GCOPY");
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t f0 = (((r >> 4) << 1) * 256 + (r & 15) * 16) >>
                                1;  // word offset of (row r, cols 0..15); cols 16..31 are the next face (+128 words)
            for (uint32_t c = 0; c < 8; ++c) {
                dst[r * 512 + f0 + c] = src[f0 + c];
                dst[r * 512 + f0 + 128 + c] = src[f0 + 128 + c];
            }
        }
    }
    cbwt.push_back(R);
    cbq.push_back(R * 4);
    if (nct > 0) {
        cbk.push_back(U * 4);  // the first key column was read above
    }

    for (ct = ct0 + 1; ct < ct0 + nct; ++ct) {
        OLZ("Z_K");
        cbk.reserve_back(U * 4);
        for (uint32_t u = 0; u < U; ++u) {
            for (uint32_t d = 0; d < 4; ++d) {
                noc.async_read(
                    ka,
                    cbk,
                    K_TILE_BYTES,
                    {.page_id = (u * K_TILES_PER_USER + ct) * 4 + d, .offset_bytes = 0},
                    {.offset_bytes = (u * 4 + d) * K_TILE_BYTES});
            }
        }
        noc.async_read_barrier();
        cbk.push_back(U * 4);
    }
}
