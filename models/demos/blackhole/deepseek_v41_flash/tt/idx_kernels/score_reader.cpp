// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused verify-indexer score, reader. Per core: (1) builds the R gate tiles (row r of tile r = w[r, :], zeros
// elsewhere), (2) loads the R x 4 q tiles (resident), (3) streams the 4 key tiles of every user for the core's key tile
// columns [ct0, ct0 + nct).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

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
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t ct0 = get_arg_val<uint32_t>(0);
    const uint32_t nct = get_arg_val<uint32_t>(1);

    Noc noc;
    const auto qa = TensorAccessor(q_args, q_addr, 2048);
    const auto ka = TensorAccessor(k_args, k_addr, K_TILE_BYTES);
    const auto wa = TensorAccessor(w_args, w_addr, 2048);
    experimental::CB cbq(cb_q), cbk(cb_k), cbwt(cb_wt), cbw(cb_w);

    // gate tiles
    cbw.reserve_back(1);
    noc.async_read(wa, cbw, 2048, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbw.get_write_ptr());
    cbwt.reserve_back(R);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbwt.get_write_ptr());
    for (uint32_t i = 0; i < R * 512; ++i) {
        z[i] = 0;
    }
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbwt.get_write_ptr());
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t f0 =
            ((r >> 4) << 1) * 256 + (r & 15) * 16;  // face of (row r, cols 0..15); cols 16..31 are the next face
        for (uint32_t c = 0; c < 16; ++c) {
            dst[r * 1024 + f0 + c] = src[f0 + c];
            dst[r * 1024 + f0 + 256 + c] = src[f0 + 256 + c];
        }
    }
    cbwt.push_back(R);

    // q: R rows x 4 d tiles (tile id r * 4 + d)
    cbq.reserve_back(R * 4);
    for (uint32_t t = 0; t < R * 4; ++t) {
        noc.async_read(qa, cbq, 2048, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * 2048});
    }
    noc.async_read_barrier();
    cbq.push_back(R * 4);

    for (uint32_t ct = ct0; ct < ct0 + nct; ++ct) {
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
