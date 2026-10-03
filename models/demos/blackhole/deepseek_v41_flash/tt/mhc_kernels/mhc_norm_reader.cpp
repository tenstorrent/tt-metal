// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC collapse norm apply, reader: gathers the NCORE partial row-sum tiles (rows 0..3 of face 0, column 0) into one
// tile R[t, k] = partial_k[t], and this core's bf16 h tiles and fp32 weight tiles (weight * sqrt(C), rows replicated).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_r = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t NCORE = get_compile_time_arg_val(5);
    constexpr uint32_t T = get_compile_time_arg_val(6);
    constexpr uint32_t SPAGES = get_compile_time_arg_val(7);
    constexpr auto h_args = TensorAccessorArgs<8>();
    constexpr auto p_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto h_acc = TensorAccessor(h_args, h_addr, 2048);
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    const auto w_acc = TensorAccessor(w_args, w_addr, 4096);
    experimental::CB cbr(cb_r), cbo(cb_ones), cbs(cb_s), cbh(cb_h), cbw(cb_w);

    cbo.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t k = 0; k < 1024; ++k) {
            o[k] = 0x3F800000u;
        }
    }
    cbo.push_back(1);

    // partial tiles -> R
    cbs.reserve_back(SPAGES);
    cbr.reserve_back(1);
    noc.async_write_zeros(cbr, 4096, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t k = 0; k < NCORE; ++k) {
        constexpr uint32_t T0 = T < 16 ? T : 16;
        noc.async_read(p_acc, cbs, T0 * 64, {.page_id = k, .offset_bytes = 0}, {.offset_bytes = k * T * 64});
        if constexpr (T > 16) {
            noc.async_read(
                p_acc, cbs, (T - 16) * 64, {.page_id = k, .offset_bytes = 2048}, {.offset_bytes = k * T * 64 + 1024});
        }
    }
    // h / w tiles
    const uint32_t ng = j1 - j0;
    cbh.reserve_back(ng);
    cbw.reserve_back(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        noc.async_read(h_acc, cbh, 2048, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 2048});
        noc.async_read(w_acc, cbw, 4096, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 4096});
    }
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
    volatile tt_l1_ptr uint32_t* R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbr.get_write_ptr());
    for (uint32_t k = 0; k < NCORE; ++k) {
        const uint32_t col = (k < 16) ? k : (256 + (k - 16));
        for (uint32_t t = 0; t < T; ++t) {
            R[col + RW(t)] = sc[k * T * 16 + t * 16];
        }
    }
    cbr.push_back(1);
    cbh.push_back(ng);
    cbw.push_back(ng);
}
