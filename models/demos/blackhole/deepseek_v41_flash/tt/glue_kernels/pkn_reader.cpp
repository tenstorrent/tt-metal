// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// T19 packed-mHC norm apply, reader.  Core (r, k): the NRC partial sum-of-squares tiles of token tile row r, its CN
// bf16 h tiles (r, c0 + j) and the CN fp32 weight tiles (c0 + j; row 0 of each tile is the weight), plus an all-ones
// tile.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t NRC = get_compile_time_arg_val(4);
    constexpr uint32_t CT = get_compile_time_arg_val(5);
    constexpr auto p_args = TensorAccessorArgs<6>();
    constexpr auto h_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t cn = get_arg_val<uint32_t>(2);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    const auto h_acc = TensorAccessor(h_args, h_addr, 2048);
    const auto w_acc = TensorAccessor(w_args, w_addr, 4096);
    experimental::CB cbp(cb_part), cbo(cb_ones), cbh(cb_h), cbw(cb_w);

    cbo.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t k = 0; k < 1024; ++k) {
            o[k] = 0x3F800000u;
        }
    }
    cbo.push_back(1);

    cbp.reserve_back(NRC);
    for (uint32_t p = 0; p < NRC; ++p) {
        noc.async_read(p_acc, cbp, 4096, {.page_id = r * NRC + p, .offset_bytes = 0}, {.offset_bytes = p * 4096});
    }
    cbh.reserve_back(cn);
    cbw.reserve_back(cn);
    for (uint32_t j = 0; j < cn; ++j) {
        noc.async_read(h_acc, cbh, 2048, {.page_id = r * CT + c0 + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
        noc.async_read(w_acc, cbw, 4096, {.page_id = c0 + j, .offset_bytes = 0}, {.offset_bytes = j * 4096});
    }
    noc.async_read_barrier();
    cbp.push_back(NRC);
    cbh.push_back(cn);
    cbw.push_back(cn);
}
