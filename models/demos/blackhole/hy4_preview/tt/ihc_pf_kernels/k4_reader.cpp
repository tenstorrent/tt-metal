// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K4 (norm) reader: core i = r * NCH + ch: ones tile (cst page 0), the NCH partial sums of squares of
// row r (accp pages r*NCH + k), then per pair of column tiles the h tiles (hbuf page r*NTD + c) and the bf16 norm
// weight tiles (wn page c, 2 KB). CT: 0 cb_c, 1 cb_a, 2 cb_h, 3 cb_w, 4 NTD, 5 CT, 6 NCH, accessors cst, accp, h, wn.
// Common: 4 addresses. RT: i.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_c = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t NTD = get_compile_time_arg_val(4);
    constexpr uint32_t CT = get_compile_time_arg_val(5);
    constexpr uint32_t NCH = get_compile_time_arg_val(6);
    constexpr auto c_args = TensorAccessorArgs<7>();
    constexpr auto a_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t r = i / NCH, ch = i % NCH;
    const auto c_acc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto a_acc = TensorAccessor(a_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto h_acc = TensorAccessor(h_args, get_common_arg_val<uint32_t>(2), 4096);
    const auto w_acc = TensorAccessor(w_args, get_common_arg_val<uint32_t>(3), 2048);
    Noc noc;
    experimental::CB cbc(cb_c), cba(cb_a), cbh(cb_h), cbw(cb_w);
    cbc.reserve_back(1);
    cba.reserve_back(NCH);
    noc.async_read(c_acc, cbc, 4096, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    for (uint32_t k = 0; k < NCH; ++k) {
        noc.async_read(a_acc, cba, 4096, {.page_id = r * NCH + k, .offset_bytes = 0}, {.offset_bytes = k * 4096});
    }
    noc.async_read_barrier();
    cbc.push_back(1);
    cba.push_back(NCH);
    for (uint32_t jb = 0; jb < CT; jb += 2) {
        cbh.reserve_back(2);
        cbw.reserve_back(2);
        for (uint32_t jj = 0; jj < 2; ++jj) {
            const uint32_t c = ch * CT + jb + jj;
            noc.async_read(h_acc, cbh, 4096, {.page_id = r * NTD + c, .offset_bytes = 0}, {.offset_bytes = jj * 4096});
            noc.async_read(w_acc, cbw, 2048, {.page_id = c, .offset_bytes = 0}, {.offset_bytes = jj * 2048});
        }
        noc.async_read_barrier();
        cbh.push_back(2);
        cbw.push_back(2);
    }
}
