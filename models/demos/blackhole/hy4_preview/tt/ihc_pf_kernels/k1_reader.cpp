// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K1 (stats) reader: core i = r * NCH + ch reads, for s < 4 and the CT column tiles of chunk ch, the x
// tile (stream s, token tile-row r, col tile c) and the hc_fn tile (s, c). CT: 0 cb_x, 1 cb_w, 2 TR, 3 NTD, 4 CT, 5
// NCH, 6 B, accessors x, w. Common: addr x, addr w. RT: i.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t TR = get_compile_time_arg_val(2);
    constexpr uint32_t NTD = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t NCH = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t r = i / NCH, ch = i % NCH;
    const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto w_acc = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 4096);
    Noc noc;
    experimental::CB cbx(cb_x), cbw(cb_w);
    for (uint32_t s = 0; s < 4; ++s) {
        for (uint32_t jb = 0; jb < CT; jb += B) {
            cbx.reserve_back(B);
            cbw.reserve_back(B);
            for (uint32_t jj = 0; jj < B; ++jj) {
                const uint32_t c = ch * CT + jb + jj;
                noc.async_read(
                    x_acc,
                    cbx,
                    4096,
                    {.page_id = s * TR * NTD + r * NTD + c, .offset_bytes = 0},
                    {.offset_bytes = jj * 4096});
                noc.async_read(
                    w_acc, cbw, 4096, {.page_id = s * NTD + c, .offset_bytes = 0}, {.offset_bytes = jj * 4096});
            }
            noc.async_read_barrier();
            cbx.push_back(B);
            cbw.push_back(B);
        }
    }
}
