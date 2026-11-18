// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K5 (expand) reader: core i = r * NCH + ch: the 4 post tiles of row r (built from the pp tile, page r;
// resident), then per column tile c: y (bf16 tile page r*NTD + c) and the 4 stream tiles x_s. CT: 0 cb_x, 1 cb_y, 2
// cb_s, 3 TR, 4 NTD, 5 CT, 6 NCH, 7 cb_p (scratch for the pp tile), accessors x, y, pp. Common: addresses. RT: i.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t TR = get_compile_time_arg_val(3);
    constexpr uint32_t NTD = get_compile_time_arg_val(4);
    constexpr uint32_t CT = get_compile_time_arg_val(5);
    constexpr uint32_t NCH = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t r = i / NCH, ch = i % NCH;
    const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto s_acc = TensorAccessor(s_args, get_common_arg_val<uint32_t>(2), 4096);
    Noc noc;
    experimental::CB cbx(cb_x), cby(cb_y), cbs(cb_s), cbp(cb_p);

    // pp tile (page r): column k of the 32 x 32 fp32 tile -> tile with that value replicated over all columns (exact
    // bit copies)
    cbp.reserve_back(1);
    noc.async_read(s_acc, cbp, 4096, {.page_id = r, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cbs.reserve_back(4);
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp.get_write_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t t = 0; t < 32; ++t) {
                const uint32_t v = src[((t >> 4) << 9) + ((t & 15) << 4) + (4 + k < 16 ? 4 + k : 256 + 4 + k - 16)];
                const uint32_t row = ((t >> 4) << 9) + ((t & 15) << 4);
                volatile tt_l1_ptr uint32_t* d = dst + k * 1024 + row;
                for (uint32_t c = 0; c < 16; ++c) {
                    d[c] = v;
                    d[256 + c] = v;
                }
            }
        }
    }
    cbs.push_back(4);
    for (uint32_t jb = 0; jb < CT; jb += 2) {
        cbx.reserve_back(8);
        cby.reserve_back(2);
        for (uint32_t jj = 0; jj < 2; ++jj) {
            const uint32_t c = ch * CT + jb + jj;
            noc.async_read(y_acc, cby, 2048, {.page_id = r * NTD + c, .offset_bytes = 0}, {.offset_bytes = jj * 2048});
            for (uint32_t s = 0; s < 4; ++s) {
                noc.async_read(
                    x_acc,
                    cbx,
                    4096,
                    {.page_id = s * TR * NTD + r * NTD + c, .offset_bytes = 0},
                    {.offset_bytes = (jj * 4 + s) * 4096});
            }
        }
        noc.async_read_barrier();
        cbx.push_back(8);
        cby.push_back(2);
    }
}
