// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 TP32 prefill rs-exact tail, reader: core t (token tile-row t of the own n tokens) reads, for each of the NCOL
// column tiles c, the 4 row-partial tiles (block k, tile row k * BLK + rowbase + t, col c) of the gathered fp32 [4 Tt,
// Dc] tensor; rowbase = this chip's mesh row * n / 32 (a per-device word read from the `sel` tensor, page 0). CT: 0
// cb_in, 1 cb_sel, 2 BLK, 3 NCOL, accessors yg, sel. Common: addr yg, addr sel. RT: t.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(1);
    constexpr uint32_t BLK = get_compile_time_arg_val(2);
    constexpr uint32_t NCOL = get_compile_time_arg_val(3);
    constexpr auto y_args = TensorAccessorArgs<4>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t t = get_arg_val<uint32_t>(0);
    const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto s_acc = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), 128);
    Noc noc;
    experimental::CB cin(cb_in), csel(cb_sel);
    csel.reserve_back(1);
    noc.async_read(s_acc, csel, 128, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t rowbase = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csel.get_write_ptr());
    for (uint32_t c = 0; c < NCOL; ++c) {
        cin.reserve_back(4);
        for (uint32_t k = 0; k < 4; ++k) {
            noc.async_read(
                y_acc,
                cin,
                4096,
                {.page_id = (k * BLK + rowbase + t) * NCOL + c, .offset_bytes = 0},
                {.offset_bytes = k * 4096});
        }
        noc.async_read_barrier();
        cin.push_back(4);
    }
}
