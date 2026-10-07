// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reads NB-tile blocks of the bf16 TILE tensor x and builds the reduce scaler tile (1.0 in row 0 of every face).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_one = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t blk0 = get_arg_val<uint32_t>(0);
    const uint32_t nblk = get_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    experimental::CB cbx(cb_x), cbo(cb_one);

    cbo.reserve_back(1);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
    for (uint32_t i = 0; i < 512; ++i) {
        z[i] = 0;
    }
    volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbo.get_write_ptr());
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t j = 0; j < 16; ++j) {
            o[f * 256 + j] = 0x3F80;
        }
    }
    cbo.push_back(1);

    for (uint32_t b = 0; b < nblk; ++b) {
        cbx.reserve_back(NB);
        for (uint32_t j = 0; j < NB; ++j) {
            noc.async_read(
                x_acc, cbx, 2048, {.page_id = (blk0 + b) * NB + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
        }
        noc.async_read_barrier();
        cbx.push_back(NB);
    }
}
