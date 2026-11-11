// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// T19 packed-mHC mixes, reader.  Core r owns token tile row r: builds an identity tile and the "sum-of-squares select"
// tile (row NCOL all ones), then streams the S partial tiles (r, s) of pk_proj into cb_part.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_eye = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t NCOL = get_compile_time_arg_val(4);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(5);  // tiles per read batch
    constexpr auto p_args = TensorAccessorArgs<6>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cbp(cb_part), cbe(cb_eye), cbs(cb_sel);

    cbe.reserve_back(1);
    noc.async_write_zeros(cbe, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbe.get_write_ptr());
        for (uint32_t k = 0; k < 32; ++k) {
            const uint32_t face = (k >= 16) ? 3 : 0;
            o[face * 256 + (k % 16) * 16 + (k % 16)] = 0x3F800000u;
        }
    }
    cbe.push_back(1);

    cbs.reserve_back(1);
    noc.async_write_zeros(cbs, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t face = ((NCOL >= 16) ? 2 : 0) + ((j >= 16) ? 1 : 0);
            o[face * 256 + (NCOL % 16) * 16 + (j % 16)] = 0x3F800000u;
        }
    }
    cbs.push_back(1);

    for (uint32_t s0 = 0; s0 < S; s0 += DEPTH) {
        cbp.reserve_back(DEPTH);
        for (uint32_t i = 0; i < DEPTH; ++i) {
            noc.async_read(
                p_acc, cbp, TILE, {.page_id = r * S + s0 + i, .offset_bytes = 0}, {.offset_bytes = i * TILE});
        }
        noc.async_read_barrier();
        cbp.push_back(DEPTH);
    }
}
