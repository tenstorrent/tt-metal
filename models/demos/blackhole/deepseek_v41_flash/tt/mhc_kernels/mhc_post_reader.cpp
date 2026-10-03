// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel, reader (one core): the 10 constant tiles, and rows 0..T-1 of each of the NPART partial tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_consts = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t NPART = get_compile_time_arg_val(2);
    constexpr uint32_t NCONST = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr auto c_args = TensorAccessorArgs<5>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t c_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cbc(cb_consts), cbp(cb_p);

    cbc.reserve_back(NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc.async_read(c_acc, cbc, TILE, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * TILE});
    }
    cbp.reserve_back(NPART);
    noc.async_write_zeros(cbp, NPART * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t s = 0; s < NPART; ++s) {
        constexpr uint32_t T0 = T < 16 ? T : 16;
        constexpr uint32_t T1 = T > 16 ? T - 16 : 0;
        noc.async_read(p_acc, cbp, T0 * 64, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = s * TILE});
        noc.async_read(p_acc, cbp, T0 * 64, {.page_id = s, .offset_bytes = 1024}, {.offset_bytes = s * TILE + 1024});
        if constexpr (T1 > 0) {
            noc.async_read(
                p_acc, cbp, T1 * 64, {.page_id = s, .offset_bytes = 2048}, {.offset_bytes = s * TILE + 2048});
            noc.async_read(
                p_acc, cbp, T1 * 64, {.page_id = s, .offset_bytes = 3072}, {.offset_bytes = s * TILE + 3072});
        }
    }
    noc.async_read_barrier();
    cbc.push_back(NCONST);
    cbp.push_back(NPART);
}
