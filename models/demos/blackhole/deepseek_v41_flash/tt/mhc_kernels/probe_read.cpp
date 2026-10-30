// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Probe (profiling only, not used by the model): the x gather of the mHC projection reader with different NoC issue
// strategies. Core r reads, for the TPJ column tiles r*TPJ.. and all T tokens, the two valid 256 B pieces of page t*NT
// + j.
//  MODE 0: noc.async_read with the TensorAccessor (the current reader)
//  MODE 1: per column tile one set_async_read_state (all tokens of a column tile live in one DRAM bank when NB | NT) +
//  with_state reads, addresses stepped by arithmetic MODE 2: like 1 but ONE 1280 B read per (token, tile) (face 0 piece
//  .. face 1 piece, 768 B of padding included) MODE 3: like 1 but whole 4096 B pages MODE 4: like 0 but only the first
//  piece of every page (half the reads)

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t TPJ = get_compile_time_arg_val(3);
    constexpr uint32_t MODE = get_compile_time_arg_val(4);
    constexpr uint32_t NB = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    experimental::CB cbx(cb_x);
    cbx.reserve_back(1);
    const uint32_t base = cbx.get_write_ptr();
    DeviceZoneScopedN("PB_ALL");
    if constexpr (MODE == 0 || MODE == 4) {
        for (uint32_t j = 0; j < TPJ; ++j) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t page = t * NT + r * TPJ + j;
                const uint32_t dst = (j * T + t) * 512;
                noc.async_read(x_acc, cbx, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = dst});
                if constexpr (MODE == 0) {
                    noc.async_read(
                        x_acc, cbx, 256, {.page_id = page, .offset_bytes = 1024}, {.offset_bytes = dst + 256});
                }
            }
        }
    } else {
        constexpr uint32_t SZ = MODE == 1 ? 256 : MODE == 2 ? 1280 : 4096;
        constexpr uint32_t STEP = (NT / NB) * TILE;  // address step between the pages of consecutive tokens (same bank)
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t page0 = r * TPJ + j;
            const uint64_t a0 = x_acc.get_noc_addr(page0, 0);
            noc_async_read_one_packet_set_state(a0, SZ);
            uint32_t a = (uint32_t)a0;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t dst = base + (j * T + t) * (SZ == 256 ? 512 : SZ);
                noc_async_read_one_packet_with_state<true>(a, dst);
                if constexpr (MODE == 1) {
                    noc_async_read_one_packet_with_state<true>(a + 1024, dst + 256);
                }
                a += STEP;
            }
        }
    }
    {
        DeviceZoneScopedN("PB_issued");
    }
    noc.async_read_barrier();
}
