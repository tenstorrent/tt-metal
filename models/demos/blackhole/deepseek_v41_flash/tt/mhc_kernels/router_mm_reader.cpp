// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 router matmul, reader (NCRISC).  Core (n, s) owns output column tile n and the K slice s of S (KS tiles of 32
// rows of the 5120-wide hidden dim): it reads the x tiles [s*KS, (s+1)*KS) (only the first XB bytes of each bf16 tile
// page: 1 KB = rows 0..15, enough for T <= 16) and the KS weight tiles of its slice (pre-arranged: page j * NC + core,
// so that step j of all cores reads consecutive pages), in blocks of B tiles so that compute starts after the first
// block.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    DeviceZoneScopedN("RR_ALL");
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t KS = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t XB = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);  // cores per K step
    constexpr uint32_t S = NC / 12;
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048;

    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t kx0 = s * KS;     // first x tile page of the slice
    const uint32_t kw0 = n * S + s;  // weight core index
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w);
    cbx.reserve_back(KS);
    cbw.reserve_back(KS);
    for (uint32_t b = 0; b < KS / B; ++b) {
        for (uint32_t j = b * B; j < (b + 1) * B; ++j) {
            noc.async_read(
                w_acc, cbw, TILE, {.page_id = j * NC + kw0, .offset_bytes = 0}, {.offset_bytes = (j - b * B) * TILE});
            noc.async_read(
                x_acc, cbx, XB, {.page_id = kx0 + j, .offset_bytes = 0}, {.offset_bytes = (j - b * B) * TILE});
        }
        noc.async_read_barrier();
        cbw.push_back(B);
        cbx.push_back(B);
    }
}
