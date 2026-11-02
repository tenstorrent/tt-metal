// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// DSV41_CMP_FUSE, commit of the ratio-2 compressor state (tt/cmp_fuse.py::commit_select): one core per fp32 tile column
// ct. prev[u] <- cs[u * N + m[u]] for the U users (m[u] = word 0 of row u of the uint32 [U,1,1024] index tensor, or N -
// 1 when MODE == 0); rows >= U of the prev tile are left as they are (read-modify-write of the tile: a decode-bucket
// view of prev_cs shares the buffer with the other users' rows).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

typedef volatile tt_l1_ptr uint32_t* PW32;

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t U = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t MODE = get_compile_time_arg_val(3);
    constexpr auto cs_args = TensorAccessorArgs<4>();
    constexpr auto ix_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    constexpr auto pv_args = TensorAccessorArgs<ix_args.next_compile_time_args_offset()>();
    const uint32_t ct = get_arg_val<uint32_t>(0);
    Noc noc;
    const auto cs_acc = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto ix_acc = TensorAccessor(ix_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto pv_acc = TensorAccessor(pv_args, get_common_arg_val<uint32_t>(2), 4096);
    experimental::CB cb(cb_id);
    cb.reserve_back(4);
    noc.async_read(cs_acc, cb, 4096, {.page_id = ct, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(pv_acc, cb, 4096, {.page_id = ct, .offset_bytes = 0}, {.offset_bytes = 4096});
    if constexpr (MODE == 1) {
        for (uint32_t u = 0; u < U; ++u) {
            noc.async_read(ix_acc, cb, 64, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = 8192 + u * 64});
        }
    }
    noc.async_read_barrier();
    const uint32_t base = cb.get_write_ptr();
    PW32 A = reinterpret_cast<PW32>(base);
    PW32 P = reinterpret_cast<PW32>(base + 4096);
    PW32 X = reinterpret_cast<PW32>(base + 8192);
    for (uint32_t u = 0; u < U; ++u) {
        uint32_t m = N - 1;
        if constexpr (MODE == 1) {
            m = X[u * 16];
            if (m > N - 1) {
                m = N - 1;
            }
        }
        const uint32_t sr = u * N + m;
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t so = (((sr >> 4) << 1) + h) * 256 + (sr & 15) * 16;
            const uint32_t dof = (((u >> 4) << 1) + h) * 256 + (u & 15) * 16;
            for (uint32_t w = 0; w < 16; ++w) {
                P[dof + w] = A[so + w];
            }
        }
    }
    noc.async_write(cb, pv_acc, 4096, {.offset_bytes = 4096}, {.page_id = ct, .offset_bytes = 0});
    noc.async_write_barrier();
}
