// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// DSV41_CMP_FUSE, ratio-2 compressor of the spec verify block (tt/cmp_fuse.py::compress_shift): one core per fp32 tile
// column ct. out[r] = (r % N == 0) ? prev[r / N] : cs[r - 1]   for the T = U * N valid rows (user-major), the other
// rows of the tile 0. Pure data movement on fp32 TILE tensors (face layout: 4 faces of 16 x 16, row r of face half h at
// ((r>>4)*2+h)*256 + (r&15)*16 words): bit-identical to to_layout(cs) / to_layout(prev) / concat / slice / reshape /
// to_layout(TILE) of the composite it replaces.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

typedef volatile tt_l1_ptr uint32_t* PW32;

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t U = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr auto cs_args = TensorAccessorArgs<3>();
    constexpr auto pv_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<pv_args.next_compile_time_args_offset()>();
    const uint32_t ct = get_arg_val<uint32_t>(0);
    Noc noc;
    const auto cs_acc = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto pv_acc = TensorAccessor(pv_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), 4096);
    experimental::CB cb(cb_id);
    cb.reserve_back(3);
    noc.async_read(cs_acc, cb, 4096, {.page_id = ct, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(pv_acc, cb, 4096, {.page_id = ct, .offset_bytes = 0}, {.offset_bytes = 4096});
    noc.async_read_barrier();
    const uint32_t base = cb.get_write_ptr();
    PW32 A = reinterpret_cast<PW32>(base);
    PW32 P = reinterpret_cast<PW32>(base + 4096);
    PW32 O = reinterpret_cast<PW32>(base + 8192);
    for (uint32_t i = 0; i < 1024; ++i) {
        O[i] = 0;
    }
    for (uint32_t u = 0; u < U; ++u) {
        for (uint32_t j = 0; j < N; ++j) {
            const uint32_t r = u * N + j;
            PW32 src = (j == 0) ? P : A;
            const uint32_t sr = (j == 0) ? u : r - 1;
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t so = (((sr >> 4) << 1) + h) * 256 + (sr & 15) * 16;
                const uint32_t dof = (((r >> 4) << 1) + h) * 256 + (r & 15) * 16;
                for (uint32_t w = 0; w < 16; ++w) {
                    O[dof + w] = src[so + w];
                }
            }
        }
    }
    noc.async_write(cb, o_acc, 4096, {.offset_bytes = 8192}, {.page_id = ct, .offset_bytes = 0});
    noc.async_write_barrier();
}
