// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// DSV41_ROPE_FULL_FUSE (tt/cmp_fuse.py::rope_full): CompactRope.full as one program. row64 [T,1,64] bf16 row-major (the
// 64 rotated columns of one table of T positions)
// -> the full-width bf16 TILE table the layers read: width W (512 or 128), columns 0..W-65 hold CONST (1.0 for cos, 0
// for sin), the last 64 the row. MODE 0: shape [A,T,1,W] / [T,1,1,W]: one tile per (user t, column tile ct), row 0
// valid, page t * NCT + ct.   MODE 1: shape [1,1,T,W]: one tile row of T valid rows, page ct. Output tiles are zero
// outside the valid rows / columns, as tilize pads (bit-identical to concat + reshape + to_layout(TILE)). Cores loop
// over pages p = core, core + NC, ...
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

typedef volatile tt_l1_ptr uint16_t* PW16;
typedef volatile tt_l1_ptr uint32_t* PW32;

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t NCT = get_compile_time_arg_val(1);  // W / 32
    constexpr uint32_t MODE = get_compile_time_arg_val(2);
    constexpr uint32_t COS = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t NP = get_compile_time_arg_val(5);  // pages
    constexpr uint32_t NC = get_compile_time_arg_val(6);  // cores
    constexpr auto r_args = TensorAccessorArgs<7>();
    constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    Noc noc;
    const auto r_acc = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), 128);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), 2048);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);  // one CB of 2048 + T * 128 bytes (page size = total)
    const uint32_t base = cb.get_write_ptr();
    PW16 O = reinterpret_cast<PW16>(base);
    PW16 R = reinterpret_cast<PW16>(base + 2048);
    PW32 O32 = reinterpret_cast<PW32>(base);
    if constexpr (MODE == 1) {
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_read(r_acc, cb, 128, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 2048 + t * 128});
        }
        noc.async_read_barrier();
    }
    for (uint32_t p = core; p < NP; p += NC) {
        for (uint32_t i = 0; i < 512; ++i) {
            O32[i] = 0;
        }
        if constexpr (MODE == 0) {
            const uint32_t t = p / NCT, ct = p % NCT;
            if (ct + 2 < NCT) {
                if (COS) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        O[(c >> 4) * 256 + (c & 15)] = 0x3F80;
                    }
                }
            } else {
                noc.async_read(r_acc, cb, 128, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 2048});
                noc.async_read_barrier();
                const uint32_t piece = ct + 2 - NCT;
                for (uint32_t c = 0; c < 32; ++c) {
                    O[(c >> 4) * 256 + (c & 15)] = R[piece * 32 + c];
                }
            }
        } else {
            const uint32_t ct = p;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t rb = (((t >> 4) << 1) * 256) + (t & 15) * 16;
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t dst = rb + (c >> 4) * 256 + (c & 15);
                    if (ct + 2 < NCT) {
                        O[dst] = COS ? 0x3F80 : 0;
                    } else {
                        O[dst] = R[t * 64 + (ct + 2 - NCT) * 32 + c];
                    }
                }
            }
        }
        noc.async_write(cb, o_acc, 2048, {.offset_bytes = 0}, {.page_id = p, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
