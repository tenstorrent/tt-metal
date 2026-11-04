// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: ascending sort of the indexer's selected entry ids, one core per row, as a BITMAP sort instead of the
// 3-pass radix sort (tt/mhc_kernels/ids_sort.cpp: 75 us per call at K = 512): the ids of a row are distinct (top-k
// positions) and < NW * 32: zero a bitmap of NW words, set bit id for every id (one load / or / store per id), then
// walk the bitmap and emit the set bits ascending (count trailing zeros). 0xFFFFFFFF sentinels (and any id >= NW * 32)
// are skipped and the row is completed with 0xFFFFFFFF. Same output as the radix sort for distinct ids. ids / out:
// uint32 ROW_MAJOR [R,1,1,K] pages of K * 4 bytes.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t NW = get_compile_time_arg_val(4);  // bitmap words (ids < NW * 32)
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t t_first =
        ol_core_idx(get_common_arg_val<uint32_t>(2));  // one row per core, no per-core runtime arguments
    const uint32_t t_stride = 1;
    const uint32_t count = 1;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    OLZ("Z_CALL");

    Noc noc;
    const auto in_acc = TensorAccessor(in_args, in_addr, K * 4);
    const auto out_acc = TensorAccessor(out_args, out_addr, K * 4);
    experimental::CB cba(cb_a), cbb(cb_b), cbc(cb_c);
    cba.reserve_back(1);
    cbb.reserve_back(1);
    cbc.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* bm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        noc.async_read(in_acc, cba, K * 4, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        {
            OLZ("Z_ZERO");
            for (uint32_t i = 0; i < NW; ++i) {
                bm[i] = 0;
            }
        }
        noc.async_read_barrier();
        {
            OLZ("Z_MARK");
#pragma GCC unroll 4
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t id = ids[i];
                if (id < NW * 32u) {
                    bm[id >> 5] |= 1u << (id & 31u);
                }
            }
        }
        uint32_t o = 0;
        {
            OLZ("Z_EMIT");
            for (uint32_t w = 0; w < NW; ++w) {
                uint32_t b = bm[w];
                while (b) {
                    const uint32_t tz = __builtin_ctz(b);
                    b &= b - 1;
                    po[o++] = (w << 5) + tz;
                }
            }
            for (; o < K; ++o) {
                po[o] = 0xFFFFFFFFu;
            }
        }
        noc.async_write(cbb, out_acc, K * 4, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
