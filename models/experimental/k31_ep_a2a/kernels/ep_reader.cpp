// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K3.1 EP all-to-all prototype — reader (NCRISC). Loads routing metadata, then streams exactly the rows
// this core's writer will send, in the writer's order, into data_cb. Reads are issued in batches so
// DRAM latency is overlapped (one barrier per BATCH rows).
#include "api/dataflow/dataflow_api.h"
#include "ep_common.hpp"

using namespace k31ep;

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t MY = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t EPD = get_compile_time_arg_val(5);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(7);
    constexpr uint32_t LINK = get_compile_time_arg_val(8);
    constexpr uint32_t ROLE = get_compile_time_arg_val(9) & 0xFF;
    constexpr bool RING = (get_compile_time_arg_val(9) >> 8) & 1;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(10);
    constexpr uint32_t data_cb = get_compile_time_arg_val(11);
    constexpr uint32_t BATCH = get_compile_time_arg_val(12);
    constexpr auto meta_args = TensorAccessorArgs<13>();
    constexpr auto row_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();

    const uint32_t meta_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t META_PAGE = (MODE == MODE_DISPATCH) ? META_BYTES : T * META_BYTES;
    constexpr uint32_t META_PAGES = (MODE == MODE_DISPATCH) ? T : N;
    const auto meta_src = TensorAccessor(meta_args, meta_addr, META_PAGE);
    const auto row_src = TensorAccessor(row_args, row_addr, ROW_BYTES);

    cb_reserve_back(meta_cb, 1);
    const uint32_t meta_l1 = get_write_ptr(meta_cb);
    for (uint32_t p = 0; p < META_PAGES; ++p) {
        noc_async_read(meta_src.get_noc_addr(p), meta_l1 + p * META_PAGE, META_PAGE);
    }
    noc_async_read_barrier();
    cb_push_back(meta_cb, 1);

    uint32_t in_batch = 0;
    uint32_t batch_base = 0;
    auto read_row = [&](uint32_t page) {
        if (in_batch == 0) {
            cb_reserve_back(data_cb, BATCH);
            batch_base = get_write_ptr(data_cb);
        }
        noc_async_read(row_src.get_noc_addr(page), batch_base + in_batch * ROW_BYTES, ROW_BYTES);
        if (++in_batch == BATCH) {
            noc_async_read_barrier();
            cb_push_back(data_cb, BATCH);
            in_batch = 0;
        }
    };

    if constexpr (MODE == MODE_DISPATCH) {
        for (uint32_t t = 0; t < T; ++t) {
            const auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + t * META_BYTES);
            uint32_t far;
            if (dispatch_need<ROLE, MY, L, LINK, N, RING>(t, hit_mask<K, EPD>(ids), far)) {
                read_row(t);
            }
        }
    } else {
        for (uint32_t s = 0; s < N; ++s) {
            for (uint32_t t = 0; t < T; ++t) {
                const auto* ids =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + (s * T + t) * META_BYTES);
                uint32_t hops;
                if (combine_need<ROLE, MY, L, LINK, N, RING>(s, t, hit_mask<K, EPD>(ids), hops)) {
                    read_row(s * T + t);
                }
            }
        }
    }
    if (in_batch) {
        noc_async_read_barrier();
        cb_push_back(data_cb, in_batch);
    }
}
