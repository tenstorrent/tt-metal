// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention, the union GATHER (tt/spa_gather.py), lane NCRISC: this chip's request entries (src_row,
// dst_slot, dst_pos) -- every pool of every chip's union that THIS chip owns, grouped by destination -- read in batches
// of EB entries; per entry the pool's 4 latent rows (1 KiB pages src_row .. src_row + 3 of the local row-major store)
// into the staging CB (one 4-KiB page per entry) and (dst_slot, dst_pos) into the meta CB, for the lane BRISC.
// Compile-time: CB_STAGE, CB_META, CB_LIST, EB, then the TensorAccessorArgs of the local store and of the entry list
// (uint32 ROW_MAJOR, one page). Runtime: store address, list address, entries.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t CB_LIST = get_compile_time_arg_val(2);
constexpr uint32_t EB = get_compile_time_arg_val(3);  // entries per list batch
constexpr uint32_t RB = 1024;                         // one latent row: 512 bf16

void kernel_main() {
    constexpr auto lat_args = TensorAccessorArgs<4>();
    constexpr auto lst_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    const uint32_t lat_addr = get_arg_val<uint32_t>(0);
    const uint32_t lst_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const auto lat = TensorAccessor(lat_args, lat_addr);
    const auto lst = TensorAccessor(lst_args, lst_addr);
    cb_reserve_back(CB_LIST, 1);  // the entry batch buffer (scratch, never pushed)
    const uint32_t lb = get_write_ptr(CB_LIST);
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lb);
    for (uint32_t i0 = 0; i0 < n; i0 += EB) {
        const uint32_t m = n - i0 < EB ? n - i0 : EB;
        noc_async_read(lst.get_noc_addr(0, i0 * 12), lb, ((m * 12) + 31) & ~31u);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < m; ++k) {
            const uint32_t src = e[3 * k], slot = e[3 * k + 1], pos = e[3 * k + 2];
            cb_reserve_back(CB_STAGE, 1);
            cb_reserve_back(CB_META, 1);
            const uint32_t sp = get_write_ptr(CB_STAGE);
            for (uint32_t j = 0; j < 4; ++j) {
                noc_async_read_page(src + j, lat, sp + j * RB);
            }
            volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META));
            mt[0] = slot;
            mt[1] = pos;
            noc_async_read_barrier();
            cb_push_back(CB_META, 1);
            cb_push_back(CB_STAGE, 1);
        }
    }
}
