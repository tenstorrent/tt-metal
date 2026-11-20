// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Test-only consumer of a converter's K tiles: writes every converted slab to a TILE tensor [USERS * TOPK, DH]
// BF16 (row = user * TOPK + list position), so the host can compare the rows the reader delivers with the host
// unpack of the same storage rows. The attention program replaces this kernel with the multicast sender
// (sm_mcast.cpp). SM_DROP (stage timing): the slabs are popped and not written anywhere.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "sm_dm.hpp"

void kernel_main() {
    constexpr uint32_t cb_kout = get_named_compile_time_arg_val("cb_kout");
    constexpr uint32_t cb_hdr = get_named_compile_time_arg_val("cb_hdr");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t NCONV = get_named_compile_time_arg_val("converters");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t TOPK = get_named_compile_time_arg_val("topk");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("dht");

    constexpr auto sel_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();

    const sm::Role role =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>();
    const uint32_t conv = role.slot;
    if (conv >= NCONV) {
        return;  // a core of the reader columns that holds no converter
    }
    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    const auto sel = TensorAccessor(sel_args, sel_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_kout);

    uint32_t counts[sm::MAX_USERS];
    sm::read_counts<USERS>(sel, cb_hdr, counts);

    sm::for_each_chunk<USERS, GROUPS, CHUNK>(counts, role.block, [&](uint32_t u, uint32_t c, uint32_t valid) {
        const uint32_t own = sm::own_slabs<NCONV>(sm::ceil_div(valid, sm::TILE_ROWS), conv);
        if (own == 0) {
            return;
        }
        // one slab at a time: the slabs of a chunk need not be contiguous in the FIFO (they may straddle its wrap)
        for (uint32_t i = 0; i < own; ++i) {
            cb_wait_front(cb_kout, DHt);
#ifndef SM_DROP
            const uint32_t src = get_read_ptr(cb_kout);
            const uint32_t slab_page = ((u * TOPK + c * CHUNK) / sm::TILE_ROWS + conv + i * NCONV) * DHt;
            for (uint32_t t = 0; t < DHt; ++t) {
                noc_async_write(src + t * tile_bytes, out.get_noc_addr(slab_page + t), tile_bytes);
            }
            noc_async_write_barrier();
#endif
            cb_pop_front(cb_kout, DHt);
        }
    });
}
