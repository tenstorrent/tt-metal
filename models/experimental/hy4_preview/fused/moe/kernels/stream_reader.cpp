// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert weight streamer of the Hy4 decode MoE program (NCRISC, NOC0), for the gate|up and the down cores alike
// (fused/moe/program.py). Derived from models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_union_reader.cpp.
//
// A DRAM bank holds, per expert, `pb` items of item_bytes for the cores of that bank (layout.py). The union of picked
// experts ([U, ids] in the meta page) makes the work list v = slot * pb + t; this core streams the items with
// v % modulus == residue, in order, each as item_blocks blocks of block_bytes (n_pk equal NOC packets per block)
// into a ring of ring_blocks blocks, extra_in_flight + 1 blocks outstanding (one transaction id per block).
// With PLAIN_TERMS = 2 a plain slot (ctl[1] = the first one) is stored as two consecutive entries (the weight as
// two BFP4 terms, layout.py two_terms): its item is followed by the same item of the next entry.
//
// runtime args: [w_addr, bank_id, vc, residue, modulus, core_offset_bytes]
// compile-time args: [cb_w, cb_ctl, cb_meta, pb, item_bytes, item_blocks, block_tiles, block_bytes, n_pk, ring_blocks,
//                     extra_in_flight, expert_stride_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "moe_common.hpp"

#ifndef PLAIN_TERMS
#define PLAIN_TERMS 1
#endif

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t residue = get_arg_val<uint32_t>(3);
    const uint32_t modulus = get_arg_val<uint32_t>(4);
    const uint32_t core_offset = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t pb = get_compile_time_arg_val(3);
    constexpr uint32_t item_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t item_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t n_pk = get_compile_time_arg_val(8);
    constexpr uint32_t ring_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t extra_in_flight = get_compile_time_arg_val(10);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(11);

    constexpr uint32_t pkt = block_bytes / n_pk;
    static_assert(pkt * n_pk == block_bytes && pkt <= NOC_MAX_BURST_SIZE, "a block is n_pk equal packets");
    static_assert(ring_blocks >= extra_in_flight + 2 && ring_blocks <= 15, "ring size (one transaction id per block)");

    const uint32_t cb_base = get_write_ptr(cb_w);  // the ring's first block (nothing reserved yet)
    const uint32_t cb_end = cb_base + ring_blocks * block_bytes;
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    [[maybe_unused]] volatile tt_l1_ptr uint16_t* ids =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_meta) + hy4moe::META_IDS);

    // BRISC saw the hub's release (or gave up): ctl[0] = U
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_LOCAL)), 1);
    invalidate_l1_cache();
    const uint32_t total = ctl[0] * pb;
    if (residue >= total) {
        return;
    }
#if PLAIN_TERMS > 1
    const uint32_t first_plain = ctl[1];  // slots from here on are stored as PLAIN_TERMS consecutive entries
#endif

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);

    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = cb_base;
    uint32_t free_blocks = ring_blocks;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t issued = 0;
    for (uint32_t v = residue; v < total; v += modulus) {
        const uint32_t slot = v / pb;
        const uint32_t t = v - slot * pb;
#ifdef STREAM_FIXED  // one fixed item per unit of ctl[0] (the router's weights): no union ids
        uint32_t src = slot * expert_stride + t * item_bytes + core_offset;
#else
        uint32_t src = static_cast<uint32_t>(ids[slot]) * expert_stride + t * item_bytes + core_offset;
#endif
#if PLAIN_TERMS > 1
        // a plain slot's further terms are the same item of the next entries
        const uint32_t n_blocks = slot >= first_plain ? PLAIN_TERMS * item_blocks : item_blocks;
#else
        constexpr uint32_t n_blocks = item_blocks;
#endif
        for (uint32_t blk = 0; blk < n_blocks; ++blk) {
#if PLAIN_TERMS > 1
            if (blk > 0 && blk % item_blocks == 0) {
                src += expert_stride - item_bytes;  // the same item of the next entry
            }
#endif
#ifndef HY4_NO_READ
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < n_pk; ++p) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                src += pkt;
                l1_write += pkt;
            }
#else  // measurement only: hand compute whatever the ring holds
            l1_write += block_bytes;
#endif
            ++issued;
            if (free_blocks == ring_blocks - extra_in_flight) {
#ifndef HY4_NO_READ
                noc_async_read_barrier_with_trid(trid_to_wait);
#endif
                cb_push_back(cb_w, block_tiles);
                trid_to_wait = trid_to_wait == ring_blocks ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == ring_blocks ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    // blocks issued but not yet pushed
    const uint32_t pending = issued < extra_in_flight ? issued : extra_in_flight;
    for (uint32_t i = 0; i < pending; ++i) {
#ifndef HY4_NO_READ
        noc_async_read_barrier_with_trid(trid_to_wait);
#endif
        cb_push_back(cb_w, block_tiles);
        trid_to_wait = trid_to_wait == ring_blocks ? 1 : trid_to_wait + 1;
    }
}
