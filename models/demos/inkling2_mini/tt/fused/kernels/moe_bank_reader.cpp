// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Bank-streamed MoE expert step, reader (NCRISC, NOC0), one worker core next to DRAM bank `bank`
// (tt/fused/moe_bank.py; weight layout: convert/bank_layout.py). Every weight read is from this core's own bank.
// 1. ids, routing weights (row 0 of their TILE tensors); pair cores also x (row 0 of its 72 tiles, as 1x32 tiles)
//    and the routing-weight tile (face 0 = 8 tiny tiles, w_e repeated), as moe_expert_reader.cpp.
// 2. Cores without a gate_up column first issue their down column for all 8 experts (cb_wd holds it all), so the
//    bank stays busy while the others stream gate_up.
// 3. gate_up cores (the gate or the up column of pair slot kslot; the extra pair slot only for experts e with
//    extra_bank(e) == bank, other experts' blocks are pushed unread and their outputs never sent):
//      for e: the column (kt_gu tiles), blocks of SBK, nbuf - 1 blocks in flight; then their down column.
// Runtime args: x, routing weights, ids, w_gate_up, w_down addresses; flags (F_* below); bank; kslot;
//   col_off (0: gate column, kt_gu: up column); dn slot.
// IDS_FROM_SEM (for a program that routes in the same program, tt/mk): ids (cb_scr bytes 0..15) and routing
//   weights (32..47) arrive by multicast and sem_ids is set, instead of being read from their tensors.
// X_MCAST: x is not read here; the collector multicasts it into cb_x and the writer pushes it, so the weight
//   stream starts right after the ids (82 cores no longer queue for the x tensor's core).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream.hpp"

constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_wd = get_named_compile_time_arg_val("cb_wd");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
constexpr uint32_t cb_scr = get_named_compile_time_arg_val("cb_scr");
constexpr uint32_t sbk = get_named_compile_time_arg_val("sbk");
constexpr uint32_t nbuf = get_named_compile_time_arg_val("nbuf");
constexpr uint32_t kt_gu = get_named_compile_time_arg_val("kt_gu");  // 72
constexpr uint32_t kt_dn = get_named_compile_time_arg_val("kt_dn");  // 36
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");    // 8
constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");         // 7
constexpr uint32_t base = get_named_compile_time_arg_val("base");          // base pairs per bank per expert (5)
constexpr uint32_t x_lo = get_named_compile_time_arg_val("x_lo");          // first bank holding extra pairs (2)
constexpr uint32_t dn_slots = get_named_compile_time_arg_val("dn_slots");  // down column slots per expert (11)

constexpr uint32_t F_GU = 1, F_GATE = 2, F_EXTRA = 4, F_DN = 8;
#ifdef IDS_FROM_SEM
constexpr uint32_t sem_ids_id = get_named_compile_time_arg_val("sem_ids");
#endif
constexpr uint32_t TINY = 64;
constexpr uint32_t DN_TRID0 = 8;  // transaction ids of the down reads (one per expert)
constexpr uint32_t FACE_BYTES = 512;

// convert/bank_layout.py extra_bank / gu_slot (one extra pair per expert)
FORCE_INLINE uint32_t extra_bank(uint32_t e) { return x_lo + e % (nb - x_lo); }
FORCE_INLINE uint32_t gu_slot(uint32_t e, uint32_t b, uint32_t k) {
    const uint32_t before = base * e + (b >= x_lo ? (e + (nb - x_lo) - 1 - (b - x_lo)) / (nb - x_lo) : 0);
    return before + k;
}

void kernel_main() {
    uint32_t a = 0;
    [[maybe_unused]] const uint32_t x_addr = get_arg_val<uint32_t>(a++);  // unused with X_MCAST
    const uint32_t wts_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gu_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t flags = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    const uint32_t kslot = get_arg_val<uint32_t>(a++);
    const uint32_t col_off = get_arg_val<uint32_t>(a++);
    const uint32_t dslot = get_arg_val<uint32_t>(a++);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto wts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
#ifndef X_MCAST
    const auto xs = TensorAccessor(x_args, x_addr);
#endif
    const auto wts = TensorAccessor(wts_args, wts_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto gu = TensorAccessor(gu_args, gu_addr);
    const auto dn = TensorAccessor(dn_args, dn_addr);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    const uint32_t scr = get_write_ptr(cb_scr);
#ifdef IDS_FROM_SEM
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ids_id)), 1);
    invalidate_l1_cache();
#else
    noc_async_read(ids.get_noc_addr(0), scr, 32);
    if (flags & F_GU) {
        noc_async_read(wts.get_noc_addr(0), scr + 32, 32);
    }
#endif
#ifdef X_MCAST
    noc_async_read_barrier();
#else
    if (flags & F_GU) {
        cb_reserve_back(cb_x, x_tiles);
        const uint32_t xl = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < x_tiles; ++t) {
            noc_async_read(xs.get_noc_addr(t, 0), xl + t * TINY, 32);
            noc_async_read(xs.get_noc_addr(t, FACE_BYTES), xl + t * TINY + 32, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, x_tiles);
    } else {
        noc_async_read_barrier();
    }
#endif

    uint32_t eid[topk];
    volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t e = 0; e < topk; ++e) {
        eid[e] = id16[e];
    }

    // down column of every routed expert, [e][kt] (the gathered hidden is [e][kt]): one run per expert, each on its
    // own transaction id, pushed as it lands, so the down chain runs behind the reads instead of after them
    static_assert(nbuf < DN_TRID0 && DN_TRID0 + topk - 1 <= NOC_MAX_TRANSACTION_ID, "stream: 1..nbuf, down: 8..15");
    auto read_down = [&]() {
        const uint32_t tb = get_tile_size(cb_wd);
        cb_reserve_back(cb_wd, topk * kt_dn);
        uint32_t wr = get_write_ptr(cb_wd);
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t slot = (eid[e] * dn_slots + dslot) * kt_dn;
            noc_async_read_set_trid(DN_TRID0 + e);
            noc_async_read(dn.get_noc_addr(slot * nb + bank), wr, kt_dn * tb);
            wr += kt_dn * tb;
        }
        for (uint32_t e = 0; e < topk; ++e) {
            noc_async_read_barrier_with_trid(DN_TRID0 + e);
            cb_push_back(cb_wd, kt_dn);
        }
        noc_async_read_set_trid(0);
    };

    if (!(flags & F_GU)) {
        if (flags & F_DN) {
            read_down();
        }
        return;
    }

    if (flags & F_GATE) {
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 32);
        cb_reserve_back(cb_ws16, 1);
        volatile tt_l1_ptr uint32_t* ws = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ws16));
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t v = w16[e];
            const uint32_t pair = v | (v << 16);
            for (uint32_t i = 0; i < 16; ++i) {
                ws[e * 16 + i] = pair;
            }
        }
        cb_push_back(cb_ws16, 1);
    }

    Stream<sbk, nbuf> s;
    s.open(cb_w);
    for (uint32_t e = 0; e < topk; ++e) {
        const bool active = !(flags & F_EXTRA) || extra_bank(eid[e]) == bank;
        // pair = gate column then up column, 2 * kt_gu tiles contiguous in this bank; this core's column
        const uint64_t src = gu.get_noc_addr((gu_slot(eid[e], bank, kslot) * 2 * kt_gu + col_off) * nb + bank);
        for (uint32_t t = 0; t < kt_gu; t += sbk) {
            if (active) {
                s.block_contig(src + t * s.tile_bytes);
            } else {  // not this expert's extra bank: compute runs on stale tiles, the writer drops the result
                s.skip();
            }
        }
    }
    s.drain();
    if (flags & F_DN) {
        read_down();
    }
    noc_async_read_set_trid(0);
}
