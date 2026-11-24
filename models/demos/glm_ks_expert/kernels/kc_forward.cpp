// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 x forwarder (NCRISC, NOC0) on a gate/up core - the flat expert's gate_up_x_forwarder without its weight
// reads: each arrived piece (XARR) is unicast into the successor's x ring once the successor has freed the block's
// slot (SFREE), DEPTH pieces in flight, the successor's XARR bumped after the ack; min(consumed (XCONS, from the
// receiver), forwarded) blocks are reported upstream (the predecessor's SFREE, or the relay's RFREE on a chain head).
// With W_STREAM it also reads this core's weights (as the flat expert's gate_up_weight_read.hpp): NE experts x NB
// blocks from its DRAM bank, WPIECES reads a block, WDEPTH in flight (transaction ids 8 ..), into the weight ring (CB 1,
// this RISC its producer) as slots free up, each block pushed once its pieces have landed. (The DRAM region holds one
// expert's blocks: every expert re-reads it - the same bytes a call as distinct experts.)
// RT: 0 x ring address, 1 predecessor xy (0: head), 2 successor xy (0: tail), 3 relay xy, 4 weights' bank address
//     (this core's region), 5 bank id
#include "api/dataflow/dataflow_api.h"
#include "kc_common.hpp"

using namespace kc;

void kernel_main() {
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t pxy = get_arg_val<uint32_t>(1), sxy = get_arg_val<uint32_t>(2), rxy = get_arg_val<uint32_t>(3);
    const uint64_t up_free = pxy ? kc_xy(pxy, get_semaphore(SFREE)) : kc_xy(rxy, get_semaphore(RFREE));
    const uint64_t s_ring = kc_xy(sxy, ring), s_xarr = kc_xy(sxy, get_semaphore(XARR));
    volatile tt_l1_ptr uint32_t* xarr = kc_sem(XARR);
    volatile tt_l1_ptr uint32_t* sfree = kc_sem(SFREE);
    volatile tt_l1_ptr uint32_t* xcons = kc_sem(XCONS);
    constexpr uint32_t total = TOTAL_BLOCKS * XP;
    uint32_t iss = 0, fwd = 0, rep = 0;
    // weight reads
    const uint32_t w_src = get_arg_val<uint32_t>(4), w_bank = get_arg_val<uint32_t>(5);
    const uint32_t w_ring = get_write_ptr(W_CB);
    constexpr uint32_t W_TOTAL = W_STREAM ? NE * NB * WPIECES : 0;
    constexpr uint32_t W_PIECE = WBLK_BYTES / WPIECES;
    uint32_t wi = 0, wd = 0, wpushed = 0;
    while (true) {
        invalidate_l1_cache();
        if constexpr (W_STREAM) {
            while (wi < W_TOTAL && wi - wd < WDEPTH &&
                   cb_pages_reservable_at_back(W_CB, (wi / WPIECES - wpushed + 1) * WBLK)) {
                const uint32_t blk = wi / WPIECES, pc = wi % WPIECES;
                noc_async_read_set_trid(8 + wi % WDEPTH);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(w_bank, w_src + (blk % NB) * WBLK_BYTES + pc * W_PIECE),
                    w_ring + (blk % WS) * WBLK_BYTES + pc * W_PIECE,
                    W_PIECE);
                ++wi;
            }
            while (wd < wi && !noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
            }
            while (wd < wi && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 8 + wd % WDEPTH)) {
                ++wd;
                if (wd % WPIECES == 0) {
                    cb_push_back(W_CB, WBLK);
                    ++wpushed;
                }
            }
        }
        if (sxy) {
            const uint32_t arrived = *xarr;
            while (iss < arrived && iss - fwd < DEPTH && (iss / XP < XS || *sfree >= iss / XP + 1 - XS)) {
                const uint32_t off = ((iss / XP) % XS) * X_BYTES + (iss % XP) * PIECE;
                for (uint32_t o = 0; o < PIECE; o += NOC_MAX_BURST_SIZE) {
                    const uint32_t n = PIECE - o < NOC_MAX_BURST_SIZE ? PIECE - o : NOC_MAX_BURST_SIZE;
                    noc_async_write_one_packet_with_trid(ring + off + o, s_ring + off + o, n, 1 + iss % DEPTH);
                }
                ++iss;
            }
            while (fwd < iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            }
            while (fwd < iss && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + fwd % DEPTH)) {
                noc_semaphore_inc(s_xarr, 1);
                ++fwd;
            }
        }
        const uint32_t cons = *xcons;
        const uint32_t freed = sxy && fwd / XP < cons ? fwd / XP : cons;
        if (freed > rep) {
            noc_semaphore_inc(up_free, freed - rep);
            rep = freed;
        }
        if (rep == TOTAL_BLOCKS && (!sxy || fwd == total) && wd == W_TOTAL) {
            break;
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_write_set_trid(0);
}
