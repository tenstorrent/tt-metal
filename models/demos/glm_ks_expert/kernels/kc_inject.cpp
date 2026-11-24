// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 x injector (NCRISC, NOC0) on a relay core: its chain's NB x blocks (one sub-block's, already tilized) are
// read from DRAM into L1 once; then for NSB sub-blocks it unicasts every block's pieces into the chain head's x ring
// (DEPTH pieces in flight, transaction ids), each block once the head has freed its slot (RFREE here, bumped by the
// head's forwarder), and bumps the head's XARR after each piece is acknowledged - the flat expert's relay -> snake
// head protocol with a relay that is never short of x.
// RT: 0 x DRAM address, 1 first x tile of this chain, 2 the local x buffer (L1, NB blocks), 3 head xy, 4 x ring
//     address (on the head)
// CT: TensorAccessorArgs(x)
#include "api/dataflow/dataflow_api.h"
#include "kc_common.hpp"

using namespace kc;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_first = get_arg_val<uint32_t>(1);
    const uint32_t buf = get_arg_val<uint32_t>(2);
    const uint32_t hxy = get_arg_val<uint32_t>(3);
    const uint32_t ring = get_arg_val<uint32_t>(4);
    constexpr auto xa = TensorAccessorArgs<0>();
    const auto xs = TensorAccessor(xa, x_addr, X_TILE);
    for (uint32_t t = 0; t < NB * X_BLK; ++t) {
        noc_async_read_page(x_first + t, xs, buf + t * X_TILE);
    }
    noc_async_read_barrier();
    const uint64_t h_ring = kc_xy(hxy, ring);
    const uint64_t h_xarr = kc_xy(hxy, get_semaphore(XARR));
    volatile tt_l1_ptr uint32_t* freed = kc_sem(RFREE);
    constexpr uint32_t total = TOTAL_BLOCKS * XP;
    uint32_t iss = 0, fwd = 0;
    while (fwd < total) {
        invalidate_l1_cache();
        while (iss < total && iss - fwd < DEPTH && (iss / XP < XS || *freed >= iss / XP + 1 - XS)) {
            const uint32_t blk = iss / XP, pc = iss % XP;
            const uint32_t src = buf + (blk % NB) * X_BYTES + pc * PIECE;
            const uint32_t off = (blk % XS) * X_BYTES + pc * PIECE;
            for (uint32_t o = 0; o < PIECE; o += NOC_MAX_BURST_SIZE) {
                const uint32_t n = PIECE - o < NOC_MAX_BURST_SIZE ? PIECE - o : NOC_MAX_BURST_SIZE;
                noc_async_write_one_packet_with_trid(src + o, h_ring + off + o, n, 1 + iss % DEPTH);
            }
            ++iss;
        }
        while (fwd < iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        while (fwd < iss && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + fwd % DEPTH)) {
            noc_semaphore_inc(h_xarr, 1);
            ++fwd;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_write_set_trid(0);
}
