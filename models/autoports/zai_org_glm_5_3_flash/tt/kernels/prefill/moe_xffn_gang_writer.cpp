// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// W8 unit 2 (GLM53_PF_MOE_XFFN_GANG, w8/combine_fusion_design.md) -- writer (NCRISC, NoC1) of the gang EP expert FFN:
// per unit the group's x rows as moe_xffn_writer.cpp (XFFN_X_ON_NOC1, XFFN_VALID_ROWS), then the in-gang y RING
// REDUCE-SCATTER over the 2 G N chunks in this core's ring order (moe_xffn_gang_reader.cpp), per tile row r:
//   h > 0     : receive -- wait RING_VALID >= blocks received + 1 (the predecessor wrote the block into this core's
//               CB_RIN slot), push it to the compute;
//   (compute) : CB_YRM = RING_IN + this slice's partial (h > 0) or the partial alone (h = 0);
//   h > 0     : credit the predecessor (its next block may reuse the slot the compute just freed);
//   h < G - 1 : send -- wait RING_CREDIT (2 MG slots ahead), write the block into the successor's CB_RIN slot
//               (sent mod 2 MG), barrier, raise the successor's RING_VALID;
//   h = G - 1 : the owner -- the full y block's valid rows -> out_rm rows d0 + 32 r + row, columns of N chunk nc
//   (once).
// CB_RIN has exactly one remote writer (the predecessor's NCRISC) and one local producer (this NCRISC, after VALID);
// the counters only grow (>= waits: memory credit-ahead / equality-semaphore lessons). One step = 2 chunks x mt <= 2 MG
// blocks fits the 2 MG slots, so no step waits on the step after it (design note sec 8: the ring cannot deadlock).
// Compile-time: H_T, NC, KC, MG. Runtime: as moe_xffn_writer.cpp + slice s (7), successor x / y (8, 9), predecessor
// x / y (10, 11) -- NoC virtual coordinates. Semaphores SEM_RING_VALID / SEM_RING_CREDIT and CB_RIN:
// moe_xffn_common.hpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_xffn_common.hpp"
#include "moe_xffn_probe.hpp"

using namespace xffn;
using namespace xffn_probe_slot;

#if !defined(XFFN_GANG) || !defined(XFFN_X_ON_NOC1)
#error "the gang writer needs XFFN_GANG and XFFN_X_ON_NOC1"
#endif

void kernel_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t KC = get_compile_time_arg_val(2);
    constexpr uint32_t MG = get_compile_time_arg_val(3);
    constexpr uint32_t G = XFFN_GANG;
    constexpr uint32_t ROW_BYTES = H_T * 32 * 2;
    constexpr uint32_t SEG = NC * 32 * 2;     // one y row's N chunk (bf16)
    constexpr uint32_t XSEG = KC * 32 * 2;    // one x row's K chunk (bf16)
    constexpr uint32_t BLK = NC * BF16_TILE;  // one ring block: NC bf16 tiles = 32 rows x NC tiles
    constexpr uint32_t SLOTS = 2 * MG;
    static_assert(H_T / NC == 2 * G, "gang chunking: H / NC = 2 G N chunks");

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t disp_addr = get_arg_val<uint32_t>(2);
    const uint32_t cap = get_arg_val<uint32_t>(3);
    const uint32_t s = get_arg_val<uint32_t>(7);
    const uint32_t succ_x = get_arg_val<uint32_t>(8);
    const uint32_t succ_y = get_arg_val<uint32_t>(9);
    const uint32_t pred_x = get_arg_val<uint32_t>(10);
    const uint32_t pred_y = get_arg_val<uint32_t>(11);
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> x_g{.bank_base_address = x_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = cap * 4};
    volatile tt_l1_ptr uint32_t* dl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WTAB));
    XP_BEGIN();

    // the ring: every core's CB_RIN starts at the same L1 address; the semaphores too
    const uint32_t rin_base = get_write_ptr(CB_RIN);
    const uint32_t sem_valid = get_semaphore(SEM_RING_VALID);
    const uint32_t sem_credit = get_semaphore(SEM_RING_CREDIT);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_valid);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_credit);
    const uint64_t succ_rin = get_noc_addr(succ_x, succ_y, rin_base);
    const uint64_t succ_valid = get_noc_addr(succ_x, succ_y, sem_valid);
    const uint64_t pred_credit = get_noc_addr(pred_x, pred_y, sem_credit);
    uint32_t sent = 0;
    uint32_t recvd = 0;

    while (true) {
        XP_TIC();
        cb_wait_front(CB_WCTL, 1);
        XP_TOC(W_CTL_WAIT);
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_WCTL));
        const uint32_t mt = c[0];
        const uint32_t rows = c[1];
        const uint32_t d0 = c[2];
        cb_pop_front(CB_WCTL, 1);
        if (mt == 0) {
            break;
        }
        // the group's x rows (only the valid ones), one K chunk per push
        XP_TIC();
        noc_async_read(disp_g.get_noc_addr(0, d0 * 4), reinterpret_cast<uint32_t>(dl), mt * 32 * 4);
        noc_async_read_barrier();
        XP_TOC(W_DISP_READ);
        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
            XP_TIC();
            cb_reserve_back(CB_XRM, MG * KC);
            XP_TOC(W_X_WAIT);
            XP_TIC();
            const uint32_t xp = get_write_ptr(CB_XRM);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(x_g.get_noc_addr(dl[r], kc * XSEG), xp + r * XSEG, XSEG);
            }
            noc_async_read_barrier();
            XP_TOC(W_X_READ);
            XP_ADD(W_X_KB, rows * XSEG / 1024);
            cb_push_back(CB_XRM, MG * KC);
        }
        // the y ring
        for (uint32_t q = 0; q < 2 * G; ++q) {
            const uint32_t h = q >> 1;
            const uint32_t nc = ((s + G - 1 - h) % G) + G * (q & 1);
            for (uint32_t r = 0; r < mt; ++r) {
                if (h > 0) {
                    cb_reserve_back(CB_RIN, NC);  // free by the credit protocol (never waits in a correct run)
                    XP_TIC();
                    noc_semaphore_wait_min(valid, ++recvd);
                    XP_TOC(W_Y_WAIT);
                    cb_push_back(CB_RIN, NC);
                }
                cb_wait_front(CB_YRM, NC);
                if (h > 0) {
                    noc_semaphore_inc(pred_credit, 1);
                }
                const uint32_t rp = get_read_ptr(CB_YRM);
                XP_TIC();
                if (h < G - 1) {
                    if (sent >= SLOTS) {
                        noc_semaphore_wait_min(credit, sent - SLOTS + 1);
                    }
                    noc_async_write(rp, succ_rin + (sent % SLOTS) * BLK, BLK);
                    noc_async_write_barrier();
                    noc_semaphore_inc(succ_valid, 1);
                    ++sent;
                } else {
                    for (uint32_t t = 0; t < 32; ++t) {
                        const uint32_t row = r * 32 + t;
                        if (row < rows) {
                            noc_async_write(rp + t * SEG, out_g.get_noc_addr(d0 + row, nc * SEG), SEG);
                        }
                    }
                    noc_async_write_barrier();
                    XP_ADD(W_Y_KB, (rows - r * 32 < 32 ? rows - r * 32 : 32) * SEG / 1024);
                }
                XP_TOC(W_Y_WRITE);
                cb_pop_front(CB_YRM, NC);
            }
        }
    }
    // drain: the successor has consumed (credited) every block this core sent, so after this no remote write or
    // semaphore increment targets this core's CB_RIN / ring semaphores (the predecessor's VALIDs all precede the last
    // receive): a composite's later phase may reuse them (W7 step 5). The standalone program ends no later.
    if (sent > 0) {
        noc_semaphore_wait_min(credit, sent);
    }
    noc_async_atomic_barrier();
#ifdef XFFN_PROBE
    XP_FLUSH(
        (reinterpret_cast<uint32_t>(dl) + 63) & ~63u,
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5) + get_arg_val<uint32_t>(6));
#endif
}
