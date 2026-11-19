// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The compute side of the in-program collective (fused/collective/collective.py): what a compute kernel calls to
// read a landing. The landing of a site is two circular buffers, one per parity, over the site's persistent L1
// (collective.hpp). The data-movement kernel pushes `members * tiles` pages into the one of this round, slot after
// slot in ring order, once wait_landed() returned.
//
//     const uint32_t cb = hy4_collective::wait_landing(cb_parity0, cb_parity1, members * tiles);
//     hy4_collective::sum_slots(cb, members, tiles, cb_out);   // or unpack tile `slot * tiles + t` of `cb`
//     cb_pop_front(cb, members * tiles);
//
// Both landing buffers of a site must have the same data format, and for Float32 payloads both need
// UnpackToDestMode::UnpackToDestFp32 with fp32_dest_acc_en, or the unpacker's source registers drop mantissa bits.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace hy4_collective {

// Which of a site's two landing buffers holds this round: the one that has `pages` pages at its front. Only the
// unpack thread reads landing buffers, so only it waits; the math and pack threads get `cb_parity0`, which has the
// same format.
ALWI uint32_t wait_landing(uint32_t cb_parity0, uint32_t cb_parity1, uint32_t pages) {
    uint32_t cb = cb_parity0;
#ifdef TRISC_UNPACK
    volatile tt_l1_ptr uint32_t* received0 = get_cb_tiles_received_ptr(cb_parity0);
    volatile tt_l1_ptr uint32_t* received1 = get_cb_tiles_received_ptr(cb_parity1);
    const uint16_t wanted = static_cast<uint16_t>(pages);
    for (;;) {
        uint16_t available = static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(received0))) -
                             get_local_cb_interface(cb_parity0).tiles_acked;
        if (available >= wanted) {
            cb = cb_parity0;
            break;
        }
        available = static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(received1))) -
                    get_local_cb_interface(cb_parity1).tiles_acked;
        if (available >= wanted) {
            cb = cb_parity1;
            break;
        }
    }
#else
    (void)cb_parity1;
    (void)pages;
#endif
    return cb;
}

// out[t] = ((slot 0 [t] + slot 1 [t]) + slot 2 [t]) + ...: the slots of landing buffer `cb` summed in ring order
// through the SFPU, so every member gets the same bits. On the board a Float32 sum equals the host's Float32 adds
// in that order bit for bit; a BFloat16 sum is that Float32 sum packed once, a tie rounded away from zero.
// `cb` must hold members * tiles pages (wait_landing); reserves and pushes `tiles` pages of cb_out. The caller
// pops `cb`.
ALWI void sum_slots(uint32_t cb, uint32_t members, uint32_t tiles, uint32_t cb_out) {
    cb_reserve_back(cb_out, tiles);
    copy_init(cb);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb, t, 0);
        add_binary_tile_init();
        for (uint32_t m = 1; m < members; ++m) {
            copy_tile(cb, m * tiles + t, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, tiles);
}

}  // namespace hy4_collective
