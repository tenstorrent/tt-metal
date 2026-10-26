// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert processing order of the fused MoE program with expert prediction (MOE_PRED, dflash/moe_fused.py): the
// router of the previous round left P predicted expert ids for this layer in its L1; the gate|up cores prefetch them
// before the union is known, and every core processes the union slots whose expert was predicted first (in
// prediction order), then the other slots ascending. All kernels of the program compute the same order.

#pragma once

#include <stdint.h>

namespace moe_order {
// pred: the P predicted expert ids; uni: [U, id_0 .. id_{U-1}]; order[i] = the union slot processed i-th;
// kept_pred[j] = the prediction index of the j-th kept slot. Returns the number of kept (predicted and routed) slots.
template <uint32_t P>
inline uint32_t make_order(
    const uint32_t* pred, volatile tt_l1_ptr uint16_t* uni, uint32_t* order, uint32_t* kept_pred) {
    const uint32_t n = uni[0];
    uint32_t used = 0;  // bit s: slot s already placed
    uint32_t kept = 0;
    for (uint32_t k = 0; k < P; ++k) {
        for (uint32_t s = 0; s < n; ++s) {
            if (uni[1 + s] == pred[k] && !((used >> s) & 1)) {
                order[kept] = s;
                kept_pred[kept] = k;
                used |= 1u << s;
                ++kept;
                break;
            }
        }
    }
    uint32_t i = kept;
    for (uint32_t s = 0; s < n; ++s) {
        if (!((used >> s) & 1)) {
            order[i++] = s;
        }
    }
    return kept;
}

// Read the P (<= 8) predicted ids (32 B at pred_addr in the hub core's L1) through scratch_l1 and tell the hub
// (program semaphore sem_id) that they were read (the router overwrites them for the next round once every reader
// did).
template <uint32_t P>
inline void read_pred(
    uint32_t hub_x, uint32_t hub_y, uint32_t pred_addr, uint32_t scratch_l1, uint32_t sem_id, uint32_t* pred) {
    static_assert(P <= 8, "the prediction holds 8 ids per layer");
    noc_async_read(get_noc_addr(hub_x, hub_y, pred_addr), scratch_l1, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t k = 0; k < P; ++k) {
        pred[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1)[k];
    }
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(sem_id)), 1);
}
}  // namespace moe_order
