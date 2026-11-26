// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2: batched staging between a lane's NCRISC walker (spa_ur.cpp / spa_ca.cpp) and its BRISC
// sender (spa_us.cpp). The walker issues the 4 page reads of up to NB pools before ONE read barrier and pushes them as
// one batch of NB pages (the gather was latency-bound at one barrier per pool: 0.53 us per 4-KiB pool); a short batch
// is padded with DUMMY entries, so every push is NB pages and every batch starts at a multiple of NB in the rings (NB
// divides the CB depth: CB push sizes tile the capacity). The sender pops a batch after one writes-flush. The end
// marker (slot SKIP) is pushed alone, after a closed batch.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace spa {

constexpr uint32_t NB = 8;       // pools per staging batch (the CB depth: 2 NB)
constexpr uint32_t RB = 1024;    // one latent row: 512 bf16
constexpr uint32_t PB = 4 * RB;  // one pool: 4 rows
constexpr uint32_t MB = 16;      // one meta page: slot, destination
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t DUMMY = 0xFFFFFFFEu;

template <uint32_t CB_STAGE, uint32_t CB_META>
struct Stager {
    uint32_t n = 0, sp = 0, mp = 0;

    template <typename Acc>
    void add(const Acc& src, uint32_t row0, uint32_t slot, uint32_t d) {
        if (n == 0) {
            cb_reserve_back(CB_STAGE, NB);
            cb_reserve_back(CB_META, NB);
            sp = get_write_ptr(CB_STAGE);
            mp = get_write_ptr(CB_META);
        }
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read_page(row0 + j, src, sp + n * PB + j * RB);
        }
        volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp + n * MB);
        mt[0] = slot;
        mt[1] = d;
        if (++n == NB) {
            close();
        }
    }

    void close() {
        if (n == 0) {
            return;
        }
        for (uint32_t k = n; k < NB; ++k) {
            volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp + k * MB);
            mt[0] = DUMMY;
            mt[1] = 0;
        }
        noc_async_read_barrier();
        cb_push_back(CB_META, NB);
        cb_push_back(CB_STAGE, NB);
        n = 0;
    }

    void end() {  // closes the open batch, then the end marker alone
        close();
        cb_reserve_back(CB_META, 1);
        volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META));
        mt[0] = SKIP;
        mt[1] = 0;
        cb_push_back(CB_META, 1);
    }
};

}  // namespace spa
