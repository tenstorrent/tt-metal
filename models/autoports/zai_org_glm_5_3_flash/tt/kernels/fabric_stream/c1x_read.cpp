// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 / C3 on the 2x4 mesh (two axes; tt/fabric_stream.FabricStreamC1X), lane NCRISC. The lane's
// list holds its OWN local blocks (ready when the producers' slices landed: LREADY + k, as c1_read.cpp) and then its
// PARTNER's blocks (they land in this chip's holder XS over axis 0 from the board partner; ready when the holder's
// arrive semaphore reached SLICES this launch, polled over the NoC at the address in SEMADDR). Each ready block is
// pulled from its holder's XS into the staging CB (NB-tile batches) for c1x_send.cpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SLICES = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB = 2048;
static_assert(TPB % NB == 0, "a batch never straddles a block");
#ifndef C1X_USE
#define C1X_USE 1  // this phase's C1 use in the launch (W7 P11 = 2): a holder's FS_ARRIVE reaches C1X_USE x SLICES
#endif

void kernel_main() {
    size_t a = ph_rt0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t n_own = get_arg_val<uint32_t>(a++);
    const uint32_t n_par = get_arg_val<uint32_t>(a++);
    const uint32_t blk_xy = a;  // (n_own + n_par) x (holder NoC 0 x, y): own blocks first, then the partner's
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    volatile tt_l1_ptr uint32_t* probe = fs::word(sync, fs::GEN_C);  // NoC-read landing word for the polls
    for (uint32_t k = 0; k < n_own + n_par; k++) {
        const uint32_t hx = get_arg_val<uint32_t>(blk_xy + 2 * k), hy = get_arg_val<uint32_t>(blk_xy + 2 * k + 1);
        if (k < n_own) {
            fs::wait_ge(fs::word(sync, fs::LREADY + k), g * SLICES);
        } else {
            const uint64_t sem = get_noc_addr(hx, hy, get_semaphore(fs::SEM_ARRIVE_ID));  // THE RING-ADDRESS RULE
            do {
                noc_async_read(sem, reinterpret_cast<uint32_t>(probe), fs::WORD);
                noc_async_read_barrier();
                invalidate_l1_cache();
            } while (*probe < C1X_USE * SLICES);  // a program semaphore: re-initialized at launch (cumulative per use)
        }
        fs::wait_eq(fs::word(sync, fs::SEMADDR), get_semaphore(fs::SEM_ARRIVE_ID));  // XSADDR is this launch's
        const uint64_t src = get_noc_addr(hx, hy, *fs::word(sync, fs::XSADDR));
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_reserve_back(CB_STAGE, NB);
            noc_async_read(src + b * NB * TB, get_write_ptr(CB_STAGE), NB * TB);
            noc_async_read_barrier();
            cb_push_back(CB_STAGE, NB);
        }
    }
#if !defined(FS_NO_ROW_BARRIER)
    {  // THE PER-HOP DRAIN (fs_common.hpp): the holders of this lane's row-peer blocks (their own + their partner's,
        // multi-hop line multicasts) all reached SLICES this launch -> LANDED = g for this lane's BRISC row barrier
        const uint32_t nl_at = blk_xy + 2 * (n_own + n_par);
        const uint32_t n_land = get_arg_val<uint32_t>(nl_at);
        const uint32_t own_sem = get_semaphore(fs::SEM_ARRIVE_ID);
        for (uint32_t i = 0; i < n_land; i++) {
            const uint32_t x = get_arg_val<uint32_t>(nl_at + 1 + 2 * i), y = get_arg_val<uint32_t>(nl_at + 2 + 2 * i);
            const uint64_t sem = get_noc_addr(x, y, own_sem);
            do {
                noc_async_read(sem, reinterpret_cast<uint32_t>(probe), fs::WORD);
                noc_async_read_barrier();
                invalidate_l1_cache();
            } while (*probe < C1X_USE * SLICES);
        }
        *fs::word(sync, fs::LANDED) = g;
    }
#endif
    *gen = g;
}
