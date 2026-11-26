// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC "pre" -- the root's round 2 (sync.cpp, MHC_ROOT; a statement fragment, included where the root runs it):
// the 8 group S2 partials -> 1/rms(h) -> (MHC_DEFER_INV) the invh tile + every reader's MK_INVH_SEM -> the 1/rms(h)
// mailbox -> the 8 leaders. Knob off: after the post / comb / coefficient tiles; MHC_RMS_EARLY: right after the
// coefficient tile (sync.cpp states the ordering invariant).
    // ---- round 2: group S2 partials -> 1/rms(h) -> mailbox -> leaders
    cb_reserve_back(CB_GATHER, N_GROUPS);
    noc_semaphore_wait(sem_ptr(SEM_G2), N_GROUPS);
    noc_semaphore_set(sem_ptr(SEM_G2), 0);  // reset for a later instance in the same program
    cb_push_back(CB_GATHER, N_GROUPS);
    cb_wait_front(CB_INVR, 1);
    MK_MARK();  // 3: round-2 (1/rms(h)) ready
    invalidate_l1_cache();
#if defined(MHC_DEFER_INV)
    // the readers scale their hn-matmul outputs by it: the tile (row t, column 0 = 1/rms(h)(t)) -> invh, then +1 on
    // every reader's MK_INVH_SEM
    noc_async_write_page(0, invh_out, get_read_ptr(CB_INVR));
    noc_async_write_barrier();
    for (uint32_t r = 0; r < n_invh; ++r) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS + 2 + 2 * r),
                get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS + 3 + 2 * r),
                get_semaphore(MK_INVH_SEM)),
            1);
    }
    noc_async_atomic_barrier();
#endif
    {
        volatile tt_l1_ptr uint32_t* inv = words(get_read_ptr(CB_INVR));
#if defined(DEBUG_TAPS)
        tap(dbg, TAP_INVR, get_read_ptr(CB_INVR));
#endif
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        for (uint32_t t = 0; t < TILE; ++t) {
            msg[t] = t < T ? inv[tile_off(t, 0)] : 0u;
        }
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_async_write(
                msg_l1,
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    msg_l1),
                TILE * 4);
        }
        noc_async_write_barrier();
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    get_semaphore(SEM_M2)),
                1);
        }
        noc_async_atomic_barrier();
    }
    cb_pop_front(CB_INVR, 1);
