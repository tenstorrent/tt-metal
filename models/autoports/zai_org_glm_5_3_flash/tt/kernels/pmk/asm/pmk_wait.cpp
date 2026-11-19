// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly (program A): the FINAL-RENDEZVOUS wait part (Stage 14, coordinator 2026-10-24 00:41:21Z: the TP8 root
// cores' late-write class). A core that RECEIVES NoC writes / atomics from other cores of the program (the mhc roots:
// their members' GATHER partials + SEM_G1 / G2) must not finish its program while one of them is still in flight: each
// sender, after its own last write to this core and its noc_async_write_barrier / atomic barrier, +1 this core's
// program semaphore SEM (pmk_sig.cpp, sent LAST); this part waits SEM >= N, so the core's completion (the dispatcher's
// done count that releases the next program into this core's config ring / CB region) follows every sender's last
// write. Runtime args at ph_rt0: [active, sem_id, n]; active = 0 -> nothing (idle cores of a composite).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a = ph_rt0;
    (void)PH_CT0;
    if (get_arg_val<uint32_t>(a + 0) == 0) {
        return;
    }
    const uint32_t sem = get_arg_val<uint32_t>(a + 1);
    const uint32_t n = get_arg_val<uint32_t>(a + 2);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)), n);
}
