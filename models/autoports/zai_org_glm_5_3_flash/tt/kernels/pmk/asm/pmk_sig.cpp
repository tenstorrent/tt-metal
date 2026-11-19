// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly (program A): the FAN-OUT SIGNAL part. After this RISC's writes landed (DRAM stores of a projection, the
// x-hat rows of the mHC pre, the RS lanes' y rows), +1 on program semaphore SEM of every listed core: the hand-off
// counters H1 (the holders' SEM_HN 4), H3 (the aggregator's SEM_QKV 5), H6 (the aggregator lane's FS_REDUCED 14).
// Runtime args at ph_rt0: [active, sem_id, n, x0, y0, x1, y1, ..]; active = 0 -> nothing (idle cores of a composite).
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
    noc_async_write_barrier();  // every write this RISC issued has landed before the counters move
    noc_async_read_barrier();
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t x = get_arg_val<uint32_t>(a + 3 + 2 * i), y = get_arg_val<uint32_t>(a + 4 + 2 * i);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    }
    noc_async_atomic_barrier();
}
