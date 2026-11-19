// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly (program A): the AGGREGATOR part of a many-to-many hand-off. The producers each +1 THIS core's program
// semaphore SEM (pmk_sig.cpp); once it reaches WAIT, the core SETS the consumers' copy of SEM to VAL: one multicast SET
// over the members' exact NoC rectangle (lx, ly) .. (hx, hy) (NDEST cores) plus unicast SETs on the EXTRA cores (the
// roots). A consumer's noc_semaphore_wait_min(SEM, VAL) passes at once (contract §4b: wait_min, never equality). The
// semaphore address of one id is the same on every core (get_semaphore(id)), as the kda / fabric_stream kernels rely
// on. Runtime args at ph_rt0: [active, sem_id, wait, val, lx, ly, hx, hy, ndest, noc1, n_extra, (x, y) x n_extra].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a = ph_rt0;
    (void)PH_CT0;
    if (get_arg_val<uint32_t>(a + 0) == 0) {
        return;
    }
    const uint32_t sem = get_arg_val<uint32_t>(a + 1);
    const uint32_t wait = get_arg_val<uint32_t>(a + 2);
    const uint32_t val = get_arg_val<uint32_t>(a + 3);
    const uint32_t lx = get_arg_val<uint32_t>(a + 4), ly = get_arg_val<uint32_t>(a + 5);
    const uint32_t hx = get_arg_val<uint32_t>(a + 6), hy = get_arg_val<uint32_t>(a + 7);
    const uint32_t ndest = get_arg_val<uint32_t>(a + 8);
    const uint32_t noc1 = get_arg_val<uint32_t>(a + 9);
    const uint32_t n_extra = get_arg_val<uint32_t>(a + 10);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
    noc_semaphore_wait_min(s, wait);
    noc_semaphore_set(s, val);  // the local word is the multicast source
    if (ndest) {
        // NoC 0 multicasts go low -> high corner, NoC 1 high -> low (ttnn's 2D mcast matmul convention; pfs_stream.cpp)
        const uint64_t dst = noc1 ? get_noc_multicast_addr(hx, hy, lx, ly, get_semaphore(sem))
                                  : get_noc_multicast_addr(lx, ly, hx, hy, get_semaphore(sem));
        noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(s), dst, ndest);
    }
    for (uint32_t i = 0; i < n_extra; i++) {
        const uint32_t x = get_arg_val<uint32_t>(a + 11 + 2 * i), y = get_arg_val<uint32_t>(a + 12 + 2 * i);
        noc_inline_dw_write(get_noc_addr(x, y, get_semaphore(sem)), val);
    }
    noc_async_write_barrier();
}
