// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel phase framework (data-movement and compute RISCs).
//
// Several phases run back to back inside ONE kernel per RISC and share circular-buffer indices (a core has 64): a CB
// index is one (format, page size) slot sized for the largest phase that uses it. Every phase assumes its CBs start
// at their base address (contiguous reservations from get_write_ptr / get_read_ptr), so at a phase boundary each RISC
// rewinds the CBs it used. Precondition: the CB is EMPTY (every pushed page popped) — the shared tiles-received /
// tiles-acked counters keep counting, only this RISC's local pointers return to the base.
#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"

namespace mk {
// TRISC: the unpacker thread owns the read side, the packer thread the write side (addresses in 16-B words here)
FORCE_INLINE void cb_rewind(uint32_t cb) {
    UNPACK(({
        LocalCBInterface& i = get_local_cb_interface(cb);
        i.fifo_rd_ptr = i.fifo_limit - i.fifo_size;
    }));
    PACK(({
        LocalCBInterface& i = get_local_cb_interface(cb);
        i.fifo_wr_ptr = i.fifo_limit - i.fifo_size;
        i.fifo_wr_tile_ptr = 0;
    }));
}
// CBs [first, first + n): a phase that runs again later in the same program (the layer megakernel's FFN mhc_pre /
// mhc_post reuse the attention instances' CB indices)
FORCE_INLINE void cb_rewind_range(uint32_t first, uint32_t n) {
    for (uint32_t cb = first; cb < first + n; ++cb) {
        cb_rewind(cb);
    }
}
}  // namespace mk
#else
#include "api/dataflow/dataflow_api.h"

namespace mk {
// BRISC / NCRISC: both pointers (a data-movement RISC is the producer or the consumer of a CB, or both)
FORCE_INLINE void cb_rewind(uint32_t cb) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    const uint32_t base = i.fifo_limit - i.fifo_size;
    i.fifo_rd_ptr = base;
    i.fifo_wr_ptr = base;
}
FORCE_INLINE void cb_rewind_range(uint32_t first, uint32_t n) {
    for (uint32_t cb = first; cb < first + n; ++cb) {
        cb_rewind(cb);
    }
}
// this core's own semaphore `sem_id` += 1 over the NoC (cross-RISC hand-off inside a core: the caller's writes have
// landed — write barrier done — before the flag is raised)
FORCE_INLINE void signal_self(uint32_t sem_id) {
    noc_semaphore_inc(get_noc_addr(get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
}

// phase hand-off: wait until `count` producers incremented semaphore `sem_id`, then reset it for the next hand-off
FORCE_INLINE void wait_handoff(uint32_t sem_id, uint32_t count) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(s, count);
    noc_semaphore_set(s, 0);
}

// phase hand-off: after this core's output landed (write barrier done by the caller), increment semaphore `sem_id`
// on each of `n` consumers whose NoC (x, y) are the runtime args at [arg0, arg0 + 2n)
FORCE_INLINE void signal_handoff(uint32_t sem_id, uint32_t arg0, uint32_t n) {
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t x = get_arg_val<uint32_t>(arg0 + 2 * r), y = get_arg_val<uint32_t>(arg0 + 2 * r + 1);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1);
    }
    noc_async_atomic_barrier();
}
}  // namespace mk
#endif
