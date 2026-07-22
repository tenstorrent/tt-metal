// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEYSTONE test kernel: self-targeted ("talk to yourself") NoC atomic increment.
//
// PURPOSE
// -------
// Prove whether a data-movement (DM) core can issue a NoC atomic increment
// (NOC_AT_INS_INCR_GET) addressed at its OWN node and have it (a) not deadlock
// and (b) be mutually atomic with the same-node atomics issued by the other DM
// cores. This is the single unproven hardware behavior that the auto-path
// "EXTERNAL" semaphore mode depends on: on a semaphore that is touched
// externally (by the NoC / another node / chip), even a *local* increment must
// go out through the NoC atomic path, because RISC-V AMOs cannot be used on the
// uncached L1 alias (they hang, see dev_mem_map.h) and the cached AMO domain is
// not coherent with the NoC. Routing every writer through the NoC atomic makes
// local + remote writers converge on one NIU atomicity point.
//
// get_noc_addr(addr) (single-arg, dataflow_api_addrgen.h) encodes THIS core's
// own NoC coordinates (my_x/my_y), so noc_semaphore_inc(get_noc_addr(sem_addr))
// is a loopback atomic RMW that leaves the core, traverses the NoC, and is
// applied back at this node's own TL1 (node L1) memory port.
//
// Every user DM thread increments the SAME 32-bit word `increment_times` times.
// The host verifies the final value == num_user_dms * increment_times:
//   - deadlock on the loopback path            -> kernel hangs (test times out)
//   - non-atomic loopback / no NIU serializing -> final count is SHORT (lost updates)
//
// Note: the word is only ever touched by NoC atomics (which land at TL1), never
// by a cached load/store, so no cache flush is needed here -- that is precisely
// the property the EXTERNAL path relies on.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

#ifdef REMOTE_TARGET
    // Remote target: atomically increment a word on ANOTHER node, whose NoC
    // coordinates are passed in. This is the genuinely-remote source that must be
    // mutually atomic with the self-targeted (loopback) increments on that node.
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);
    const uint64_t target_noc_addr = get_noc_addr(remote_noc_x, remote_noc_y, sem_addr);
#else
    // Self target (loopback): get_noc_addr(addr) encodes THIS core's own NoC
    // coordinates (my_x/my_y), so the atomic RMW is applied back at this node's TL1.
    const uint64_t target_noc_addr = get_noc_addr(sem_addr);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        noc_semaphore_inc(target_noc_addr, 1);
        // Drain after each atomic: bounds the number of outstanding atomics (so we
        // never exceed the NIU's in-flight tracking) and makes completion
        // unambiguous. Cross-core interleaving is unaffected -- that concurrency
        // between independent DM cores/nodes is exactly what this test stresses.
        noc_async_atomic_barrier();
    }
}
