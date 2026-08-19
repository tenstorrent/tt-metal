// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reports which mechanism the host baked for each of the two bound semaphores. Built for the
// id-collision promotion test: semaphore ids are unique per core (not per node), so two
// semaphores on disjoint nodes share this kernel's one scope-table slot, the host must promote
// both to the same (EXTERNAL) mechanism, or the emitted asserts fail this kernel's build.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {
        volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
        report[0] = static_cast<uint32_t>(sem_scope_of(sem::near_sem));
        report[1] = static_cast<uint32_t>(sem_scope_of(sem::far_sem));
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        flush_l2_cache_line(report_addr);
#endif
    }
}
