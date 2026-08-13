// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reports which mechanism the host baked for each of the TWO bound semaphores. Used by the
// id-collision promotion test: semaphore ids are unique per CORE, so two semaphores on disjoint
// nodes can share this kernel's one scope-table slot -- the host must have promoted both to the
// same (EXTERNAL) mechanism, or this kernel's build fails on the emitted tripwires.
// Layout: report[0] = scope(sem::near_sem), report[1] = scope(sem::far_sem).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {  // lowest user DM reports
        volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
        report[0] = static_cast<uint32_t>(sem_scope_of(sem::near_sem));
        report[1] = static_cast<uint32_t>(sem_scope_of(sem::far_sem));
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        flush_l2_cache_line(report_addr);  // make the report visible to the host readback of TL1
#endif
    }
}
