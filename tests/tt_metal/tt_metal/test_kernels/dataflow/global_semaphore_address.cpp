// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore bound as global_sem::sem. Runtime args {report_addr, value}.
// Reports address(), then writes value to the semaphore through that address with the legacy primitive, as a kernel
// that uses the address directly does.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t value = get_arg(args::value);

    GlobalSemaphore sem(global_sem::sem);
    const uint32_t sem_addr = sem.address();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), value);

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = sem_addr;
}
