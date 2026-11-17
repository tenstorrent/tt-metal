// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: sets the bound GlobalSemaphore to the value of the named RTA "value".

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    experimental::GlobalSemaphore signal(global_sem::signal);
    signal.set(get_arg(args::value));
}
