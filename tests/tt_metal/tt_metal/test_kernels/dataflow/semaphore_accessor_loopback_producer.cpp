// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal test kernel: increments a semaphore via its sem:: accessor name.
// The sole purpose is to exercise the sem-accessor-name → ID resolution machinery

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    experimental::Semaphore s(sem::signal);
    s.up(1);
}
