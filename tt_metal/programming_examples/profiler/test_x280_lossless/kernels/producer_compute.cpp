// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Step-0 producer, compute variant (the 3 TRISCs). Modeled on the profiler's full_buffer_compute.cpp:
// a compute kernel here uses plain kernel_main() and runs on all three TRISCs, each with its own
// build-provided PROCESSOR_INDEX (TRISC0/1/2 -> 2/3/4), so each writes to its own ring.
#include "api/compute/compute_kernel_api.h"

#ifndef PROCESSOR_INDEX
#error "producer_compute.cpp needs the build-provided PROCESSOR_INDEX (per-TRISC risc id)"
#endif
#define PROC_IDX PROCESSOR_INDEX
#include "producer_common.h"

void kernel_main() { producer_run(); }
