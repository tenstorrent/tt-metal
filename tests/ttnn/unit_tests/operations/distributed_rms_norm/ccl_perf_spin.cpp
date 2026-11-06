// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// eval/ccl_perf.py spin op: one core waits get_compile_time_arg_val(0) wall-clock cycles, then exits. No runtime
// args, no CBs: a long op that costs next to nothing to dispatch, so the calls after it start with their dispatch
// hidden and a small launch skew between chips.

#include "api/dataflow/dataflow_api.h"

void kernel_main() { riscv_wait(get_compile_time_arg_val(0)); }
