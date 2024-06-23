// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

// TODO FIXME: this build system is ridiculously stupid
#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "defines_generated.h"
#endif

#ifdef DATA_MOVEMENT
namespace {
void kernel_main() {
#endif
#ifdef COMPUTE
namespace NAMESPACE {
void MAIN  {
#endif
    volatile uint32_t tt_l1_ptr *results = (volatile uint32_t tt_l1_ptr *)RESULTS_ADDR;
    for (int i = 0; i < NUM_RUNTIME_ARGS; i++) {
#ifdef COMMON_RUNTIME_ARGS
        results[i] = get_common_arg_val<uint32_t>(i);
#else
        results[i] = get_arg_val<uint32_t>(i);
#endif
    }
}
}
