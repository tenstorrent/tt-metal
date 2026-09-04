// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "internal/risc_attribs.h"
#include "api/debug/dprint.h"
// TODO FIXME: this build system is ridiculously stupid
#ifdef COMPILE_FOR_TRISC
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "defines_generated.h"
#endif

// Callers that use common runtime args may specify a separate common-arg count; otherwise it matches the unique
// count, preserving behavior for existing callers.
#ifdef COMMON_RUNTIME_ARGS
#ifndef NUM_COMMON_RUNTIME_ARGS
#define NUM_COMMON_RUNTIME_ARGS NUM_RUNTIME_ARGS
#endif
#endif

void kernel_main() {
    volatile uint32_t tt_l1_ptr* results = (volatile uint32_t tt_l1_ptr*)RESULTS_ADDR;
    constexpr uint32_t kCommonRTASeparation = 1024;

    // Number of unique runtime args set on this core. When the same kernel binary is placed across two core ranges
    // with different arg counts, the host defines CR1_START_Y and NUM_RUNTIME_ARGS_CR1 so each core reads exactly
    // the count it was given (cores at absolute logical y >= CR1_START_Y belong to the second range). Reading only
    // the args that were actually set keeps the watcher's runtime-arg bounds check satisfied.
    uint32_t num_unique_runtime_args = NUM_RUNTIME_ARGS;
#if defined(NUM_RUNTIME_ARGS_CR1) && defined(CR1_START_Y)
    if (get_absolute_logical_y() >= CR1_START_Y) {
        num_unique_runtime_args = NUM_RUNTIME_ARGS_CR1;
    }
#endif

    for (uint32_t i = 0; i < num_unique_runtime_args; i++) {
        results[i] = get_arg_val<uint32_t>(i);
    }
#ifdef COMMON_RUNTIME_ARGS
    for (uint32_t i = 0; i < NUM_COMMON_RUNTIME_ARGS; i++) {
        results[i + kCommonRTASeparation] = get_common_arg_val<uint32_t>(i);
    }
#endif

#ifdef COORDS_ADDR
#ifdef DATA_MOVEMENT
    volatile uint32_t tt_l1_ptr* coords = (volatile uint32_t tt_l1_ptr*)COORDS_ADDR;
    coords[0] = my_x[noc_index];
    coords[1] = my_y[noc_index];
    coords[2] = get_absolute_logical_x();
    coords[3] = get_absolute_logical_y();
    coords[4] = get_relative_logical_x();
    coords[5] = get_relative_logical_y();
#endif
#endif
}
