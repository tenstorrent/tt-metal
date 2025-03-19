// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "risc_attribs.h"
#include "debug/dprint.h"
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
    void MAIN {
#endif
        volatile uint32_t tt_l1_ptr* results = (volatile uint32_t tt_l1_ptr*)RESULTS_ADDR;
        int i;
        for (i = 0; i < NUM_RUNTIME_ARGS; i++) {
#ifdef COMMON_RUNTIME_ARGS
            constexpr uint32_t kCommonRTASeparation = 1024;
            results[i + kCommonRTASeparation] = get_common_arg_val<uint32_t>(i);
#endif
            results[i] = get_arg_val<uint32_t>(i);
        }

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
    }
