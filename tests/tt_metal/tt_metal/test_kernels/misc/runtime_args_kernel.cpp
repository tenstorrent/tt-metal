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
            results[i] = get_common_arg_val<uint32_t>(i);
#else
            results[i] = get_arg_val<uint32_t>(i);
#endif
        }

#ifdef COORDS_ADDR
        volatile uint32_t tt_l1_ptr* coords = (volatile uint32_t tt_l1_ptr*)COORDS_ADDR;
#ifdef DATA_MOVEMENT
        coords[0] = my_x[noc_index];
        coords[1] = my_y[noc_index];
#else
#ifdef COMPUTE
        coords[0] = 0;
        coords[1] = 0;
#else
        coords[0] = 0xdeadbeef;
        coords[1] = 0xdeadbeef;
#endif
#endif

        coords[2] = my_logical_x;
        coords[3] = my_logical_y;
        coords[4] = my_sub_device_x;
        coords[5] = my_sub_device_y;
#endif
    }
    }
