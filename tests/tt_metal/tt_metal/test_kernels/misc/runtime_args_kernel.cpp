// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

#ifdef DATA_MOVEMENT
namespace {
void kernel_main() {
#endif
#ifdef COMPUTE
    namespace NAMESPACE {
    void MAIN {
#endif
#if defined(SKIP_CORE_LOG_X) && defined(SKIP_CORE_LOG_Y)
        // Skip execution on a specific core (by logical coords) to test partial RTA coverage
        // This core doesn't have RTAs set, so accessing them would trigger watcher assert
        if (get_relative_logical_x() == SKIP_CORE_LOG_X && get_relative_logical_y() == SKIP_CORE_LOG_Y) {
            return;
        }
#endif

        volatile uint32_t tt_l1_ptr* results = (volatile uint32_t tt_l1_ptr*)RESULTS_ADDR;
        int i;
        // Read unique runtime args
        for (i = 0; i < NUM_RUNTIME_ARGS; i++) {
            results[i] = get_arg_val<uint32_t>(i);
        }
#ifdef COMMON_RUNTIME_ARGS
        // Read common runtime args
        constexpr uint32_t kCommonRTASeparation = 1024;
        for (i = 0; i < NUM_COMMON_RUNTIME_ARGS; i++) {
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
    }
