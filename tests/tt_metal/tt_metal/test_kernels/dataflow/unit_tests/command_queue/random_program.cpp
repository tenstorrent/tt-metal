
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#ifdef DATA_MOVEMENT
#include <stdint.h>
#include "dataflow_api.h"
#endif

#ifdef COMPUTE
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#endif

#ifdef DATA_MOVEMENT
namespace {
void kernel_main() {
#endif
#ifdef COMPUTE
namespace NAMESPACE {
void MAIN  {
#endif
    constexpr volatile uint32_t outer_loop = get_compile_time_arg_val(0);
    constexpr volatile uint32_t middle_loop = get_compile_time_arg_val(1);
    constexpr volatile uint32_t inner_loop = get_compile_time_arg_val(2);

    // Go through all the CBs + Semaphores and confirm the data looks correct
    constexpr volatile uint32_t num_cbs = get_compile_time_arg_val(3);
    constexpr volatile uint32_t num_sems = get_compile_time_arg_val(4);
    constexpr volatile uint32_t page_size = get_compile_time_arg_val(5);

    for (uint32_t i = 0; i < num_cbs; i++) {
        if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CIRCULAR_BUFFER_CONFIG_BASE + i * 16)[3] != ((i + 1) * page_size) >> 4) {
            while(true); // Purposefully hang the kernel if CBs did not arrive correctly
        }
    }

    for (uint32_t i = 0; i < num_sems; i++) {
        if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SEMAPHORE_BASE + i * 16)[0] != i + 1) {
            while(true); // Purposefully hang the kernel if semaphores did not arrive correctly
        }
    }

    #pragma unroll(get_compile_time_arg_val(0))
    for (volatile uint32_t i = 0; i < outer_loop; i++) {
        #pragma unroll(get_compile_time_arg_val(1))
        for (volatile uint32_t j = 0; j < middle_loop; j++) {
            #pragma unroll(get_compile_time_arg_val(2))
            for (volatile uint32_t k = 0; k < inner_loop; k++) {
                // Do nothing
            }
        }
    }
}
}
