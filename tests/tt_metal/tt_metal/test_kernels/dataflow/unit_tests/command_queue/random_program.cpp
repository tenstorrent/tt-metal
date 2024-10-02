
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

#include "debug/dprint.h"

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

    // Go through all the CBs + Semaphores + RTArgs and confirm the data looks correct
    constexpr volatile uint32_t num_cbs = get_compile_time_arg_val(3);
    constexpr volatile uint32_t num_sems = get_compile_time_arg_val(4);
    constexpr volatile uint32_t num_unique_rt_args = get_compile_time_arg_val(5);
    constexpr volatile uint32_t num_common_rt_args = get_compile_time_arg_val(6);
    constexpr volatile uint32_t page_size = get_compile_time_arg_val(7);

    for (uint32_t i = 0; i < num_cbs; i++) {
        tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
        uint32_t kernel_config_base = mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
        uint32_t tt_l1_ptr *cb_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.cb_offset);
        uint32_t cb_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_base + i * 4)[3];
        uint32_t expected = ((i + 1) * page_size) >> 4;
        if (cb_val != expected) {
            DPRINT << "Problem with CB idx: " << i << " Expected: " << expected << " Got: " << cb_val << ENDL();
            while(true); // Purposefully hang the kernel if CBs did not arrive correctly
        }
    }

#ifdef DATA_MOVEMENT
    for (uint32_t i = 0; i < num_sems; i++) {
        uint32_t sem_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i))[0];
        uint32_t expected = i + 1;
        if (sem_val != expected) {
            DPRINT << "Problem with Sem idx: " << i << " Expected: " << expected << " Got: " << sem_val << ENDL();
            while(true); // Purposefully hang the kernel if semaphores did not arrive correctly
        }
    }
#endif

    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        uint32_t rt_arg = get_arg_val<uint32_t>(i);
        uint32_t expected = i;
        if (rt_arg != expected) {
            DPRINT << "Problem with unique RT Arg idx: " << i << " Expected: " << expected << " Got: " << rt_arg << ENDL();
            while(true); // Purposefully hang the kernel if Unique RT Args did not arrive correctly.
        }
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        uint32_t rt_arg = get_common_arg_val<uint32_t>(i);
        uint32_t expected = i+100;
        if (rt_arg != expected){
            DPRINT << "Problem with common RT Arg idx: " << i << " Expected: " << expected << " Got: " << rt_arg << ENDL();
            while(true); // Purposefully hang the kernel if Common RT Args did not arrive correctly.
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
