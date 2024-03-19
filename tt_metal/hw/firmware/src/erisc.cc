// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"

// Number of registers to save for early exit
#define CONTEXT_SIZE (13 * 4)

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t device_function_sums[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
uint64_t device_function_starts[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
}

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

void __attribute__((section("erisc_l1_code"))) Application(void) {
    DEBUG_STATUS('I');
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    // Not using firmware_kernel_common_init since it is copying to registers
    // TODO: need to find free space that routing FW is not using
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    DEBUG_STATUS('R', 'E', 'W');
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }
    DEBUG_STATUS('R', 'E', 'D');


    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (mailboxes->launch.run == RUN_MSG_GO) {
            DEBUG_STATUS('R');
            kernel_profiler::init_profiler();
            kernel_profiler::mark_time(CC_MAIN_START);
            kernel_init();
            kernel_profiler::store_function_sums();
            kernel_profiler::mark_time(CC_MAIN_END);
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}
void __attribute__((section("erisc_l1_code"), naked)) ApplicationHandler(void) {
    // Save the registers, stack pointer, return address so that we can early exit in the case of
    // an error.
    __asm__(
        "addi sp, sp, -%[context_size]\n\t"
        "sw x1, 0 * 4( sp )\n\t" // Return addr saved on stack
        "sw x8, 1 * 4( sp )\n\t"
        "sw x9, 2 * 4( sp )\n\t"
        "sw x18, 3 * 4( sp )\n\t"
        "sw x19, 4 * 4( sp )\n\t"
        "sw x20, 5 * 4( sp )\n\t"
        "sw x21, 6 * 4( sp )\n\t"
        "sw x22, 7 * 4( sp )\n\t"
        "sw x23, 8 * 4( sp )\n\t"
        "sw x24, 9 * 4( sp )\n\t"
        "sw x25, 10 * 4( sp )\n\t"
        "sw x26, 11 * 4( sp )\n\t"
        "sw x27, 12 * 4( sp )\n\t"
        "li x10, %[stack_save_addr]\n\t"
        "sw  sp, 0( x10 )\n\t"
        : /* No Inputs */
        : [context_size] "i" (CONTEXT_SIZE), [stack_save_addr] "i" (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE)
        : "x10", "memory"
    );
    Application();
    __asm__(
        "lw  x1, 0 * 4( sp )\n\t"
        "lw  x8, 1 * 4( sp )\n\t"
        "lw  x9, 2 * 4( sp )\n\t"
        "lw  x18, 3 * 4( sp )\n\t"
        "lw  x19, 4 * 4( sp )\n\t"
        "lw  x20, 5 * 4( sp )\n\t"
        "lw  x21, 6 * 4( sp )\n\t"
        "lw  x22, 7 * 4( sp )\n\t"
        "lw  x23, 8 * 4( sp )\n\t"
        "lw  x24, 9 * 4( sp )\n\t"
        "lw  x25, 10 * 4( sp )\n\t"
        "lw  x26, 11 * 4( sp )\n\t"
        "lw  x27, 12 * 4( sp )\n\t"
        "addi sp, sp, %[context_size]\n\t"
        "ret\n\t"
        : /* No Inputs */
        : [context_size] "i" (CONTEXT_SIZE)
        :
    );
}
