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

  void ApplicationHandler(void);

#ifdef __cplusplus
}
#endif

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
    uint16_t core_flat_id __attribute__((used));
}
#endif

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t atomic_ret_val __attribute__ ((section ("l1_data"))) __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

void __attribute__((section("erisc_l1_code.1"), noinline)) Application(void) {
    DEBUG_STATUS("I");
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    // Not using firmware_kernel_common_init since it is copying to registers
    // TODO: need to find free space that routing FW is not using
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    risc_init();
    noc_init();
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    DEBUG_STATUS("REW");
    uint32_t count = 0;
    while (routing_info->routing_enabled != 1) {
        volatile uint32_t *ptr = (volatile uint32_t *)0xffb2010c;
        count++;
        *ptr = 0xAABB0000 | (count & 0xFFFF);
        internal_::risc_context_switch();
    }
    DEBUG_STATUS("RED");


    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (mailboxes->launch.go.run == RUN_MSG_GO) {
            DeviceZoneScopedMainN("ERISC-FW");
            DeviceZoneSetCounter(mailboxes->launch.kernel_config.host_assigned_id);

            firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, DISPATCH_CLASS_ETH_DM0);

            DEBUG_STATUS("R");
            kernel_init();
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}
void __attribute__((section("erisc_l1_code.0"), naked)) ApplicationHandler(void) {
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
