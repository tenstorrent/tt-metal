// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "core_config.h"
#include "ckernel_globals.h"
#include "tensix_functions.h"
#include "risc_attribs.h"
#include "compile_time_args.h"
#include "dev_mem_map.h"
#include "hostdevcommon/kernel_structs.h"
#include <dev_msgs.h>
#include "noc/noc_parameters.h"
#include "debug/dprint.h"
#include "risc_common.h"
#if !defined(COMPILE_FOR_TRISC)
#include "dataflow_api.h"
#endif

extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
extern int32_t bank_to_l1_offset[NUM_L1_BANKS];

extern void kernel_init(uint32_t kernel_init);
extern void kernel_launch(uint32_t kernel_base_addr);
void l1_to_local_mem_copy(uint32_t* dst, uint32_t tt_l1_ptr* src, int32_t len);

inline void do_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear bss.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // Copy initialized data.
    extern uint32_t __ldm_data_start[];
    extern uint32_t __ldm_data_end[];
    l1_to_local_mem_copy(__ldm_data_start, data_image, __ldm_data_end - __ldm_data_start);
}

inline void noc_bank_table_init(uint64_t mem_bank_to_noc_addr) {
    int32_t dram_to_noc_size_bytes = sizeof(dram_bank_to_noc_xy);
    l1_to_local_mem_copy((uint*)dram_bank_to_noc_xy, (uint tt_l1_ptr*)mem_bank_to_noc_addr, dram_to_noc_size_bytes >> 2);
    int32_t l1_to_noc_size_bytes = sizeof(l1_bank_to_noc_xy);
    l1_to_local_mem_copy((uint*)l1_bank_to_noc_xy, (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes), l1_to_noc_size_bytes >> 2);

    int32_t dram_offsets_size_bytes = sizeof(bank_to_dram_offset);
    l1_to_local_mem_copy((uint*)bank_to_dram_offset, (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes + l1_to_noc_size_bytes), dram_offsets_size_bytes >> 2);
    int32_t l1_offsets_size_bytes = sizeof(bank_to_l1_offset);
    l1_to_local_mem_copy((uint*)bank_to_l1_offset, (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes + l1_to_noc_size_bytes + dram_offsets_size_bytes), l1_offsets_size_bytes >> 2);
}

FORCE_INLINE
uint32_t firmware_config_init(
    tt_l1_ptr mailboxes_t* const mailboxes, uint32_t core_type_index, uint32_t dispatch_class) {
    extern uint32_t tt_l1_ptr* rta_l1_base;
    extern uint32_t tt_l1_ptr* crta_l1_base;
    extern uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT];

    // TODO: check the asm for this loop to be sure loads are scheduled ok
    uint32_t kernel_config_base[ProgrammableCoreType::COUNT];
    launch_msg_t* launch_msg_address = &(mailboxes->launch[mailboxes->launch_msg_rd_ptr]);
#pragma GCC unroll ProgrammableCoreType::COUNT
    for (uint32_t index = 0; index < ProgrammableCoreType::COUNT; index++) {
        kernel_config_base[index] = launch_msg_address->kernel_config.kernel_config_base[index];
        sem_l1_base[index] =
            (uint32_t tt_l1_ptr*)(kernel_config_base[index] + launch_msg_address->kernel_config.sem_offset[index]);
    }
    rta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                        launch_msg_address->kernel_config.rta_offset[dispatch_class].rta_offset);
    crta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                         launch_msg_address->kernel_config.rta_offset[dispatch_class].crta_offset);

    return kernel_config_base[core_type_index];
}

FORCE_INLINE
void wait_for_go_message() {
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);

    while (mailboxes->go_message.signal != RUN_MSG_GO) {
        invalidate_l1_cache();
    }
}

#if !defined(COMPILE_FOR_TRISC)
FORCE_INLINE uint64_t calculate_dispatch_addr(volatile go_msg_t* go_message_in) {
    go_msg_t go_message;
    go_message.all = go_message_in->all;
    uint64_t addr = NOC_XY_ADDR(
        NOC_X(go_message.master_x),
        NOC_Y(go_message.master_y),
        DISPATCH_MESSAGE_ADDR + NOC_STREAM_REG_SPACE_SIZE * go_message.dispatch_message_offset);
    return addr;
}

FORCE_INLINE void notify_dispatch_core_done(uint64_t dispatch_addr, uint8_t noc_index) {
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
}
#endif
