// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
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
#include "dev_msgs.h"
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
    // Workaround for BH inline writes does not apply here because this writes to a stream register.
    // See comment in `noc_get_interim_inline_value_addr` for more details.
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

#if defined(DEBUG_EARLY_RETURN_KERNELS) && !defined(DISPATCH_KERNEL)
// Used to early-return when NULLing out kernels. Will always return true while a kernel is running, but can't be
// optimized away.
FORCE_INLINE
bool is_message_go() {
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);

    return mailboxes->go_message.signal == RUN_MSG_GO;
}

#define EARLY_RETURN_FOR_DEBUG \
    if (is_message_go()) { goto early_debug_exit; }
#define EARLY_RETURN_FOR_DEBUG_EXIT early_debug_exit:
#else
#define EARLY_RETURN_FOR_DEBUG
#define EARLY_RETURN_FOR_DEBUG_EXIT
#endif

inline __attribute__((always_inline)) void configure_gathering() {
#if defined(ARCH_BLACKHOLE) && !defined(ENABLE_GATHERING)
    // Workaround for tt-metal#16439, making sure gathering multiple instructions issued to Tensix is disabled
    // Brisc does not issue Tensix instructions but to be consistent for all riscs around Tensix we disable it
    // Disable gathering: set bit 18
    asm(R"ASM(
        .option push
        li   t1, 0x2
        csrrs zero, 0x7c0, t1
        li   t1, 0x1
        slli t1, t1, 18
        fence
        csrrs zero, 0x7c0, t1
        li   t1, 0x2
        csrrc zero, 0x7c0, t1
        fence
        .option pop
         )ASM" ::
            : "t1");
#endif
}

inline __attribute__((always_inline)) void configure_l1_data_cache() {
#if defined(ARCH_BLACKHOLE)
#if defined(DISABLE_L1_DATA_CACHE)
    // Disables Blackhole's L1 cache by setting bit 3. Grayskull and Wormhole do not have L1 cache
    // L1 cache can be disabled by setting `TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS` env var
    // export TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS=<BR,NC,TR*,ER*>
    asm(R"ASM(
            li t1, 0x8
            csrrs zero, 0x7c0, t1
             )ASM" ::
            : "t1");
#elif !defined(ENABLE_HW_CACHE_INVALIDATION)
    // Disable gathering to stop HW from invalidating the data cache after 128 transactions by setting bit 24
    // This is default enabled
    asm(R"ASM(
            li   t1, 0x1
            slli t1, t1, 24
            fence
            csrrs zero, 0x7c0, t1
             )ASM" ::
            : "t1");
#endif
#endif
}

inline __attribute__((always_inline)) void disable_relaxed_memory_ordering() {
#if defined(ARCH_BLACKHOLE) && defined(DISABLE_RELAXED_MEMORY_ORDERING)
    // Disable relaxed ordering which allows loads to bypass stores when going to separate addresses (bit 0)
    asm(R"ASM(
        li t1, 0x1
        csrrs zero, 0x7c0, t1
            )ASM" ::
            : "t1");
#endif
}

inline __attribute__((always_inline)) void configure_csr() {
    configure_gathering();
    configure_l1_data_cache();
    disable_relaxed_memory_ordering();
}
