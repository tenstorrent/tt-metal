// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "core_config.h"
#include "ckernel_globals.h"
#include "internal/tensix_functions.h"
#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"
#include "dev_mem_map.h"
#include "hostdevcommon/kernel_structs.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"
#include "api/debug/dprint.h"
#include "risc_common.h"
#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#endif

constexpr size_t round_up_to_mult_of_4(size_t value) { return ((value + 3) / 4) * 4; }

extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
extern int32_t bank_to_l1_offset[NUM_L1_BANKS];

// These arrays are used to store the worker logical to virtual coordinate mapping. Only
// defined in cores that need this information for NOC transactions (e.g. DM cores).
// Round up to nearest multiple of 4 to ensure uint32_t alignment for L1 to local copies
extern uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)];
extern uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)];

void l1_to_local_mem_copy(uint32_t* dst, uint32_t tt_l1_ptr* src, int32_t len);

inline void do_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear bss.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // Copy initialized data.
    extern uint32_t __ldm_data_start[];
    extern uint32_t __ldm_data_end[];
    if (__ldm_data_start != data_image) {
        l1_to_local_mem_copy(__ldm_data_start, data_image, __ldm_data_end - __ldm_data_start);
    }
}

inline void do_thread_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear thread bss.
    extern thread_local uint32_t __ldm_tbss_start[];
    extern thread_local uint32_t __ldm_tbss_end[];
    wzerorange(__ldm_tbss_start, __ldm_tbss_end);

    // Copy thread initialized data.
    extern thread_local uint32_t __ldm_tdata_start[];
    extern thread_local uint32_t __ldm_tdata_end[];
    extern uint32_t __tdata_lma[];
    l1_to_local_mem_copy(__ldm_tdata_start, data_image, __ldm_tdata_end - __ldm_tdata_start);
}

inline void noc_bank_table_init(uint64_t mem_bank_to_noc_addr) {
    int32_t dram_to_noc_size_bytes = sizeof(dram_bank_to_noc_xy);
    l1_to_local_mem_copy(
        (uint*)dram_bank_to_noc_xy, (uint tt_l1_ptr*)mem_bank_to_noc_addr, dram_to_noc_size_bytes >> 2);
    int32_t l1_to_noc_size_bytes = sizeof(l1_bank_to_noc_xy);
    l1_to_local_mem_copy(
        (uint*)l1_bank_to_noc_xy,
        (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes),
        l1_to_noc_size_bytes >> 2);

    int32_t dram_offsets_size_bytes = sizeof(bank_to_dram_offset);
    l1_to_local_mem_copy(
        (uint*)bank_to_dram_offset,
        (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes + l1_to_noc_size_bytes),
        dram_offsets_size_bytes >> 2);
    int32_t l1_offsets_size_bytes = sizeof(bank_to_l1_offset);
    l1_to_local_mem_copy(
        (uint*)bank_to_l1_offset,
        (uint tt_l1_ptr*)(mem_bank_to_noc_addr + dram_to_noc_size_bytes + l1_to_noc_size_bytes +
                          dram_offsets_size_bytes),
        l1_offsets_size_bytes >> 2);
}

inline void noc_worker_logical_to_virtual_map_init(uint64_t worker_logical_to_virtual_map_addr) {
    int32_t worker_logical_col_to_virtual_col_size_bytes = sizeof(worker_logical_col_to_virtual_col);
    l1_to_local_mem_copy(
        (uint*)worker_logical_col_to_virtual_col,
        (uint tt_l1_ptr*)(worker_logical_to_virtual_map_addr),
        worker_logical_col_to_virtual_col_size_bytes >> 2);

    int32_t worker_logical_row_to_virtual_row_size_bytes = sizeof(worker_logical_row_to_virtual_row);
    l1_to_local_mem_copy(
        (uint*)worker_logical_row_to_virtual_row,
        (uint tt_l1_ptr*)(worker_logical_to_virtual_map_addr + worker_logical_col_to_virtual_col_size_bytes),
        worker_logical_row_to_virtual_row_size_bytes >> 2);
}

FORCE_INLINE
uint32_t firmware_config_init(
    tt_l1_ptr mailboxes_t* const mailboxes, uint32_t core_type_index, uint32_t processor_index) {
#ifdef ARCH_QUASAR
    extern thread_local uint32_t tt_l1_ptr* rta_l1_base;
    extern thread_local uint32_t tt_l1_ptr* crta_l1_base;
#else
    extern uint32_t tt_l1_ptr* rta_l1_base;
    extern uint32_t tt_l1_ptr* crta_l1_base;
#endif
    extern uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT];

    // TODO: check the asm for this loop to be sure loads are scheduled ok
    uintptr_t kernel_config_base[ProgrammableCoreType::COUNT];
    launch_msg_t* launch_msg_address = &(mailboxes->launch[mailboxes->launch_msg_rd_ptr]);
#pragma GCC unroll ProgrammableCoreType::COUNT
    for (uint32_t index = 0; index < ProgrammableCoreType::COUNT; index++) {
        kernel_config_base[index] = launch_msg_address->kernel_config.kernel_config_base[index];
        sem_l1_base[index] =
            (uint32_t tt_l1_ptr*)(kernel_config_base[index] + launch_msg_address->kernel_config.sem_offset[index]);
    }
    rta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                        launch_msg_address->kernel_config.rta_offset[processor_index].rta_offset);
    crta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                         launch_msg_address->kernel_config.rta_offset[processor_index].crta_offset);

    return kernel_config_base[core_type_index];
}

FORCE_INLINE
void wait_for_go_message() {
#ifdef ARCH_QUASAR
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE);
#else
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
    uint32_t go_message_index = mailboxes->go_message_index;

    while (mailboxes->go_messages[go_message_index].signal != RUN_MSG_GO) {
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
    uint32_t go_message_index = mailboxes->go_message_index;

    return mailboxes->go_messages[go_message_index].signal == RUN_MSG_GO;
}

#define EARLY_RETURN_FOR_DEBUG \
    if (is_message_go()) {     \
        goto early_debug_exit; \
    }
#define EARLY_RETURN_FOR_DEBUG_EXIT \
    early_debug_exit:
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
    // Blackhole's L1 cache can be disabled by setting bit 3 and enabled by clearing it. Grayskull and Wormhole do not
    // have L1 cache The cache is default disabled. When hw APIs better hide L1 cache, we can keep it enabled L1 cache
    // can be enabled by setting `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS` env var. It can be enabled risc by risc export
    // TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=<BR,NC,TR*,ER*>
#if defined(ENABLE_L1_DATA_CACHE)
    set_l1_data_cache<true>();
#else
    set_l1_data_cache<false>();
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

struct coord_t {
    uint8_t x;
    uint8_t y;
};

FORCE_INLINE coord_t get_virtual_coord_from_worker_logical_coord(uint8_t worker_x, uint8_t worker_y) {
    return {worker_logical_col_to_virtual_col[worker_x], worker_logical_row_to_virtual_row[worker_y]};
}
