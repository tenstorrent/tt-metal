// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** @file @brief Main firmware code */

#include <cstdint>

// clang-format off
#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"
#include "dataflow_api.h"
#include "dev_mem_map.h"

#include "debug/watcher_common.h"
#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

uint8_t noc_index;

constexpr uint32_t RISCV_IC_BRISC_MASK = 0x1;
constexpr uint32_t RISCV_IC_NCRISC_MASK = 0x10;
constexpr uint32_t RISCV_IC_TRISC0_MASK = 0x2;
constexpr uint32_t RISCV_IC_TRISC1_MASK = 0x4;
constexpr uint32_t RISCV_IC_TRISC2_MASK = 0x8;
constexpr uint32_t RISCV_IC_TRISC_ALL_MASK = RISCV_IC_TRISC0_MASK | RISCV_IC_TRISC1_MASK | RISCV_IC_TRISC2_MASK;

constexpr uint32_t num_cbs_to_early_init = 4;  // safe small number to overlap w/ ncrisc copy

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
uint32_t ncrisc_kernel_start_offset16;

c_tensix_core core;

volatile tt_l1_ptr uint32_t* instrn_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* pc_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* mailbox[MAX_THREADS];

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

#define MEM_MOVER_VIEW_IRAM_BASE_ADDR (0x4 << 12)

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

void enable_power_management() {
    // Mask and Hyst taken from tb_tensix math_tests
    uint32_t pm_mask = 0xFFFF;
    uint32_t pm_hyst = 32;

    #ifdef ARCH_BLACKHOLE
    uint32_t hyst_val = pm_hyst;
    #else
    // Important: program hyteresis first then enable, otherwise the en_pulse will fail to latch the value
    uint32_t hyst_val = pm_hyst & 0x7f;
    #endif

    {
        // Program slightly off values for each CG
        uint32_t hyst0_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;
        uint32_t hyst1_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;
        uint32_t hyst2_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;

        // Force slightly off values for each CG
        // uint32_t hyst0_reg_data = ((hyst_val+3) << 24) | ((hyst_val+2) << 16) | ((hyst_val+1) << 8) | (hyst_val+0);
        // uint32_t hyst1_reg_data = ((hyst_val-4) << 24) | ((hyst_val-3) << 16) | ((hyst_val-2) << 8) | (hyst_val-1);
        // uint32_t hyst2_reg_data = ((hyst_val-6) << 24) | ((hyst_val-5) << 16) | ((hyst_val+5) << 8) | (hyst_val+4);
        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST0, hyst0_reg_data);
        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST1, hyst1_reg_data);
        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST2, hyst2_reg_data);
    }

    #ifdef ARCH_BLACKHOLE
    /*FIXME: need to deal with srcb ctrl bit not fitting in 16 bits. For  */
    /*now just always turn it on */
    *((uint32_t volatile*)RISCV_DEBUG_REG_CG_CTRL_EN) = 0x10000 | (pm_mask);
    #else
    // core.ex_setc16(CG_CTRL_EN_Hyst_ADDR32, command_data[1] >> 16, instrn_buf[0]);
    core.ex_setc16(CG_CTRL_EN_Regblocks_ADDR32, pm_mask, instrn_buf[0]);
    #endif

    if (((pm_mask & 0x0100) >> 8) == 1) {  // enable noc clk gatting

        uint32_t hyst_val = pm_hyst & 0x7f;

        // FFB4_0090 - set bit 0 (overlay clkgt en)
        core.write_stream_register(
            0,
            STREAM_PERF_CONFIG_REG_INDEX,
            pack_field(1, CLOCK_GATING_EN_WIDTH, CLOCK_GATING_EN) |
                pack_field(hyst_val, CLOCK_GATING_HYST_WIDTH, CLOCK_GATING_HYST) |
                // XXX: This is a performance optimization for relay streams, not power management related
                pack_field(32, PARTIAL_SEND_WORDS_THR_WIDTH, PARTIAL_SEND_WORDS_THR));

        // FFB2_0100 - set bit 0 (NOC0 NIU clkgt en)
        uint32_t oldval;
        oldval = NOC_READ_REG(NOC0_REGS_START_ADDR + 0x100);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC0_REGS_START_ADDR + 0x100, oldval);

        // FFB2_0104 - set bit 0 (NOC0 router clkgt en)
        oldval = NOC_READ_REG(NOC0_REGS_START_ADDR + 0x104);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC0_REGS_START_ADDR + 0x104, oldval);

        // FFB3_0100 - set bit 0 (NOC1 NIU clkgt en)
        oldval = NOC_READ_REG(NOC1_REGS_START_ADDR + 0x100);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC1_REGS_START_ADDR + 0x100, oldval);

        // FFB3_0104 - set bit 0 (NOC1 router clkgt en)
        oldval = NOC_READ_REG(NOC1_REGS_START_ADDR + 0x104);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC1_REGS_START_ADDR + 0x104, oldval);
    }
}

void set_deassert_addresses() {
    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);

#ifdef ARCH_BLACKHOLE
    WRITE_REG(RISCV_DEBUG_REG_NCRISC_RESET_PC, MEM_NCRISC_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC0_RESET_PC, MEM_TRISC0_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC1_RESET_PC, MEM_TRISC1_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC2_RESET_PC, MEM_TRISC2_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC_RESET_PC_OVERRIDE, 0b111);
    WRITE_REG(RISCV_DEBUG_REG_NCRISC_RESET_PC_OVERRIDE, 0x1);
#else
    cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = MEM_NCRISC_FIRMWARE_BASE;
    cfg_regs[TRISC_RESET_PC_SEC0_PC_ADDR32] = MEM_TRISC0_FIRMWARE_BASE;
    cfg_regs[TRISC_RESET_PC_SEC1_PC_ADDR32] = MEM_TRISC1_FIRMWARE_BASE;
    cfg_regs[TRISC_RESET_PC_SEC2_PC_ADDR32] = MEM_TRISC2_FIRMWARE_BASE;
    cfg_regs[TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0b111;
    cfg_regs[NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0x1;
#endif
}

void l1_to_ncrisc_iram_copy(uint16_t size, uint32_t address_offset = 0) {
#ifdef NCRISC_HAS_IRAM
    // Always copy ncrisc even if its size is 0 (save branch)...
    // Copy NCRISC firmware from L1 to local IRAM using tensix DMA
    tdma_xmov(
        TDMA_MOVER0,
        (MEM_NCRISC_INIT_IRAM_L1_BASE >> 4) + address_offset,
        MEM_MOVER_VIEW_IRAM_BASE_ADDR + address_offset,
        size,
        XMOV_L1_TO_L0);
#endif
}

void l1_to_ncrisc_iram_copy_wait() {
#ifdef NCRISC_HAS_IRAM
    // Wait for DMA to finish
    wait_tdma_movers_done(RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK);
#endif
}

void device_setup() {
    instrn_buf[0] = core.instrn_buf_base(0);
    instrn_buf[1] = core.instrn_buf_base(1);
    instrn_buf[2] = core.instrn_buf_base(2);

    pc_buf[0] = core.pc_buf_base(0);
    pc_buf[1] = core.pc_buf_base(1);
    pc_buf[2] = core.pc_buf_base(2);

    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);

    // FIXME MT: enable later
    // enable_power_management();

#ifdef ARCH_BLACKHOLE
    // Disable DEST CG
    *((uint32_t volatile*)RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;
#endif

    WRITE_REG(RISCV_TDMA_REG_CLK_GATE_EN, 0x3f);  // Enable clock gating

    noc_set_active_instance(0);
    uint32_t niu_cfg0 = noc_get_cfg_reg(NIU_CFG_0);
    noc_set_cfg_reg(NIU_CFG_0, niu_cfg0 | 0x1);
    uint32_t router_cfg0 = noc_get_cfg_reg(ROUTER_CFG_0);
    noc_set_cfg_reg(ROUTER_CFG_0, router_cfg0 | 0x1);

    noc_set_active_instance(1);
    uint32_t niu_cfg1 = noc_get_cfg_reg(NIU_CFG_0);
    noc_set_cfg_reg(NIU_CFG_0, niu_cfg1 | 0x1);
    uint32_t router_cfg1 = noc_get_cfg_reg(ROUTER_CFG_0);
    noc_set_cfg_reg(ROUTER_CFG_0, router_cfg1 | 0x1);
    noc_set_active_instance(0);

    set_deassert_addresses();

    wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE);

    // Invalidate tensix icache for all 4 risc cores
    cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

    // Clear destination registers
    core.ex_zeroacc(instrn_buf[0]);

    // Enable CC stack
    core.ex_encc(instrn_buf[0]);

    // Set default sfpu constant register state
    core.ex_load_const(instrn_buf[0]);

    // Enable ECC scrubber
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Enable_RMW, 1);
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Scrub_On_Error_RMW, 1);
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Delay_RMW, 0x100);

    core.initialize_tensix_semaphores(instrn_buf[0]);

    // // unpacker semaphore
    // core.ex_sem_init(semaphore::UNPACK_MISC, 1, 1, instrn_buf[0]);

    // // unpacker sync semaphore
    // core.ex_sem_init(semaphore::UNPACK_SYNC, 2, 0, instrn_buf[0]);

    // // config state semaphore
    // core.ex_sem_init(semaphore::CFG_STATE_BUSY, MAX_CONFIG_STATES, 0, instrn_buf[0]);
}

void init_sync_registers() {
    volatile tt_reg_ptr uint* tiles_received_ptr;
    volatile tt_reg_ptr uint* tiles_acked_ptr;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        tiles_received_ptr = get_cb_tiles_received_ptr(operand);
        tiles_received_ptr[0] = 0;
        tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
        tiles_acked_ptr[0] = 0;
    }
}

inline void deassert_ncrisc_trisc() {
    // Below sets ncrisc to go so we can wait until it is cleared on first iteration
    mailboxes->slave_sync.all = RUN_SYNC_MSG_ALL_SLAVES_DONE;

    uint16_t fw_size16 = mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.ncrisc_kernel_size16;
    ncrisc_kernel_start_offset16 = fw_size16;

    // Copies from L1 to IRAM on chips where NCRISC has IRAM
    l1_to_ncrisc_iram_copy(fw_size16);
    l1_to_ncrisc_iram_copy_wait();

    // Bring ncrisc/triscs out of reset
    deassert_all_reset();
}

inline __attribute__((always_inline)) void wait_for_ncrisc_to_halt() {
#ifdef NCRISC_HAS_IRAM
    WAYPOINT("INW");
    while (mailboxes->slave_sync.ncrisc != RUN_SYNC_MSG_DONE);
    WAYPOINT("IND");
#endif
}

inline __attribute__((always_inline)) void reset_ncrisc_with_iram() {
#ifdef NCRISC_HAS_IRAM
    assert_just_ncrisc_reset();
#endif
}

inline void set_ncrisc_kernel_resume_deassert_address() {
#ifdef NCRISC_HAS_IRAM
    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
    WAYPOINT("INW");
    while (mailboxes->ncrisc_halt.resume_addr == 0);
    WAYPOINT("IND");
    cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = mailboxes->ncrisc_halt.resume_addr;
#endif
}

inline void run_triscs(dispatch_core_processor_masks enables) {
    if (enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE) {
        mailboxes->slave_sync.all = RUN_SYNC_MSG_ALL_TRISCS_GO;
    }
}

inline void finish_ncrisc_copy_and_run(dispatch_core_processor_masks enables) {
    if (enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1) {
        mailboxes->slave_sync.ncrisc = RUN_SYNC_MSG_GO;

        l1_to_ncrisc_iram_copy_wait();

        // Note: only ncrisc is in reset, so just deasserts ncrisc
        deassert_all_reset();
    }
}

inline void wait_ncrisc_trisc() {
    WAYPOINT("NTW");
    while (mailboxes->slave_sync.all != RUN_SYNC_MSG_ALL_SLAVES_DONE);
    WAYPOINT("NTD");
}

int main() {
    conditionally_disable_l1_cache();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, (uint tt_l1_ptr*)MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH, num_words);

    mailboxes->launch_msg_rd_ptr = 0; // Initialize the rdptr to 0
    noc_index = 0;
    risc_init();
    device_setup();

    // Set ncrisc's resume address to 0 so we know when ncrisc has overwritten it
    mailboxes->ncrisc_halt.resume_addr = 0;
    mailboxes->slave_sync.ncrisc = RUN_SYNC_MSG_GO;
    deassert_ncrisc_trisc();
    // When NCRISC has IRAM, it needs to be halted before data can be copied from L1 to IRAM
    // This routine allows us to resume NCRISC after the copy is done
    set_ncrisc_kernel_resume_deassert_address();

    // Wait for ncrisc to halt
    wait_for_ncrisc_to_halt();

    mailboxes->go_message.signal = RUN_MSG_DONE;

    uint8_t noc_mode;
    uint8_t prev_noc_mode = DM_INVALID_NOC;
    while (1) {
        init_sync_registers();
        reset_ncrisc_with_iram();

        WAYPOINT("GW");
        uint8_t go_message_signal = RUN_MSG_DONE;
        while ((go_message_signal = mailboxes->go_message.signal) != RUN_MSG_GO) {
            // While the go signal for kernel execution is not sent, check if the worker was signalled
            // to reset its launch message read pointer.
            if (go_message_signal == RUN_MSG_RESET_READ_PTR) {
                // Set the rd_ptr on workers to specified value
                mailboxes->launch_msg_rd_ptr = 0;
                // Querying the noc_index is safe here, since the RUN_MSG_RESET_READ_PTR go signal is currently guaranteed
                // to only be seen after a RUN_MSG_GO signal, which will set the noc_index to a valid value.
                // For future proofing, the noc_index value is initialized to 0, to ensure an invalid NOC txn is not issued.
                uint64_t dispatch_addr =
                    NOC_XY_ADDR(NOC_X(mailboxes->go_message.master_x),
                    NOC_Y(mailboxes->go_message.master_y), DISPATCH_MESSAGE_ADDR);
                mailboxes->go_message.signal = RUN_MSG_DONE;
                // Notify dispatcher that this has been done
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                noc_fast_atomic_increment(
                    noc_index,
                    NCRISC_AT_CMD_BUF,
                    dispatch_addr,
                    NOC_UNICAST_WRITE_VC,
                    1,
                    31 /*wrap*/,
                    false /*linked*/);
            }
        }

        WAYPOINT("GD");

        {
            // Only include this iteration in the device profile if the launch message is valid. This is because all workers get a go signal regardless of whether
            // they're running a kernel or not. We don't want to profile "invalid" iterations.
            DeviceZoneScopedMainN("BRISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
            // Copies from L1 to IRAM on chips where NCRISC has IRAM
            l1_to_ncrisc_iram_copy(launch_msg_address->kernel_config.ncrisc_kernel_size16, ncrisc_kernel_start_offset16);

            // Invalidate the i$ now the kernels have loaded and before running
            volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
            cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

            enum dispatch_core_processor_masks enables = (enum dispatch_core_processor_masks)launch_msg_address->kernel_config.enables;

            run_triscs(enables);

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            noc_mode = launch_msg_address->kernel_config.brisc_noc_mode;

            // re-initialize the NoCs
            if (prev_noc_mode != noc_mode) {
                if (noc_mode == DM_DEDICATED_NOC) {
                    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                } else {
                    dynamic_noc_init();
                }
            }
            prev_noc_mode = noc_mode;

            uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, DISPATCH_CLASS_TENSIX_DM0);
            uint32_t tt_l1_ptr *cb_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
                launch_msg_address->kernel_config.cb_offset);
            setup_cb_read_write_interfaces(cb_l1_base, 0, num_cbs_to_early_init, true, true, false);
            finish_ncrisc_copy_and_run(enables);

            // Run the BRISC kernel
            WAYPOINT("R");
            if (enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0) {
                setup_cb_read_write_interfaces(cb_l1_base, num_cbs_to_early_init, launch_msg_address->kernel_config.max_cb_index, true, true, false);
                kernel_init();
                RECORD_STACK_USAGE();
            } else {
                // This was not initialized in kernel_init
                if (noc_mode == DM_DEDICATED_NOC) {
                    noc_local_state_init(noc_index);
                }
            }
            WAYPOINT("D");

            wait_ncrisc_trisc();

            mailboxes->go_message.signal = RUN_MSG_DONE;

            // Notify dispatcher core that tensix has completed running kernels, if the launch_msg was populated
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                // Set launch message to invalid, so that the next time this slot is encountered, kernels are only run if a valid launch message is sent.
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr =
                    NOC_XY_ADDR(NOC_X(mailboxes->go_message.master_x),
                        NOC_Y(mailboxes->go_message.master_y), DISPATCH_MESSAGE_ADDR);
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                noc_fast_atomic_increment(
                    noc_index,
                    NCRISC_AT_CMD_BUF,
                    dispatch_addr,
                    NOC_UNICAST_WRITE_VC,
                    1,
                    31 /*wrap*/,
                    false /*linked*/);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
                // Only executed if watcher is enabled. Ensures that we don't report stale data due to invalid launch messages in the ring buffer
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
            }
        }
    }

    return 0;
}
