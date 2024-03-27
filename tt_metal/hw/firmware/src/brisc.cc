// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** @file @brief Main firmware code */

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "noc_addr_ranges_gen.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"
#include "dataflow_api.h"

#include "debug/status.h"
#include "debug/dprint.h"

uint8_t noc_index;

constexpr uint32_t RISCV_IC_BRISC_MASK = 0x1;
constexpr uint32_t RISCV_IC_TRISC0_MASK = 0x2;
constexpr uint32_t RISCV_IC_TRISC1_MASK = 0x4;
constexpr uint32_t RISCV_IC_TRISC2_MASK = 0x8;
constexpr uint32_t RISCV_IC_TRISC_ALL_MASK = RISCV_IC_TRISC0_MASK | RISCV_IC_TRISC1_MASK | RISCV_IC_TRISC2_MASK;

constexpr uint32_t num_cbs_to_early_init = 4;  // safe small number to overlap w/ ncrisc copy

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);
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
uint32_t atomic_ret_val __attribute__ ((section ("l1_data"))) __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#define MEM_MOVER_VIEW_IRAM_BASE_ADDR (0x4 << 12)

namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}

void enable_power_management() {
    // Mask and Hyst taken from tb_tensix math_tests
    uint32_t pm_mask = 0xFFFF;
    uint32_t pm_hyst = 32;
    {
        // Important: program hyteresis first then enable, otherwise the en_pulse will fail to latch the value
        uint32_t hyst_val = pm_hyst & 0x7f;

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

    // core.ex_setc16(CG_CTRL_EN_Hyst_ADDR32, command_data[1] >> 16, instrn_buf[0]);
    core.ex_setc16(CG_CTRL_EN_Regblocks_ADDR32, pm_mask, instrn_buf[0]);

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

    cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = MEM_NCRISC_IRAM_BASE;
    cfg_regs[TRISC_RESET_PC_SEC0_PC_ADDR32] = MEM_TRISC0_BASE;
    cfg_regs[TRISC_RESET_PC_SEC1_PC_ADDR32] = MEM_TRISC1_BASE;
    cfg_regs[TRISC_RESET_PC_SEC2_PC_ADDR32] = MEM_TRISC2_BASE;
    cfg_regs[TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0b111;
    cfg_regs[NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0x1;
}

void l1_to_ncrisc_iram_copy(uint32_t src, uint32_t dst, uint16_t size) {
    // Copy NCRISC firmware from L1 to local IRAM using tensix DMA
    tdma_xmov(
        TDMA_MOVER0,
        src,
        dst,
        size,
        XMOV_L1_TO_L0);
}

void l1_to_ncrisc_iram_copy_wait() {
    // Wait for DMA to finish
    wait_tdma_movers_done(RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK);
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
    cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK;

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

    // Initialize sempahores - check if we need to do this still
    // math->packer semaphore - max set to 1, as double-buffering is disabled by default
    core.ex_sem_init(ckernel::semaphore::MATH_PACK, 1, 0, instrn_buf[0]);

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

inline void deassert_ncrisc_trisc()
{
    // Below sets ncrisc to go so we can wait until it is cleared on first iteration
    mailboxes->slave_sync.all = RUN_SYNC_MSG_ALL_SLAVES_DONE;

    uint16_t fw_size16 = mailboxes->launch.ncrisc_kernel_size16;
    ncrisc_kernel_start_offset16 = fw_size16;

    l1_to_ncrisc_iram_copy(MEM_NCRISC_INIT_IRAM_L1_BASE >> 4, MEM_MOVER_VIEW_IRAM_BASE_ADDR, fw_size16);
    l1_to_ncrisc_iram_copy_wait();

    // Bring ncrisc/triscs out of reset
    deassert_all_reset();
}

inline void set_ncrisc_kernel_resume_deassert_address()
{
    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
    DEBUG_STATUS('I', 'N', 'W');
    while (mailboxes->ncrisc_halt.resume_addr == 0);
    DEBUG_STATUS('I', 'N', 'D');
    cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = mailboxes->ncrisc_halt.resume_addr;
}

inline void run_triscs()
{
    if (mailboxes->launch.enable_triscs) {
        mailboxes->slave_sync.all = RUN_SYNC_MSG_ALL_TRISCS_GO;
    }
}

inline void finish_ncrisc_copy_and_run()
{
   if (mailboxes->launch.enable_ncrisc) {
       mailboxes->slave_sync.ncrisc = RUN_SYNC_MSG_GO;

       l1_to_ncrisc_iram_copy_wait();

       // Note: only ncrisc is in reset, so just deasserts ncrisc
       deassert_all_reset();
   }
}

inline void wait_ncrisc_trisc()
{
    DEBUG_STATUS('N', 'T', 'W');
    while (mailboxes->slave_sync.all != RUN_SYNC_MSG_ALL_SLAVES_DONE);
    DEBUG_STATUS('N', 'T', 'D');
}

int main() {

    DEBUG_STATUS('I');

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, (uint tt_l1_ptr *)MEM_BRISC_INIT_LOCAL_L1_BASE, num_words);

    risc_init();
    device_setup();
    noc_init();

    // Set ncrisc's resume address to 0 so we know when ncrisc has overwritten it
    mailboxes->ncrisc_halt.resume_addr = 0;
    mailboxes->slave_sync.ncrisc = RUN_SYNC_MSG_GO;
    deassert_ncrisc_trisc();
    set_ncrisc_kernel_resume_deassert_address();

    // Wait for ncrisc to halt
    DEBUG_STATUS('I', 'N', 'W');
    while (mailboxes->slave_sync.ncrisc != RUN_SYNC_MSG_DONE);
    DEBUG_STATUS('I', 'N', 'D');

    mailboxes->launch.run = RUN_MSG_DONE;

    while (1) {
        init_sync_registers();
        assert_just_ncrisc_reset();

        DEBUG_STATUS('G', 'W');
        while (mailboxes->launch.run != RUN_MSG_GO);
        DEBUG_STATUS('G', 'D');

        {
            DeviceZoneScopedMainN("BRISC-FW");

            // Always copy ncrisc even if its size is 0 (save branch)...
            l1_to_ncrisc_iram_copy((MEM_NCRISC_INIT_IRAM_L1_BASE >> 4) + ncrisc_kernel_start_offset16,
                                   MEM_MOVER_VIEW_IRAM_BASE_ADDR + ncrisc_kernel_start_offset16,
                                   mailboxes->launch.ncrisc_kernel_size16);

            // Invalidate the i$ now the kernels have loaded and before running
            volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
            cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK;

            run_triscs();

            noc_index = mailboxes->launch.brisc_noc_id;

            setup_cb_read_write_interfaces(0, num_cbs_to_early_init, true, true);
            finish_ncrisc_copy_and_run();

            // Run the BRISC kernel
            DEBUG_STATUS('R');
            if (mailboxes->launch.enable_brisc) {
                setup_cb_read_write_interfaces(num_cbs_to_early_init, mailboxes->launch.max_cb_index, true, true);
                kernel_init();
            } else {
                // This was not initialized in kernel_init
                noc_local_state_init(noc_index);
            }
            DEBUG_STATUS('D');

            wait_ncrisc_trisc();

            mailboxes->launch.run = RUN_MSG_DONE;

            // Notify dispatcher core that it has completed
            if (mailboxes->launch.mode == DISPATCH_MODE_DEV) {
                uint64_t dispatch_addr = NOC_XY_ADDR(NOC_X(DISPATCH_CORE_X), NOC_Y(DISPATCH_CORE_Y), DISPATCH_MESSAGE_ADDR);
                noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
            }
        }
    }

    return 0;
}
