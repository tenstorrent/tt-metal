// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** @file @brief Main firmware code */

#include <unistd.h>
#include <cstdint>

#include "risc.h"
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "ckernel_globals.h"
#include "run_sync.h"
#include "tools/profiler/kernel_profiler.hpp"

#include "debug_status.h"
#include "debug_print.h"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"


// TODO(pgk) move this too
static_assert(MEM_KERNEL_LAUNCH_PACKET_MAILBOX_ADDRESS % 16 == 0);

constexpr uint32_t RISCV_IC_BRISC_MASK = 0x1;
constexpr uint32_t RISCV_IC_TRISC0_MASK = 0x2;
constexpr uint32_t RISCV_IC_TRISC1_MASK = 0x4;
constexpr uint32_t RISCV_IC_TRISC2_MASK = 0x8;
constexpr uint32_t RISCV_IC_TRISC_ALL_MASK = RISCV_IC_TRISC0_MASK | RISCV_IC_TRISC1_MASK | RISCV_IC_TRISC2_MASK;

volatile tt_l1_ptr uint32_t * const brisc_run = (volatile tt_l1_ptr uint32_t *)(MEM_RUN_MAILBOX_ADDRESS);
volatile tt_l1_ptr run_sync_message_t * const slave_run = (volatile tt_l1_ptr run_sync_message_t *)(MEM_SLAVE_RUN_MAILBOX_ADDRESS);
volatile tt_l1_ptr uint32_t * const ncrisc_resume_addr = (volatile tt_l1_ptr uint32_t *)MEM_NCRISC_RESUME_ADDR_MAILBOX_ADDRESS;

c_tensix_core core;

volatile tt_l1_ptr uint32_t* instrn_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* pc_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* mailbox[MAX_THREADS];

volatile uint32_t local_mem_barrier __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t noc_size_x __attribute__((used));
uint8_t noc_size_y __attribute__((used));
uint8_t kernel_noc_id_var __attribute__((used));
uint64_t dispatch_addr __attribute__((used));

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

bool staggered_start_enabled() {
    uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    return soft_reset_0 & (1u << 31);
}

void stagger_startup() {
    if (staggered_start_enabled()) {
        const uint32_t NOC_ID_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1;
        uint32_t noc_id = noc_local_node_id() & 0xFFF;
        uint32_t noc_id_x = noc_id & NOC_ID_MASK;
        uint32_t noc_id_y = (noc_id >> NOC_ADDR_NODE_ID_BITS) & NOC_ID_MASK;

        uint32_t flat_id = (noc_id_y - 1) * 12 + (noc_id_x - 1);
        // To stagger 120 cores by 500us at 1.5GHz works out to 6250 AICLK per core.
        // Use an easy-to-multiply constant close to that.
        uint32_t delay = flat_id * ((1 << 12) | (1 << 11));

        uint64_t end = core.read_wall_clock() + delay;

        while (core.read_wall_clock() < end) { /* empty */
        }
    }
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

void l1_to_ncrisc_iram_copy() {
    // Copy NCRISC firmware from L1 to local IRAM using tensix DMA
    tdma_xmov(
        TDMA_MOVER0,
        (MEM_NCRISC_INIT_IRAM_L1_BASE) >> 4,
        (0x4 << 12),
        (MEM_NCRISC_IRAM_SIZE) >> 4,
        XMOV_L1_TO_L0);
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

    mailbox[0] = core.mailbox_base(0);
    mailbox[1] = core.mailbox_base(1);
    mailbox[2] = core.mailbox_base(2);

    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);

    stagger_startup();

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

    // Read counter at start
    core.wall_clock_mailbox()[0] = core.read_wall_clock();
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
    slave_run->all = RUN_SYNC_MESSAGE_ALL_SLAVES_DONE;

    l1_to_ncrisc_iram_copy();

    // Bring ncrisc/triscs out of reset
    deassert_all_reset();
}

inline void set_ncrisc_kernel_resume_deassert_address()
{
    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
    DEBUG_STATUS('I', 'N', 'W');
    while (*ncrisc_resume_addr == 0);
    DEBUG_STATUS('I', 'N', 'D');
    cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = *ncrisc_resume_addr;
}

inline void run_ncrisc_trisc()
{
    bool use_triscs = *(volatile tt_l1_ptr uint32_t*)(MEM_ENABLE_TRISC_MAILBOX_ADDRESS);
    if (use_triscs) {
        slave_run->all = RUN_SYNC_MESSAGE_ALL_TRISCS_GO;
    }

    bool use_ncrisc = *(volatile tt_l1_ptr uint32_t*)(MEM_ENABLE_NCRISC_MAILBOX_ADDRESS);
    if (use_ncrisc) {
        slave_run->ncrisc = RUN_SYNC_MESSAGE_GO;

        // TODO(pgk): don't copy all of iram! 1K cycles?
        l1_to_ncrisc_iram_copy();

        // Note: only ncrisc is in reset, so just deasserts ncrisc
        deassert_all_reset();
    }
}

inline void wait_ncrisc_trisc()
{
    DEBUG_STATUS('N', 'T', 'W');
    while (slave_run->all != RUN_SYNC_MESSAGE_ALL_SLAVES_DONE);
    DEBUG_STATUS('N', 'T', 'D');
}

int main() {

    DEBUG_STATUS('I');

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, (uint tt_l1_ptr *)MEM_BRISC_INIT_LOCAL_L1_BASE, num_words);


    RISC_POST_STATUS(0x10000000);

    risc_init();
    device_setup();

    // Set ncrisc's resume address to 0 so we know when ncrisc has overwritten it
    *ncrisc_resume_addr = 0;
    deassert_ncrisc_trisc();
    set_ncrisc_kernel_resume_deassert_address();

    // Wait for ncrisc to halt
    DEBUG_STATUS('I', 'N', 'W');
    while (slave_run->ncrisc != RUN_SYNC_MESSAGE_DONE);
    DEBUG_STATUS('I', 'N', 'D');

    // Cleanup profiler buffer incase we never get the go message
    kernel_profiler::init_profiler();
    while (1) {

        init_sync_registers();
        assert_just_ncrisc_reset();

        // Wait...
        DEBUG_STATUS('G', 'W');
        while (*brisc_run != RUN_MESSAGE_GO);
        DEBUG_STATUS('G', 'D');

        kernel_profiler::init_profiler();
        kernel_profiler::mark_time(CC_MAIN_START);

        // Invalidate the i$ now the kernels have loaded and before running
        volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
        cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK;

        run_ncrisc_trisc();

        // Run the BRISC kernel
        DEBUG_STATUS('R');
        kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
        kernel_init();
        kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
        DEBUG_STATUS('D');

        wait_ncrisc_trisc();

        *brisc_run = RUN_MESSAGE_DONE;

        // Not including any dispatch related code
        kernel_profiler::mark_time(CC_MAIN_END);

        // Notify dispatcher core that it has completed
        if (dispatch_addr != 0) {
            noc_fast_atomic_increment(kernel_noc_id_var, NCRISC_AT_CMD_BUF, dispatch_addr, 1, 31 /*wrap*/, false /*linked*/);
        }

    }

    return 0;
}
