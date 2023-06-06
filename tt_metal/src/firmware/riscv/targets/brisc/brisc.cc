/** @file @brief Main firmware code */

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"


#include "debug_print.h"

// TODO: commonize this w/ the runtime -- it's the same configs
// these consts must be constexprs
constexpr uint32_t TRISC_L1_MAILBOX_OFFSET = MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET;

constexpr uint32_t trisc_sizes[3] = {MEM_TRISC0_SIZE, MEM_TRISC1_SIZE, MEM_TRISC2_SIZE};

constexpr uint32_t trisc_mailbox_addresses[3] = {
    MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC0_OFFSET,
    MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC1_OFFSET,
    MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC2_OFFSET};

c_tensix_core core;

volatile uint32_t* instrn_buf[MAX_THREADS];
volatile uint32_t* pc_buf[MAX_THREADS];
volatile uint32_t* mailbox[MAX_THREADS];

volatile uint32_t local_mem_barrier;

uint8_t my_x[NUM_NOCS];
uint8_t my_y[NUM_NOCS];
#ifdef NOC_INDEX
uint8_t loading_noc = NOC_INDEX;
#else
uint8_t loading_noc = 0;  // BRISC uses NOC-0
#endif

uint8_t noc_size_x;
uint8_t noc_size_y;

// to reduce the amount of code changes, both BRISC and NRISCS instatiate these counter for both NOCs (ie NUM_NOCS)
// however, atm NCRISC uses only NOC-1 and BRISC uses only NOC-0
// this way we achieve full separation of counters / cmd_buffers etc.
uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

int post_index;

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

void set_trisc_address() {
    volatile uint32_t* cfg_regs = core.cfg_regs_base(0);

    // cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = l1_mem::address_map::NCRISC_FIRMWARE_BASE;
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

    volatile uint32_t* cfg_regs = core.cfg_regs_base(0);

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

    set_trisc_address();

    wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE);

    volatile uint32_t* use_ncrisc = (volatile uint32_t*)(RUNTIME_CONFIG_BASE);
    #ifdef DEVICE_DISPATCH_MODE
    *use_ncrisc = true;
    #endif
    if (*use_ncrisc) {
        l1_to_ncrisc_iram_copy();
        // Bring NCRISC out of reset, keep TRISCs under reset
        WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, 0x7000);
    }

    // Invalidate tensix icache for all 4 risc cores
    cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = 0xf;

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

    // Check before clearing debug mailboxes below
    // bool debugger_en = debugger::is_enabled();

    // Initialize debug mailbox to 0s
    for (int i = 0; i < MEM_DEBUG_MAILBOX_SIZE; i++) ((uint32_t *)MEM_DEBUG_MAILBOX_ADDRESS)[i] = 0;

    // Read counter at start
    core.wall_clock_mailbox()[0] = core.read_wall_clock();
}


#include "dataflow_api.h"
#include "kernel.cpp"

int main() {

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, (uint*)MEM_BRISC_INIT_LOCAL_L1_BASE, num_words);

    kernel_profiler::init_BR_profiler();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_MAIN_START);
#endif
    RISC_POST_STATUS(0x10000000);

    // note: BRISC uses NOC0, NCRISC uses NOC1
    // "risc_init" currently initialized global variables for both NOCs, but it is just reg reads, and it's probably
    // fine because this is done before we launch NCRISC (in "device_setup")
    // TODO: we could specialize it via "noc_id", in the same manner as "noc_init" (see below)
    risc_init();

#if not defined(DEVICE_DISPATCH_MODE) or defined(IS_DISPATCH_KERNEL)
    volatile uint32_t* enable_core_mailbox_ptr =
        (volatile uint32_t*)(MEM_ENABLE_CORE_MAILBOX);
    while (enable_core_mailbox_ptr[0] != 0x1);
#endif

    init_sync_registers();  // this init needs to be done before NCRISC / TRISCs are launched, only done by BRISC

#if defined(IS_DISPATCH_KERNEL)
    setup_cq_read_write_interface();
#else
    setup_cb_read_write_interfaces();                // done by both BRISC / NCRISC
#endif

    init_dram_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC
    init_l1_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC

    device_setup();  // NCRISC is disabled/enabled here

    noc_init(loading_noc);

    volatile uint32_t* use_triscs = (volatile uint32_t*)(RUNTIME_CONFIG_BASE + 4);

    #ifdef DEVICE_DISPATCH_MODE
    *use_triscs = true;
    #endif

    if (*use_triscs) {
        // FIXME: this is not sufficient to bring Trisc / Tensix out of a bad state
        // do we need do more than just assert_trisc_reset() ?
        // for now need to call /device/bin/silicon/tensix-reset from host when TRISCs/Tensix get into a bad state
        assert_trisc_reset();

        // Bring TRISCs out of reset
        deassert_trisc_reset();
    }

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
#endif
    // Run the BRISC kernel

kernel_main();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif

    if (*use_triscs) {
        // Wait for all the TRISCs to finish (it assumes all 3 TRISCs have been launched and will finish)
        while (
            !(*((volatile uint32_t*)trisc_mailbox_addresses[0]) == 1 &&
              *((volatile uint32_t*)trisc_mailbox_addresses[1]) == 1 &&
              *((volatile uint32_t*)trisc_mailbox_addresses[2]) == 1)) {
        }

        // Once all 3 have finished, assert reset on all of them
        assert_trisc_reset();
    }

    volatile uint32_t* test_mailbox_ptr =
        (volatile uint32_t*)(MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET);
    if (test_mailbox_ptr[0] != RISC_DETECTED_STREAM_ASSERT)
        test_mailbox_ptr[0] = 0x1;

// disable core once we're done
#if not defined(DEVICE_DISPATCH_MODE) or defined(IS_DISPATCH_KERNEL)
    enable_core_mailbox_ptr[0] = 0x0;
#endif

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_MAIN_END);
#endif

#if defined(DEVICE_DISPATCH_MODE) and not defined(IS_DISPATCH_KERNEL)
    // Notify dispatcher core that it has completed
    volatile uint64_t* dispatch_addr = reinterpret_cast<volatile uint64_t*>(DISPATCH_MESSAGE_ADDR);

    noc_semaphore_inc(*dispatch_addr, 1);
#endif

    while (true);
    return 0;
}
