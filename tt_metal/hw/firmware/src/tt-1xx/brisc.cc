// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** @file @brief Main firmware code */

#include <cstdint>

// clang-format off
#undef PROFILE_NOC_EVENTS
#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "internal/circular_buffer_init.h"
#include "dev_mem_map.h"
#include "noc_overlay_parameters.h"

#include "internal/debug/watcher_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"

// clang-format on

uint8_t noc_index;

constexpr uint32_t RISCV_IC_BRISC_MASK = 0x1;
constexpr uint32_t RISCV_IC_NCRISC_MASK = 0x10;
constexpr uint32_t RISCV_IC_TRISC0_MASK = 0x2;
constexpr uint32_t RISCV_IC_TRISC1_MASK = 0x4;
constexpr uint32_t RISCV_IC_TRISC2_MASK = 0x8;
constexpr uint32_t RISCV_IC_TRISC_ALL_MASK = RISCV_IC_TRISC0_MASK | RISCV_IC_TRISC1_MASK | RISCV_IC_TRISC2_MASK;

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;

c_tensix_core core;

volatile tt_l1_ptr uint32_t* instrn_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* pc_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* mailbox[MAX_THREADS];

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));
uint8_t prev_noc_mode = DM_DEDICATED_NOC;

// These arrays are used to store the worker logical to virtual coordinate mapping
// Round up to nearest multiple of 4 to ensure uint32_t alignment for L1 to local copies
uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)] __attribute__((used));
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)] __attribute__((used));

#define MEM_MOVER_VIEW_IRAM_BASE_ADDR (0x4 << 12)

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
uint32_t traceCount __attribute__((used));
}  // namespace kernel_profiler
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
    *((volatile uint32_t*)RISCV_DEBUG_REG_CG_CTRL_EN) = 0x10000 | (pm_mask);
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
    *((volatile uint32_t*)RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;
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
    cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] =
        RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

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

inline void deassert_ncrisc_trisc() {
    // Below sets ncrisc to go so we can wait until it is cleared on first iteration
    subordinate_sync->all = RUN_SYNC_MSG_ALL_INIT;

    // Bring ncrisc/triscs out of reset
    deassert_all_reset();
}

inline void run_triscs(uint32_t enables) {
    // Wait for init_sync_registers to complete. Should always be done by the time we get here.
    while (subordinate_sync->trisc0 != RUN_SYNC_MSG_DONE) {
        invalidate_l1_cache();
    }

    if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::MATH0))) {
        subordinate_sync->trisc0 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc1 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc2 = RUN_SYNC_MSG_GO;
    }
}

inline void start_ncrisc_kernel_run_early(uint32_t enables) {
    // On Wormhole, start_ncrisc_kernel_run will reset NCRISC to start the
    // kernel running. We delay it until later to give the NCRISC time to load
    // CBs before we wait on it.
#if !defined(ARCH_WORMHOLE)
    if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1))) {
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
    }
#endif
}

inline void start_ncrisc_kernel_run(uint32_t enables) {
#if defined(ARCH_WORMHOLE)
    if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1))) {
        // The NCRISC behaves badly if it jumps from L1 to IRAM, so instead halt it and then reset it to the IRAM
        // address it provides.
        while (subordinate_sync->dm1 != RUN_SYNC_MSG_WAITING_FOR_RESET);
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
        volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
        cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = mailboxes->ncrisc_halt.resume_addr;
        assert_just_ncrisc_reset();
        // Wait a bit to ensure NCRISC has time to actually reset (otherwise it
        // may just continue where it left off). This wait value was chosen
        // empirically.
        riscv_wait(5);
        deassert_all_reset();
    }
#endif
}

inline void wait_ncrisc_trisc() {
    WAYPOINT("NTW");
    while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
#if defined(ARCH_WORMHOLE)
        // Avoid hammering L1 while other cores are trying to work. Seems not to
        // be needed on Blackhole, probably because invalidate_l1_cache takes
        // time.
        asm volatile("nop; nop; nop; nop; nop");
#endif
        invalidate_l1_cache();
    }
    WAYPOINT("NTD");
}

inline void trigger_sync_register_init() { subordinate_sync->trisc0 = RUN_SYNC_MSG_INIT_SYNC_REGISTERS; }

inline void barrier_remote_cb_interface_setup(uint8_t noc_index, uint32_t noc_mode, uint32_t end_cb_index) {
#if defined(ARCH_BLACKHOLE)
    // cq_dispatch does not update noc transaction counts so skip this barrier on the dispatch core
    if (end_cb_index != NUM_CIRCULAR_BUFFERS) {
        WAYPOINT("NABW");
        if (noc_mode == DM_DYNAMIC_NOC) {
            do {
                invalidate_l1_cache();
            } while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_index));
        } else {
            while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
        }
        invalidate_l1_cache();
        WAYPOINT("NABD");
    }
#endif
}

int main() {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t*)MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0
    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    risc_init();
    device_setup();

    // Set ncrisc's resume address to 0 so we know when ncrisc has overwritten it
    mailboxes->ncrisc_halt.resume_addr = 0;
    deassert_ncrisc_trisc();

    // Wait for all cores to be finished initializing before reporting initialization done.
    wait_ncrisc_trisc();
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;

    // Initialize the NoCs to a safe state
    // This ensures if we send any noc txns without running a kernel setup are valid
    // ex. Immediately after starting, we send a RUN_MSG_RESET_READ_PTR signal
    uint8_t noc_mode;
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    noc_local_state_init(noc_index);
    trigger_sync_register_init();

    DeviceProfilerInit();
    while (1) {
        WAYPOINT("GW");
        uint8_t go_message_signal = RUN_MSG_DONE;
        // kernel_configs.preload is last in the launch message. so other data is
        // valid by the time it's set. All multicast data from the dispatcher is
        // written in order, so it will arrive in order. We also have a barrier
        // before mcasting the launch message (as a hang workaround), which
        // ensures that the unicast data will also have been received.
        while (
            ((go_message_signal = mailboxes->go_messages[mailboxes->go_message_index].signal) != RUN_MSG_GO) &&
            !(mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.preload & DISPATCH_ENABLE_FLAG_PRELOAD)) {
            invalidate_l1_cache();
            // While the go signal for kernel execution is not sent, check if the worker was signalled
            // to reset its launch message read pointer.
            if ((go_message_signal == RUN_MSG_RESET_READ_PTR) ||
                (go_message_signal == RUN_MSG_RESET_READ_PTR_FROM_HOST) ||
                (go_message_signal == RUN_MSG_REPLAY_TRACE)) {
                // Set the rd_ptr on workers to specified value
                mailboxes->launch_msg_rd_ptr = 0;
                if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
                    if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                        DeviceIncrementTraceCount();
                        DeviceTraceOnlyProfilerInit();
                    }
                    uint32_t go_message_index = mailboxes->go_message_index;
                    // Querying the noc_index is safe here, since the RUN_MSG_RESET_READ_PTR go signal is currently
                    // guaranteed to only be seen after a RUN_MSG_GO signal, which will set the noc_index to a valid
                    // value. For future proofing, the noc_index value is initialized to 0, to ensure an invalid NOC txn
                    // is not issued.
                    uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                    mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;
                    // Notify dispatcher that this has been done
                    DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                    notify_dispatch_core_done(dispatch_addr, noc_index);
                }
            }
        }

        WAYPOINT("GD");

        {
            // Only include this iteration in the device profile if the launch message is valid. This is because all
            // workers get a go signal regardless of whether they're running a kernel or not. We don't want to profile
            // "invalid" iterations.
            DeviceZoneScopedMainN("BRISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
            uint32_t enables = launch_msg_address->kernel_config.enables;
            // Trigger the NCRISC to start loading CBs and IRAM as soon as possible.
            if (enables &
                (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1))) {
                subordinate_sync->dm1 = RUN_SYNC_MSG_LOAD;
            }
            // Copies from L1 to IRAM on chips where NCRISC has IRAM
            uint32_t kernel_config_base =
                firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, PROCESSOR_INDEX);
            // Invalidate the i$ now the kernels have loaded and before running
            volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
            cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] =
                RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

            run_triscs(enables);

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            noc_mode = launch_msg_address->kernel_config.brisc_noc_mode;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;

            // re-initialize the NoCs
            uint8_t cmd_buf;
            if (noc_mode == DM_DEDICATED_NOC) {
                if (prev_noc_mode != noc_mode) {
                    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                }
#ifdef ARCH_BLACKHOLE
                // Need to add this to allow adding barrier after setup_remote_cb_interfaces
                noc_local_state_init(noc_index);
#endif
                cmd_buf = BRISC_AT_CMD_BUF;
            } else {
                if (prev_noc_mode != noc_mode) {
                    dynamic_noc_init();
                }
                dynamic_noc_local_state_init();
                cmd_buf = DYNAMIC_NOC_BRISC_AT_CMD_BUF;
            }
            prev_noc_mode = noc_mode;

            uint32_t tt_l1_ptr* cb_l1_base =
                (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.local_cb_offset);
            start_ncrisc_kernel_run_early(enables);

            // Run the BRISC kernel
            WAYPOINT("R");
            int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM0);
            if (enables & (1u << index)) {
                uint32_t local_cb_mask = launch_msg_address->kernel_config.local_cb_mask;
                setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);
                cb_l1_base =
                    (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.remote_cb_offset);
                uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                experimental::setup_remote_cb_interfaces<true>(
                    cb_l1_base, end_cb_index, noc_index, noc_mode, true, cmd_buf);
                barrier_remote_cb_interface_setup(noc_index, noc_mode, end_cb_index);
                start_ncrisc_kernel_run(enables);
                uint32_t kernel_lma =
                    (kernel_config_base + launch_msg_address->kernel_config.kernel_text_offset[index]);
                auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                record_stack_usage(stack_free);
            } else {
#if defined(PROFILE_KERNEL)
                // This was not initialized in the kernel
                // Currently FW does not issue a barrier except when using profiler
                if (noc_mode == DM_DEDICATED_NOC) {
                    noc_local_state_init(noc_index);
                }
#endif
                // Brisc is responsible for issuing any noc cmds needed when initializing remote cbs
                // So have brisc setup remote cb interfaces even when brisc is not in use
                if (launch_msg_address->kernel_config.enables) {
                    cb_l1_base =
                        (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.remote_cb_offset);
                    uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                    experimental::setup_remote_cb_interfaces<true>(
                        cb_l1_base, end_cb_index, noc_index, noc_mode, true, cmd_buf);
                    barrier_remote_cb_interface_setup(noc_index, noc_mode, end_cb_index);
                }
                start_ncrisc_kernel_run(enables);
                wait_for_go_message();
            }
            WAYPOINT("D");

            wait_ncrisc_trisc();

            trigger_sync_register_init();

            if constexpr (ASSERT_ENABLED) {
                if (noc_mode == DM_DYNAMIC_NOC) {
                    WAYPOINT("NKFW");
                    // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed
                    // and the NOC interface is in a known idle state for the next kernel.
                    invalidate_l1_cache();
                    for (int noc = 0; noc < NUM_NOCS; noc++) {
                        ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_atomics_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_posted_writes_sent(noc));
                    }
                    WAYPOINT("NKFD");
                }
            }

#if defined(PROFILE_KERNEL)
            if (noc_mode == DM_DYNAMIC_NOC) {
                // re-init for profiler to able to run barrier in dedicated noc mode
                noc_local_state_init(noc_index);
            }
#endif

            uint32_t go_message_index = mailboxes->go_message_index;
            mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;

            // Notify dispatcher core that tensix has completed running kernels, if the launch_msg was populated
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                // Set launch message to invalid, so that the next time this slot is encountered, kernels are only run
                // if a valid launch message is sent.
                launch_msg_address->kernel_config.enables = 0;
                launch_msg_address->kernel_config.preload = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                // Only executed if watcher is enabled. Ensures that we don't report stale data due to invalid launch
                // messages in the ring buffer. Must be executed before the atomic increment, as after that the launch
                // message is no longer owned by us.
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                notify_dispatch_core_done(dispatch_addr, noc_index);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    return 0;
}
