// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/tunneling.h"
#include "dev_mem_map.h"
#include "eth_fw_api.h"
#include "internal/ethernet/erisc.h"

#include "internal/debug/watcher_common.h"
#include "api/debug/waypoint.h"

uint8_t noc_index;
// Renamed to kg_noc_mode to avoid conflict with noc_mode in dataflow_api_comon
// noc_mode is the same for all erisc kernels in the program
uint8_t kg_noc_mode;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
uint32_t traceCount __attribute__((used));
}  // namespace kernel_profiler
#endif

void set_deassert_addresses() {
#if defined(ENABLE_2_ERISC_MODE)
    WRITE_REG(SUBORDINATE_AERISC_RESET_PC, MEM_SUBORDINATE_AERISC_FIRMWARE_BASE);
#endif
}

inline void run_subordinate_eriscs(uint32_t enables) {
#if defined(ENABLE_2_ERISC_MODE)
    tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;
    // List of subordinate eriscs to run
    if (enables & (1u << static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM1))) {
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
    }
#endif
}

inline void wait_subordinate_eriscs() {
#if defined(ENABLE_2_ERISC_MODE)
    WAYPOINT("SEW");
    tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;
    do {
        invalidate_l1_cache();
        // If the subordinate is using dynamic NOC mode, it may use NOC0 but we don't need to sync the counters
        // as they are in a shared L1 region with base firmware
        internal_::risc_context_switch(kg_noc_mode == DM_DYNAMIC_NOC);
    } while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
    WAYPOINT("SED");
#endif
}

// Copy from init scratch space to local memory
inline void initialize_local_memory() {
    uint32_t* data_image = (uint32_t*)MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH;
    extern uint32_t __ldm_data_start[];
    extern uint32_t __ldm_data_end[];
    const uint32_t ldm_data_size = (uint32_t)__ldm_data_end - (uint32_t)__ldm_data_start;
    // Copy data from data_image in __ldm_data_start for ldm_data_size bytes
    l1_to_local_mem_copy(__ldm_data_start, data_image, ldm_data_size);
}

#define STR(x) #x
#define STRINGIFY(s) STR(s)
#define MEM_ERISC_HALT_STACK_MAILBOX_ADDRESS (MEM_AERISC_MAILBOX_BASE + 4)
#define MEM_ERISC_L1_TEMP_STORAGE (0x00040000)

extern "C" __attribute__((naked, used)) void resume_from_reset() {
    __asm__ volatile(
        // Restore contents of local memory from L1
	    ".option push\n"
        ".option norelax\n"
        "li   t0, " STRINGIFY(MEM_ERISC_L1_TEMP_STORAGE) "\n\t" // src
        "li   t1, " STRINGIFY(MEM_LOCAL_BASE) "\n\t"   // dst
        "li   t2, 2048\n\t"
        "0:\n\t"
        "lw   t3, 0(t0)\n\t"
        "sw   t3, 0(t1)\n\t"
        "addi t0, t0, 4\n\t"
        "addi t1, t1, 4\n\t"
        "addi t2, t2, -1\n\t"
        "bnez t2, 0b\n\t"
        "lw  sp, " STRINGIFY(MEM_ERISC_HALT_STACK_MAILBOX_ADDRESS) "( zero )\n"
        // Restore contents of callee-saved registers from the stack.
        "lw  ra, 0 * 4( sp )\n"
        "lw  s0, 1 * 4( sp )\n"
        "lw  s1, 2 * 4( sp )\n"
        "lw  s2, 3 * 4( sp )\n"
        "lw  s3, 4 * 4( sp )\n"
        "lw  s4, 5 * 4( sp )\n"
        "lw  s5, 6 * 4( sp )\n"
        "lw  s6, 7 * 4( sp )\n"
        "lw  s7, 8 * 4( sp )\n"
        "lw  s8, 9 * 4( sp )\n"
        "lw  s9, 10 * 4( sp )\n"
        "lw  s10, 11 * 4( sp )\n"
        "lw  s11, 12 * 4( sp )\n"
        "lw gp, 13 * 4( sp )\n"
        ".option pop\n"
        "addi sp, sp, (16 * 4)\n"
        "ret\n"
    );
}

// After running the base firmware, some core state (for erisc0) seems broken, so jumps into the kernel may occasionally
// hang. Resetting the core fixes the issue. We need to save all the GPR and local memory to L1, because local memory is
// cleared on reset. ERISC1 is responsible for triggering the reset, which willl start execution in resume_from_reset.
extern "C" __attribute__((naked)) void enter_reset(void) {
    __asm__ volatile(
        // Save contents to stack
        "addi sp, sp, -(16 * 4)\n"
        "sw ra, 0 * 4( sp )\n"
        "sw s0, 1 * 4( sp )\n"
        "sw s1, 2 * 4( sp )\n"
        "sw s2, 3 * 4( sp )\n"
        "sw s3, 4 * 4( sp )\n"
        "sw s4, 5 * 4( sp )\n"
        "sw s5, 6 * 4( sp )\n"
        "sw s6, 7 * 4( sp )\n"
        "sw s7, 8 * 4( sp )\n"
        "sw s8, 9 * 4( sp )\n"
        "sw s9, 10 * 4( sp )\n"
        "sw s10, 11 * 4( sp )\n"
        "sw s11, 12 * 4( sp )\n"
        "sw gp, 13 * 4( sp )\n"

        // Copy contents of local memory to L1, because local memory is cleared on reset. There is a register that in
        // theory can be set to prevent resetting the core from clearing local memory, but the hardware doesn't support it.
        "li   t0, " STRINGIFY(MEM_LOCAL_BASE) "\n\t"   // src
        "li   t1, " STRINGIFY(MEM_ERISC_L1_TEMP_STORAGE) "\n\t"   // dst (256 KiB)
        "li   t2, 2048\n\t"         // 8 KiB / 4B = 2048 words
        "0:\n\t"
        "lw   t3, 0(t0)\n\t"
        "sw   t3, 0(t1)\n\t"
        "addi t0, t0, 4\n\t"
        "addi t1, t1, 4\n\t"
        "addi t2, t2, -1\n\t"
        "bnez t2, 0b\n\t"

        // Store stack pointer to allow restoring after reset.
        "sw sp, " STRINGIFY(MEM_ERISC_HALT_STACK_MAILBOX_ADDRESS) "( zero )\n"
        // Infinite loop to prevent the core from continuing execution.
        "done_loop:\n"
        "j done_loop\n"
    );
}

int __attribute__((noinline)) main(void) {
    WAYPOINT("I");
    configure_csr();
    initialize_local_memory();
    noc_bank_table_init(MEM_AERISC_BANK_TO_NOC_SCRATCH);

    disable_interrupts();
    update_next_link_status_check_timestamp();

    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;

    risc_init();

#if defined(ENABLE_2_ERISC_MODE)
    subordinate_sync->all = RUN_SYNC_MSG_ALL_SUBORDINATES_DONE;
    subordinate_sync->dm1 = RUN_SYNC_MSG_INIT;

    // ERISC firmware >= 1.7.2 has already done this step. But on older firmware versions we need to do it here
    // and it will write to an "unused" region in base firmware.
    dynamic_noc_local_state_init();
#endif

    set_deassert_addresses();

    kg_noc_mode = DM_DEDICATED_NOC;
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    uint8_t prev_noc_mode = DM_DEDICATED_NOC;
    ncrisc_noc_full_sync();

#if defined(ENABLE_2_ERISC_MODE)
    deassert_all_reset();

    WRITE_REG(AERISC_RESET_PC, (uint32_t)(void*)resume_from_reset);
    enter_reset();
#endif
    wait_subordinate_eriscs();
    flag_disable[0] = 1;
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0

    // Add an invalidate before the first read of mailboxes->go_messages[0].signal
    invalidate_l1_cache();

    DeviceProfilerInit();
    while (1) {
        // Wait...
        WAYPOINT("GW");

        uint8_t go_message_signal = RUN_MSG_DONE;
        while ((go_message_signal = mailboxes->go_messages[0].signal) != RUN_MSG_GO) {
            invalidate_l1_cache();

            // While the go signal for kernel execution is not sent, check if the worker was signalled
            // to reset its launch message read pointer.
            if (flag_disable[0] != 1) {
                return 0;
            } else if (
                go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_RESET_READ_PTR_FROM_HOST ||
                go_message_signal == RUN_MSG_REPLAY_TRACE) {
                // Set the rd_ptr on workers to specified value
                mailboxes->launch_msg_rd_ptr = 0;
                if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
                    if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                        DeviceIncrementTraceCount();
                        DeviceTraceOnlyProfilerInit();
                    }
                    uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[0]);
                    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
                    // Notify dispatcher that this has been done
                    internal_::notify_dispatch_core_done(dispatch_addr);
                }
            } else {
                internal_::risc_context_switch();
            }
        }
        WAYPOINT("GD");

        {
            // Only include this iteration in the device profile if the launch message is valid. This is because all
            // workers get a go signal regardless of whether they're running a kernel or not. We don't want to profile
            // "invalid" iterations.
            DeviceZoneScopedMainN("ERISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);

            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);

            // Host side guarantees that if active_erisc is running on ERISC1 (single ERISC mode),
            // the noc_index will be NOC_1 which ensures no conflict with base firmware running concurrently on ERISC0
            //
            // cmd_buf allocation is determined based on the physical ERISC index in tt_metal/hw/inc/dataflow_cmd_bufs.h
            // ERISC0 is BRISC, ERISC1 is NCRISC.
            //
            kg_noc_mode = launch_msg_address->kernel_config.brisc_noc_mode;
            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;

            if (kg_noc_mode == DM_DEDICATED_NOC) {
                if (prev_noc_mode != kg_noc_mode) {
                    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                }
                noc_local_state_init(noc_index);
            } else {
                if (prev_noc_mode != kg_noc_mode) {
                    dynamic_noc_init();
                }
                dynamic_noc_local_state_init();
            }
            prev_noc_mode = kg_noc_mode;

            uint32_t enables = launch_msg_address->kernel_config.enables;
            run_subordinate_eriscs(enables);

            constexpr int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM0);
            if (enables & (1u << index)) {
                WAYPOINT("R");

                flush_erisc_icache();
                uint32_t kernel_config_base =
                    firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, PROCESSOR_INDEX);
                uint32_t kernel_lma =
                    kernel_config_base +
                    mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index];
                reinterpret_cast<void (*)()>(kernel_lma)();
                WAYPOINT("D");
            }

            wait_subordinate_eriscs();
            mailboxes->go_messages[0].signal = RUN_MSG_DONE;

            // Notify dispatcher core that it has completed
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[0]);
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                internal_::notify_dispatch_core_done(dispatch_addr);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    // Getting here is an invalid state
    return 0;
}
