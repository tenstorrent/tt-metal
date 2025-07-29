// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "circular_buffer.h"
#include "dataflow_api.h"

#include "debug/watcher_common.h"
#include "debug/waypoint.h"
#include "debug/stack_usage.h"

uint8_t noc_index;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

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

//c_tensix_core core;

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

//inline void RISC_POST_STATUS(uint32_t status) {
//  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
//  ptr[0] = status;
//}

void set_deassert_addresses() {
#ifdef ARCH_BLACKHOLE
    WRITE_REG(SUBORDINATE_IERISC_RESET_PC, MEM_SUBORDINATE_IERISC_FIRMWARE_BASE);
#endif
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

inline void run_subordinate_eriscs(dispatch_core_processor_masks enables) {
    if (enables & DISPATCH_CLASS_MASK_ETH_DM1) {
        mailboxes->subordinate_sync.dm1 = RUN_SYNC_MSG_GO;
    }
}

inline void wait_subordinate_eriscs(uint32_t &heartbeat) {
    WAYPOINT("SEW");
    while (mailboxes->subordinate_sync.all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
        invalidate_l1_cache();
        RISC_POST_HEARTBEAT(heartbeat);
    }
    WAYPOINT("SED");
}

int main() {
    configure_csr();
    WAYPOINT("I");
    do_crt1((uint32_t *)MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH);
    uint32_t heartbeat = 0;

    noc_bank_table_init(MEM_IERISC_BANK_TO_NOC_SCRATCH);

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    risc_init();

    mailboxes->subordinate_sync.all = RUN_SYNC_MSG_ALL_SUBORDINATES_DONE;
#ifdef ARCH_BLACKHOLE
    mailboxes->subordinate_sync.dm1 = RUN_SYNC_MSG_INIT;
#endif
    set_deassert_addresses();
    //device_setup();

    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    deassert_all_reset(); // Bring all riscs on eth cores out of reset
    // Wait for all subordinate ERISCs to be ready before reporting the core is done initializing.
    wait_subordinate_eriscs(heartbeat);
    mailboxes->go_message.signal = RUN_MSG_DONE;
    mailboxes->launch_msg_rd_ptr = 0; // Initialize the rdptr to 0
    // Cleanup profiler buffer incase we never get the go message


    while (1) {

        init_sync_registers();
        // Wait...
        WAYPOINT("GW");
        while (mailboxes->go_message.signal != RUN_MSG_GO) {
            invalidate_l1_cache();
            RISC_POST_HEARTBEAT(heartbeat);
        };
        WAYPOINT("GD");

        {
            // Idle ERISC Kernels aren't given go-signals corresponding to empty launch messages. Always profile this iteration, since it's guaranteed to be valid.
            DeviceZoneScopedMainN("ERISC-IDLE-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;

            flush_erisc_icache();

            enum dispatch_core_processor_masks enables = (enum dispatch_core_processor_masks)launch_msg_address->kernel_config.enables;
            run_subordinate_eriscs(enables);

            uint32_t kernel_config_base =
                firmware_config_init(mailboxes, ProgrammableCoreType::IDLE_ETH, DISPATCH_CLASS_ETH_DM0);

            // Run the ERISC kernel
            if (enables & DISPATCH_CLASS_MASK_ETH_DM0) {
                WAYPOINT("R");
                int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM0);
                uint32_t kernel_lma = (kernel_config_base +
                                       launch_msg_address->kernel_config.kernel_text_offset[index]);
                auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                record_stack_usage(stack_free);
                WAYPOINT("D");
            }

            wait_subordinate_eriscs(heartbeat);

            mailboxes->go_message.signal = RUN_MSG_DONE;

            // Notify dispatcher core that it has completed
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_message);
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                notify_dispatch_core_done(dispatch_addr, noc_index);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    return 0;
}
