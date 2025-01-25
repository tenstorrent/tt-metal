// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "circular_buffer.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"

#include "debug/watcher_common.h"
#include "debug/waypoint.h"
#include "debug/stack_usage.h"
#include "debug/dprint.h"

uint8_t noc_index;

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

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

// c_tensix_core core;

// tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

int main() {
    configure_l1_data_cache();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");
    do_crt1((uint32_t*)eth_l1_mem::address_map::MEM_ERISC_INIT_LOCAL_L1_BASE_SCRATCH);

    // put this into scratch space similar to idle erisc
    noc_bank_table_init(eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SCRATCH);

    // volatile tt_l1_ptr uint32_t* debug_addr_ptr =
    //     reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
    // debug_addr_ptr[0] = 0xDEADDEAD;

    // debug_addr_ptr[0] = 0x12341234;
    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0
    noc_index = 0;

    risc_init();

    // debug_addr_ptr[0] = 0x56785678;

    mailboxes->slave_sync.all = RUN_SYNC_MSG_ALL_SLAVES_DONE;

    // debug_addr_ptr[0] = 0xABCDABCD;

    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    // debug_addr_ptr[0] = 0xFACEFACE;

    mailboxes->go_message.signal = RUN_MSG_DONE;

    // DPRINT << "eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE " << HEX()
    //        << (uint32_t)eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE << ENDL();

    while (1) {
        // Wait...
        go_msg_t* go_msg_address = &(mailboxes->go_message);
        // DPRINT << "Waiting for go signal at " << (uint32_t)go_msg_address << ENDL();
        // debug_addr_ptr[0] = 0x1234ABCD;
        WAYPOINT("GW");

        uint8_t go_message_signal = RUN_MSG_DONE;
        while ((go_message_signal = mailboxes->go_message.signal) != RUN_MSG_GO) {
            invalidate_l1_cache();
            // While the go signal for kernel execution is not sent, check if the worker was signalled
            // to reset its launch message read pointer.
            if (go_message_signal == RUN_MSG_RESET_READ_PTR) {
                // Set the rd_ptr on workers to specified value
                mailboxes->launch_msg_rd_ptr = 0;
                uint64_t dispatch_addr = NOC_XY_ADDR(
                    NOC_X(mailboxes->go_message.master_x),
                    NOC_Y(mailboxes->go_message.master_y),
                    DISPATCH_MESSAGE_ADDR + mailboxes->go_message.dispatch_message_offset);
                mailboxes->go_message.signal = RUN_MSG_DONE;
                // Notify dispatcher that this has been done
                internal_::notify_dispatch_core_done(dispatch_addr);
            }
        }
        // DPRINT << "Done waiting for go signal" << ENDL();
        WAYPOINT("GD");

        {
            // Only include this iteration in the device profile if the launch message is valid. This is because all
            // workers get a go signal regardless of whether they're running a kernel or not. We don't want to profile
            // "invalid" iterations.
            DeviceZoneScopedMainN("ACTIVE-ERISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);

            // DPRINT << "launch msg rd ptr " << (uint32_t)launch_msg_rd_ptr << ENDL();

            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;

            flush_erisc_icache();

            enum dispatch_core_processor_masks enables =
                (enum dispatch_core_processor_masks)launch_msg_address->kernel_config.enables;

            // DPRINT << "in aerisc enables is " << HEX() << uint32_t(enables) << DEC() << ENDL();

            // Run the ERISC kernel, no kernel config buffer on active eth
            if (enables & DISPATCH_CLASS_MASK_ETH_DM0) {
                WAYPOINT("R");
                uint32_t kernel_config_base =
                    firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, DISPATCH_CLASS_ETH_DM0);
                int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM0);
                void (*kernel_address)(uint32_t) = (void (*)(uint32_t))(
                    mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index]);
                // void (*kernel_address)(uint32_t) = (void (*)(uint32_t))
                //     (kernel_config_base +
                //     mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index]);
                // DPRINT
                //     << "Kernel config base " << kernel_config_base << " index " << index << " offset "
                //     <<
                //     (uint32_t)mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index]
                //     << " kernel address " << (uint32_t)kernel_address << ENDL();
                (*kernel_address)((uint32_t)kernel_address);

                // DPRINT << "bytes sent " << erisc_info->channels[0].bytes_sent << ENDL();

                RECORD_STACK_USAGE();
                WAYPOINT("D");
            }
            // else {
            //     DPRINT << "not running the kernel" << ENDL();
            // }

            mailboxes->go_message.signal = RUN_MSG_DONE;

            // DPRINT << "kernel config mode " << HEX() << (uint32_t)launch_msg_address->kernel_config.mode << DEC()
            //        << ENDL();

            // Notify dispatcher core that it has completed
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr = NOC_XY_ADDR(
                    NOC_X(mailboxes->go_message.master_x),
                    NOC_Y(mailboxes->go_message.master_y),
                    DISPATCH_MESSAGE_ADDR + mailboxes->go_message.dispatch_message_offset);
                // dispatch addr 95056
                // DPRINT << "dispatch core: " << (uint32_t)NOC_X(mailboxes->go_message.master_x) << " " <<
                // (uint32_t)NOC_Y(mailboxes->go_message.master_y) << ENDL();
                // DPRINT << "dispatch addr " << DISPATCH_MESSAGE_ADDR + mailboxes->go_message.dispatch_message_offset
                //        << ENDL();
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                internal_::notify_dispatch_core_done(dispatch_addr);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    return 0;
}
