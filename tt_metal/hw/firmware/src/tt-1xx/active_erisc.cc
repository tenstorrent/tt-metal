// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "dev_mem_map.h"

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
}  // namespace kernel_profiler
#endif

int main() {
    configure_csr();
    WAYPOINT("I");
    do_crt1((uint32_t*)MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH);

    // put this into scratch space similar to idle erisc
    noc_bank_table_init(MEM_AERISC_BANK_TO_NOC_SCRATCH);

    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0
    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    risc_init();

    mailboxes->subordinate_sync.all = RUN_SYNC_MSG_ALL_SUBORDINATES_DONE;

    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    mailboxes->go_message.signal = RUN_MSG_DONE;

    while (1) {
        // Wait...
        WAYPOINT("GW");

        uint8_t go_message_signal = RUN_MSG_DONE;
        while ((go_message_signal = mailboxes->go_message.signal) != RUN_MSG_GO) {
            invalidate_l1_cache();
            // While the go signal for kernel execution is not sent, check if the worker was signalled
            // to reset its launch message read pointer.
            if (go_message_signal == RUN_MSG_RESET_READ_PTR) {
                // Set the rd_ptr on workers to specified value
                mailboxes->launch_msg_rd_ptr = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_message);
                mailboxes->go_message.signal = RUN_MSG_DONE;
                // Notify dispatcher that this has been done
                internal_::notify_dispatch_core_done(dispatch_addr);
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

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;

            flush_erisc_icache();

            firmware_config_init(mailboxes, ProgrammableCoreType::ACTIVE_ETH, DISPATCH_CLASS_ETH_DM0);

            enum dispatch_core_processor_masks enables =
                (enum dispatch_core_processor_masks)launch_msg_address->kernel_config.enables;

            // Run the ERISC kernel, no kernel config buffer on active eth
            if (enables & DISPATCH_CLASS_MASK_ETH_DM0) {
                WAYPOINT("R");
#ifdef ARCH_BLACKHOLE
                // #18384: This register was left dirty by eth training.
                // It is not used in dataflow api, so it can be set to 0
                // one time here instead of setting it everytime in dataflow_api.
                NOC_CMD_BUF_WRITE_REG(0 /* noc */, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE_1, 0);
#endif
                // TODO: This currently runs on second risc on active eth cores but with newer drop of syseng FW
                //  this will run on risc0
                int index = static_cast<std::underlying_type<EthProcessorTypes>::type>(EthProcessorTypes::DM0);
                uint32_t kernel_lma =
                    mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[index];
                auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                record_stack_usage(stack_free);
                WAYPOINT("D");
            }

            mailboxes->go_message.signal = RUN_MSG_DONE;

            // Notify dispatcher core that it has completed
            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_message);
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                internal_::notify_dispatch_core_done(dispatch_addr);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    return 0;
}
