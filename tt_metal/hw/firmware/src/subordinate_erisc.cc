// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "dev_msgs.h"
#include "stream_io_map.h"
#include "firmware_common.h"
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "circular_buffer.h"
#include "core_config.h"

#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

// Required defines
// PROCESSOR_TYPE_INDEX: Which DM this firmware is running on. E.g., DM1, DM2, etc. DM0 should use active_erisc.cc or
// idle_erisc.cc. Only DM1 supported right now PROGRAMMABLE_CORE_TYPE: Active ethernet or idle ethernet
// DISPATCH_CLASS_INDEX: Dispatch class index. E.g., DISPATCH_CLASS_ETH_DM1
// PROFILER_NAME: Name of the profiler. E.g., "SUBORDINATE-IDLE-ERISC-FW"
static_assert(
    PROCESSOR_TYPE_INDEX == 1, "Only DM1 supported right now. DM0 should use active_erisc.cc or idle_erisc.cc.");
const auto k_ProgrammableCoreType = static_cast<ProgrammableCoreType>(PROGRAMMABLE_CORE_TYPE);
static_assert(
    k_ProgrammableCoreType == ProgrammableCoreType::IDLE_ETH ||
        k_ProgrammableCoreType == ProgrammableCoreType::ACTIVE_ETH,
    "This firmware is only supported for idle ethernet or active ethernet.");
static_assert(DISPATCH_CLASS_INDEX == DISPATCH_CLASS_ETH_DM1, "Only DISPATCH_CLASS_ETH_DM1 supported right now.");

#if (defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 1)

#define MAILBOX_ADDR MEM_AERISC_MAILBOX_BASE
#define INIT_LOCAL_L1_SCRATCH_BASE MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH
#define BANK_TO_NOC_SCRATCH MEM_AERISC_BANK_TO_NOC_SCRATCH
#define PROFILER_NAME "SUBORDINATE-ACTIVE-ERISC1-FW"

#elif (defined(COMPILE_FOR_IDLE_ERISC) && COMPILE_FOR_IDLE_ERISC == 1)

#define MAILBOX_ADDR MEM_IERISC_MAILBOX_BASE
#define INIT_LOCAL_L1_SCRATCH_BASE MEM_SUBORDINATE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH
#define BANK_TO_NOC_SCRATCH MEM_IERISC_BANK_TO_NOC_SCRATCH
#define PROFILER_NAME "SUBORDINATE-IDLE-ERISC1-FW"

#else
#error "Invalid compile target. Must be COMPILE_FOR_AERISC=1 or COMPILE_FOR_IDLE_ERISC=1"
#endif

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MAILBOX_ADDR);
volatile tt_l1_ptr uint8_t* const subordinate_erisc_run = &mailboxes->subordinate_sync.dm1;

uint8_t noc_index = 0;  // TODO: hardcoding needed for profiler

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

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

inline __attribute__((always_inline)) void signal_subordinate_erisc_completion() {
    *subordinate_erisc_run = RUN_SYNC_MSG_DONE;
}

int main() {
    configure_csr();
    WAYPOINT("I");
    do_crt1((uint32_t*)INIT_LOCAL_L1_SCRATCH_BASE);

    noc_bank_table_init(BANK_TO_NOC_SCRATCH);

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    risc_init();
    signal_subordinate_erisc_completion();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        while (*subordinate_erisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        DeviceZoneScopedMainN(PROFILER_NAME);

        flush_erisc_icache();

        uint32_t kernel_config_base = firmware_config_init(mailboxes, k_ProgrammableCoreType, DISPATCH_CLASS_INDEX);
        my_relative_x_ =
            my_logical_x_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_x;
        my_relative_y_ =
            my_logical_y_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_y;

        WAYPOINT("R");
        uint32_t kernel_lma =
            kernel_config_base +
            mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[PROCESSOR_TYPE_INDEX];
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        record_stack_usage(stack_free);
        WAYPOINT("D");

        signal_subordinate_erisc_completion();
    }

    return 0;
}
