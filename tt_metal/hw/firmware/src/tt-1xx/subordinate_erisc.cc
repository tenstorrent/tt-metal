// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "hostdev/dev_msgs.h"
#include "stream_io_map.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "core_config.h"

#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
// clang-format on

// Required defines
// PROCESSOR_INDEX: Which DM this firmware is running on. E.g., DM1, DM2, etc. DM0 should use active_erisc.cc or
// idle_erisc.cc. Only DM1 supported right now PROGRAMMABLE_CORE_TYPE: Active ethernet or idle ethernet
// PROFILER_NAME: Name of the profiler. E.g., "SUBORDINATE-IDLE-ERISC-FW"
static_assert(PROCESSOR_INDEX == 1, "Only DM1 supported right now. DM0 should use active_erisc.cc or idle_erisc.cc.");
const auto k_ProgrammableCoreType = static_cast<ProgrammableCoreType>(PROGRAMMABLE_CORE_TYPE);
static_assert(
    k_ProgrammableCoreType == ProgrammableCoreType::IDLE_ETH ||
        k_ProgrammableCoreType == ProgrammableCoreType::ACTIVE_ETH,
    "This firmware is only supported for idle ethernet or active ethernet.");

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
volatile tt_l1_ptr uint8_t* const subordinate_erisc_run = mailboxes->subordinate_sync.map;

// Note: This is just for the firmware
// The kernel defines NOC_MODE and NOC_INDEX
uint8_t noc_index = 0;

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
uint32_t traceCount __attribute__((used));
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
#if defined(ENABLE_2_ERISC_MODE)
    if (k_ProgrammableCoreType == ProgrammableCoreType::ACTIVE_ETH) {
        while (true) {
            // Wait for ERISC0 to signal that it has saved its state to L1
            if (mailboxes->ncrisc_halt.stack_save != 0) {
                // Trigger a soft reset of ERISC0. Wait 100 cycles, and then deassert
                WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_BRISC);
                riscv_wait(100);
                WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NONE);

                break;
            }
            invalidate_l1_cache();
        }
    }
#endif

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        while (*subordinate_erisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        DeviceZoneScopedMainN(PROFILER_NAME);

        flush_erisc_icache();

        uint32_t kernel_config_base = firmware_config_init(mailboxes, k_ProgrammableCoreType, PROCESSOR_INDEX);
        my_relative_x_ =
            my_logical_x_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_x;
        my_relative_y_ =
            my_logical_y_ - mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.sub_device_origin_y;

        WAYPOINT("R");
        uint32_t kernel_lma =
            kernel_config_base +
            mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.kernel_text_offset[PROCESSOR_INDEX];
#if defined(COMPILE_FOR_AERISC)
        // Stack usage is not implemented yet for subordinate active eth (active_erisck.cc)
        reinterpret_cast<void (*)()>(kernel_lma)();
#else
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        record_stack_usage(stack_free);
#endif
        WAYPOINT("D");

        signal_subordinate_erisc_completion();
    }

    return 0;
}
