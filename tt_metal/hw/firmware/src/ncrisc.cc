// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include <cstdint>
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
#include "circular_buffer_init.h"
#include "tdma_xmov.h"

#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
// clang-format on

uint32_t halt_stack_ptr_save;

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t *const ncrisc_run = &mailboxes->slave_sync.dm1;

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

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

extern "C" void notify_brisc_and_halt_to_iram(uint32_t status, uint32_t first_argument);

inline __attribute__((always_inline)) void notify_brisc_and_wait() {
    while (true) {
        uint8_t run_value = *ncrisc_run;
        if (run_value == RUN_SYNC_MSG_GO || run_value == RUN_SYNC_MSG_LOAD) {
            break;
        }
#if defined(ARCH_WORMHOLE)
        // Avoid hammering L1 while other cores are trying to work. Seems not to
        // be needed on Blackhole, probably because invalidate_l1_cache takes
        // time.
        asm volatile("nop; nop; nop; nop; nop");
#endif
        invalidate_l1_cache();
    }
}

inline __attribute__((always_inline)) void signal_ncrisc_completion() { *ncrisc_run = RUN_SYNC_MSG_DONE; }

#if defined(ARCH_WORMHOLE)
#define MEM_MOVER_VIEW_IRAM_BASE_ADDR (0x4 << 12)
void l1_to_ncrisc_iram_copy(uint32_t src_addr, uint16_t size, uint32_t address_offset = 0) {
    // Always copy ncrisc even if its size is 0 (save branch)...
    // Copy NCRISC firmware from L1 to local IRAM using tensix DMA
    tdma_xmov(TDMA_MOVER0, src_addr, MEM_MOVER_VIEW_IRAM_BASE_ADDR + address_offset, size, XMOV_L1_TO_L0);
}

void l1_to_ncrisc_iram_copy_wait() {
    // Wait for DMA to finish
    wait_tdma_movers_done(RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK);
}
#endif

int main(int argc, char *argv[]) {
    configure_csr();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");

    do_crt1((uint32_t tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);

    risc_init();

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        notify_brisc_and_wait();
        DeviceZoneScopedMainN("NCRISC-FW");

        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

        uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, DISPATCH_CLASS_TENSIX_DM1);
        int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1);

#if defined(ARCH_WORMHOLE)
        uint32_t ncrisc_kernel_src_address = kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index];
        l1_to_ncrisc_iram_copy(ncrisc_kernel_src_address >> 4, launch_msg->kernel_config.ncrisc_kernel_size16, 0);
#endif
        uint32_t tt_l1_ptr* cb_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
        uint32_t end_cb_index = launch_msg->kernel_config.max_local_cb_end_index;
        setup_local_cb_read_write_interfaces(cb_l1_base, 0, end_cb_index, true, true, false);

#if defined(ARCH_WORMHOLE)
        l1_to_ncrisc_iram_copy_wait();
#endif

        cb_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.remote_cb_offset);
        end_cb_index = launch_msg->kernel_config.min_remote_cb_start_index;
        // NOC argument is unused
        experimental::setup_remote_cb_interfaces<false>(cb_l1_base, end_cb_index, 0, 0, 0, 0);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

        WAYPOINT("R");

        void (*kernel_address)(uint32_t) = (void (*)(uint32_t))
            (kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index]);
#if !defined(ARCH_WORMHOLE)
        while (*ncrisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        (*kernel_address)((uint32_t)kernel_address);
#else
        // Jumping to IRAM causes bizarre behavior, so signal the brisc to reset the ncrisc to the IRAM address.
        mailboxes->ncrisc_halt.resume_addr = (uint32_t)kernel_init;
        notify_brisc_and_halt_to_iram(RUN_SYNC_MSG_WAITING_FOR_RESET, (uint32_t)kernel_address);
#endif
        RECORD_STACK_USAGE();
        WAYPOINT("D");

        signal_ncrisc_completion();
    }

    return 0;
}
