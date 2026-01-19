// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include <cstdint>
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "hostdev/dev_msgs.h"
#include "stream_io_map.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "internal/circular_buffer_init.h"
#include "tdma_xmov.h"

#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
// clang-format on

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t* const ncrisc_run = mailboxes->subordinate_sync.map;

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
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

// These arrays are used to store the worker logical to virtual coordinate mapping
uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)] __attribute__((used));
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

#ifdef ARCH_WORMHOLE
extern "C" uint32_t wh_iram_trampoline(uint32_t status, uint32_t first_argument);
#endif

inline __attribute__((always_inline)) void wait_for_brisc_notification() {
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

int main(int argc, char* argv[]) {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t tt_l1_ptr*)MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    risc_init();

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    signal_ncrisc_completion();

    DeviceProfilerInit();
    while (1) {
        WAYPOINT("W");
        wait_for_brisc_notification();
        DeviceZoneScopedMainN("NCRISC-FW");

        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

        uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, PROCESSOR_INDEX);
        int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1);

        uint32_t kernel_lma = kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index];
#if defined(ARCH_WORMHOLE)
        static_assert(MEM_NCRISC_KERNEL_BASE == MEM_NCRISC_IRAM_BASE, "NCRISC kernel vma mismatch");
        l1_to_ncrisc_iram_copy(kernel_lma >> 4, launch_msg->kernel_config.ncrisc_kernel_size16, 0);
#endif
        uint32_t tt_l1_ptr* cb_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
        uint32_t local_cb_mask = launch_msg->kernel_config.local_cb_mask;
        setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);

#if defined(ARCH_WORMHOLE)
        l1_to_ncrisc_iram_copy_wait();
#endif

        cb_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.remote_cb_offset);
        uint32_t end_cb_index = launch_msg->kernel_config.min_remote_cb_start_index;
        // NOC argument is unused
        experimental::setup_remote_cb_interfaces<false>(cb_l1_base, end_cb_index, 0, 0, 0, 0);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

        WAYPOINT("R");

#if defined(ARCH_WORMHOLE)
        // Jumping to IRAM causes bizarre behavior, so signal the
        // brisc to reset the ncrisc to the IRAM address
        uint32_t kernel_vma = MEM_NCRISC_KERNEL_BASE;
        mailboxes->ncrisc_halt.resume_addr = kernel_vma;
        auto stack_free = wh_iram_trampoline(RUN_SYNC_MSG_WAITING_FOR_RESET, kernel_lma - kernel_vma);
#else
        while (*ncrisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
#endif
        record_stack_usage(stack_free);
        WAYPOINT("D");

        signal_ncrisc_completion();
#if defined(ARCH_WORMHOLE)
        // Ensure branch predictor will only ever predict into L1. Otherwise, the branch predictor may predict an IRAM
        // address, which can cause an instruction to be fetched from IRAM while the mover is writing to IRAM, which can
        // cause corruption.  See
        // https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/InstructionRAM.md
        // for more details.
        // This loop unrolls to 54 instructions, taking 110 cycles (assuming all branches are mispredicted).
        asm volatile(
            ".rept 13\n"
            "bne x0, x0, .\n"
            "bne x0, x0, .\n"
            "nop\n"
            "nop\n"
            ".endr\n"
            "bne x0, x0, .\n"
            "bne x0, x0, .");
#endif
    }

    return 0;
}
