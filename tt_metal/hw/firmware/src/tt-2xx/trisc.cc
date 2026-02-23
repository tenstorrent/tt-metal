// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "ckernel.h"
#include "internal/firmware_common.h"
#include "risc_common.h"
#include <tensix.h>
#include "hostdev/dev_msgs.h"

#include "tools/profiler/kernel_profiler.hpp"

#include "internal/debug/fw_debug.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "api/debug/ring_buffer.h"
#if !defined(UCK_CHLKC_MATH)
#include "internal/circular_buffer_interface.h"
#include "internal/circular_buffer_init.h"
#endif
#include "internal/dataflow_buffer_init.h"
#include "tt-metalium/circular_buffer_constants.h"
// clang-format on

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

thread_local uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
thread_local uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));

uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

thread_local ::experimental::LocalDFBInterface g_dfb_interface[experimental::NUM_DFBS] __attribute__((used));

namespace ckernel {

// Transition shim
#if defined(__PTR_CONST)
#define PTR_CONST const
#else
#define PTR_CONST
#endif
// volatile tt_reg_ptr uint* const reg_base = reinterpret_cast<volatile uint*>(0xFFB10000);
// volatile tt_reg_ptr uint* const pc_buf_base = reinterpret_cast<volatile uint*>(PC_BUF_BASE);
// volatile tt_reg_ptr uint* const regfile = reinterpret_cast<volatile uint*>(REGFILE_BASE);
#undef PTR_CONST

uint32_t cfg_state_id __attribute__((used)) = 0;    // Flip between 0 and 1 to keep state between kernel calls
uint32_t dest_offset_id __attribute__((used)) = 0;  // Flip between 0 and 1 to keep dest pointer between kernel calls

uint32_t op_info_offset __attribute__((used)) = 0;

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE);
}  // namespace ckernel

#if !defined(UCK_CHLKC_MATH)
uint32_t tt_l1_ptr* cb_l1_base __attribute__((used));
#endif

#if defined(UCK_CHLKC_UNPACK)
constexpr bool cb_init_read = true;
#else
constexpr bool cb_init_read = false;
#endif
#if defined(UCK_CHLKC_PACK)
constexpr bool cb_init_write = true;
#else
constexpr bool cb_init_write = false;
#endif

using namespace ckernel;

void init_sync_registers() {
    // TODO: check if this is needed with tranistion to DFBs
    // https://github.com/tenstorrent/tt-metal/issues/36889
    // volatile tt_reg_ptr uint* tiles_received_ptr;
    // volatile tt_reg_ptr uint* tiles_acked_ptr;
    // for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
    //     tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    //     tiles_received_ptr[0] = 0;
    //     tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    //     tiles_acked_ptr[0] = 0;
    // }
}

extern "C" uint32_t _start1() {
    configure_csr();
    std::uint64_t hartid;
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    std::uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    hartid = 8 + 4 * neo_id + trisc_id;  // after 8 DM cores
    DPRINT << "hartid: " << hartid << ENDL();
    volatile tt_l1_ptr uint8_t* const trisc_run = &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE))
                                                       ->subordinate_sync.map[hartid];  // first entry is for NCRISC
    WAYPOINT("I");

    extern uint32_t __ldm_data_start[];
    do_crt1(__ldm_data_start);
    extern uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    // Initialize GPRs to all 0s
#pragma GCC unroll 0
    for (int i = 0; i < 64; i++) {
        regfile[i] = 0;
    }
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    *trisc_run = RUN_SYNC_MSG_DONE;

    DeviceProfilerInit();
    DPRINT << "TRISC-FW: initialized" << ENDL();
    while (1) {
        WAYPOINT("W");
        while (*trisc_run != RUN_SYNC_MSG_GO) {
            if constexpr (COMPILE_FOR_TRISC == 0) {
                if (*trisc_run == RUN_SYNC_MSG_INIT_SYNC_REGISTERS) {
                    init_sync_registers();
                    *trisc_run = RUN_SYNC_MSG_DONE;
                }
            }
            invalidate_l1_cache();
        }
        DeviceZoneScopedMainN("TRISC-FW");
        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

        uint32_t kernel_config_base = launch_msg->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];

#if !defined(UCK_CHLKC_MATH)
        // uint32_t tt_l1_ptr* cb_l1_base =
        //     (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
        // uint32_t local_cb_mask = launch_msg->kernel_config.local_cb_mask;
        // setup_local_cb_read_write_interfaces<cb_init_read, cb_init_write, cb_init_write>(cb_l1_base, 0,
        // local_cb_mask);

        // cb_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.remote_cb_offset);
        // uint32_t end_cb_index = launch_msg->kernel_config.min_remote_cb_start_index;
        // // NOC argument is unused
        // experimental::setup_remote_cb_interfaces<false>(cb_l1_base, end_cb_index, 0, 0, 0, 0);
#endif

        uint32_t tt_l1_ptr* dfb_l1_base = (uint32_t tt_l1_ptr*)(MEM_L1_UNCACHED_BASE + kernel_config_base +
                                                                launch_msg->kernel_config.local_cb_offset);
        uint32_t num_local_dfbs = launch_msg->kernel_config.local_cb_mask;
        experimental::setup_local_dfb_interfaces(dfb_l1_base, num_local_dfbs);

        rta_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.rta_offset[hartid].rta_offset);
        crta_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.rta_offset[hartid].crta_offset);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

        WAYPOINT("R");
        uint32_t kernel_lma =
            (kernel_config_base +
             launch_msg->kernel_config.kernel_text_offset[hartid]);  // TODO verify if depends on kernel
        asm("FENCE.i");
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        record_stack_usage(stack_free);
        WAYPOINT("D");

        // Signal completion
        DPRINT << "SIGNALING COMPLETION " << HEX() << (uint32_t)*trisc_run << DEC() << ENDL();
        tensix_sync();
        *trisc_run = RUN_SYNC_MSG_DONE;
        DPRINT << "COMPLETION SIGNED OFF" << HEX() << (uint32_t)*trisc_run << DEC() << ENDL();
    }
}
