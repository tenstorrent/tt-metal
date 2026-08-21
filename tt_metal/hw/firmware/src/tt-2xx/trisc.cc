// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "ckernel.h"
#include "internal/firmware_common.h"
#include "risc_common.h"
#include <tensix.h>
#include <cstdint>
#include "hostdev/dev_msgs.h"

#include "tools/profiler/kernel_profiler.hpp"

#include "internal/debug/fw_debug.h"
#include "internal/hw_thread.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "api/debug/ring_buffer.h"
#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h"
#endif
#include "tt-metalium/circular_buffer_constants.h"
#include "api/kernel_thread_globals.h"
#include "llk_bfd_alloc.h"  // ckernel::trisc::BfdAllocatorState (definition of bfd_state below)

// clang-format on

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
thread_local std::uint32_t wIndex __attribute__((used));
thread_local std::uint32_t stackSize __attribute__((used));
thread_local std::uint32_t sums[SUM_COUNT] __attribute__((used));
thread_local std::uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

thread_local std::uint32_t hw_thread_idx __attribute__((used));
thread_local std::uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
thread_local std::uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
thread_local std::uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
thread_local std::uint32_t rta_count __attribute__((used));
thread_local std::uint32_t crta_count __attribute__((used));
#endif

std::uint8_t my_logical_x_ __attribute__((used));
std::uint8_t my_logical_y_ __attribute__((used));
std::uint8_t my_relative_x_ __attribute__((used));
std::uint8_t my_relative_y_ __attribute__((used));

#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
#if defined(UCK_CHLKC_PACK)
thread_local LocalDFBInterface g_dfb_interface[dfb::MAX_ACTIVE_DFBS_PACK] __attribute__((used));
thread_local std::uint8_t g_dfb_logical_to_compact[dfb::NUM_DFBS] __attribute__((used));
#else
thread_local LocalDFBInterface g_dfb_interface[dfb::NUM_DFBS] __attribute__((used));
#endif
#endif
// Defined for all TRISC types (including math) so kernels that include dataflow_buffer headers link cleanly.
// For math TRISC, setup_local_dfb_interfaces is not called, so this stays 0; dfb_ensure_ready
// returns immediately for any DFB math TRISC is not a participant in (expected_signal == 0).
thread_local uintptr_t g_dfb_config_base_addr __attribute__((used));

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

// Flip between 0 and 1 to keep state between kernel calls
thread_local std::uint32_t cfg_state_id __attribute__((used)) = 0;
thread_local std::uint32_t op_info_offset __attribute__((used)) = 0;
namespace trisc {
// Flip between 0 and 1 to keep dest pointer between kernel calls
thread_local std::uint32_t dest_register_offset __attribute__((used)) = 0;
// BFD id allocator state (Quasar). Defined for all TRISC images alongside dest_register_offset;
// thread_local so the host-threaded emulation gives each TRISC its own allocator (tt-llk#1678).
// do_thread_crt1 clears this per-thread state before first use; bfd_alloc then sets next to the
// partition base and current[] to BFD_ID_INVALID on the first allocation.
thread_local BfdAllocatorState bfd_state __attribute__((used)) = {.next = 0, .current = {}, .initialized = false};
}  // namespace trisc

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE);
}  // namespace ckernel

using namespace ckernel;

void init_sync_registers() {
    // TODO: check if this is needed with transition to DFBs
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

inline void enable_cc_stack() {
#if defined(UCK_CHLKC_MATH)
    constexpr std::uint32_t SFPENCC_IMM12_BOTH = 3;
    constexpr std::uint32_t SFPENCC_MOD1_EI_RI = 10;
    TTI_SFPENCC(SFPENCC_IMM12_BOTH, SFPENCC_MOD1_EI_RI);  // Enable all the SFPU lanes
#endif
}

extern "C" std::uint32_t _start1() {
    configure_csr();
    // Raw read: hw_thread_idx has not been filled yet, and do_thread_crt1() below zeroes the .tbss
    // it lives in, so caching it any earlier would just be discarded.
    std::uint32_t hartid = internal_::read_hw_thread_idx();
    std::uint32_t neo_id = internal_::get_neo_id();
    std::uint32_t trisc_id = internal_::get_trisc_id();
    volatile tt_l1_ptr std::uint8_t* const trisc_run =
        &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE))
             ->subordinate_sync.map[hartid];  // first entry is for NCRISC

    if (neo_id == 0) {
        extern std::uint32_t __ldm_data_start[];
        do_crt1(__ldm_data_start);
        (*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[MaxDMProcessorsPerCoreType + trisc_id] =
            SHARED_GLOBALS_READY_GO;
    }
    extern std::uint32_t __ldm_tdata_init[];
    do_thread_crt1(__ldm_tdata_init);
    // .tbss has been zeroed: cache this thread's hw index.
    internal_::init_hw_thread_idx();
    // DEVICE_PRINT and WAYPOINT index their per-thread slots via get_hw_thread_idx(), so they have to
    // come after the cache is filled.
    DEVICE_PRINT("hartid: {}\n", hartid);
    WAYPOINT("I");

    while ((*GET_MAILBOX_ADDRESS_DEV(fw_shared_globals_ready))[MaxDMProcessorsPerCoreType + trisc_id] !=
           SHARED_GLOBALS_READY_GO) {
    }
    // Initialize GPRs to all 0s
#pragma GCC unroll 0
    for (int i = 0; i < 64; i++) {
        regfile[i] = 0;
    }
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;
    *trisc_run = RUN_SYNC_MSG_DONE;
    setup_isr_csrs();
    enable_cc_stack();
    DeviceProfilerInit();
    DPRINT("TRISC-FW: initialized\n");
    while (1) {
        WAYPOINT("W");
        while (*trisc_run != RUN_SYNC_MSG_GO) {
            if constexpr (COMPILE_FOR_TRISC == 0) {
                if (*trisc_run == RUN_SYNC_MSG_INIT_SYNC_REGISTERS) {
                    init_sync_registers();
                    *trisc_run = RUN_SYNC_MSG_DONE;
                }
            }
        }
        // Scope the FW zone so finish_profiler flushes this TRISC's DEVICE_BUFFER_END_INDEX before
        // RUN_SYNC_MSG_DONE is signaled.
        {
            DeviceZoneScopedMainN("TRISC-FW");
            std::uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

            uintptr_t kernel_config_base = launch_msg->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];

#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
            std::uint32_t tt_l1_ptr* dfb_l1_base =
                (std::uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
            std::uint32_t num_local_dfbs = launch_msg->kernel_config.local_cb_mask;
#if defined(UCK_CHLKC_PACK)
            const DfbPackerRemapperRange packer_rmp = setup_local_dfb_interfaces(dfb_l1_base, num_local_dfbs);
#else
            setup_local_dfb_interfaces(dfb_l1_base, num_local_dfbs);
#endif
#endif

            // TODO: Remove MEM_L1_UNCACHED_BASE here and invalidate cache lines when PR #38124 is merged
            rta_l1_base = (std::uint32_t tt_l1_ptr*)(kernel_config_base +
                                                     launch_msg->kernel_config.rta_offset[hartid].rta_offset +
                                                     MEM_L1_UNCACHED_BASE);
            crta_l1_base = (std::uint32_t tt_l1_ptr*)(kernel_config_base +
                                                      launch_msg->kernel_config.rta_offset[hartid].crta_offset +
                                                      MEM_L1_UNCACHED_BASE);
            sem_l1_base[ProgrammableCoreType::TENSIX] =
                (std::uint32_t tt_l1_ptr*)(kernel_config_base +
                                           launch_msg->kernel_config.sem_offset[ProgrammableCoreType::TENSIX]);
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
            // Initialize RTA count from L1 memory
            // Set to 0 if: 1. offset is sentinel (no args set)
            //              2. memory contains known garbage pattern 0xBEEF#### (uninitialized slot)
            if (launch_msg->kernel_config.rta_offset[hartid].rta_offset == RTA_CRTA_NO_ARGS_SENTINEL ||
                ((rta_l1_base[0] & 0xFFFF0000) == WATCHER_RTA_UNSET_PATTERN)) {
                rta_count = 0;
            } else {
                rta_count = rta_l1_base[0];
                rta_l1_base += 1;  // Skip count word
            }

            // Initialize CRTA count from L1 memory
            // Set to 0 if: 1. offset is sentinel (no common args set)
            //              2. memory contains known garbage pattern 0xBEEF#### (unicast mode, kernel has no CRTAs)
            if (launch_msg->kernel_config.rta_offset[hartid].crta_offset == RTA_CRTA_NO_ARGS_SENTINEL ||
                ((crta_l1_base[0] & 0xFFFF0000) == WATCHER_RTA_UNSET_PATTERN)) {
                crta_count = 0;
            } else {
                crta_count = crta_l1_base[0];
                crta_l1_base += 1;  // Skip count word
            }
#endif

            my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

            WAYPOINT("R");
            uintptr_t kernel_lma =
                (kernel_config_base +
                 launch_msg->kernel_config.kernel_text_offset[hartid]);  // TODO verify if depends on kernel
            auto stack_free = reinterpret_cast<std::uint32_t (*)()>(kernel_lma)();
            record_stack_usage(stack_free);
            WAYPOINT("D");
            DEVICE_PRINT_KERNEL_FINISHED();

#if defined(UCK_CHLKC_PACK)
            // Tear down packer remapper pairs programmed this launch so they cannot leak into the next.
            dfb_clear_packer_remapper_window(packer_rmp.lo, packer_rmp.hi);
#endif

            // Signal completion
            DPRINT("SIGNALING COMPLETION {:x}\n", (std::uint32_t)*trisc_run);
            tensix_sync();
        }
        *trisc_run = RUN_SYNC_MSG_DONE;
        DPRINT("COMPLETION SIGNED OFF {:x}\n", (std::uint32_t)*trisc_run);
    }
}
