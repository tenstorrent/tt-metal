// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "ckernel.h"
#include "firmware_common.h"
#include "risc_common.h"
#include <tensix.h>
#include "dev_msgs.h"

#include "tools/profiler/kernel_profiler.hpp"

#include "debug/fw_debug.h"
#include "debug/waypoint.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
#include "circular_buffer.h"
// clang-format on

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}
#endif

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));

namespace ckernel {

enum class ttRiscCores : std::uint32_t { Unpack = 0, Math = 1, Pack = 2, Brisc = 3, Nrisc = 4 };

volatile tt_reg_ptr uint *reg_base = reinterpret_cast<volatile uint *>(0xFFB10000);
volatile tt_reg_ptr uint *pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile tt_reg_ptr uint *regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile tt_reg_ptr uint *instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
tt_reg_ptr uint *regmem = reinterpret_cast<tt_reg_ptr uint *>(REGFILE_BASE);

uint32_t cfg_state_id __attribute__((used)) = 0;    // Flip between 0 and 1 to keep state between kernel calls
uint32_t dest_offset_id __attribute__((used)) = 0;  // Flip between 0 and 1 to keep dest pointer between kernel calls

uint32_t op_info_offset __attribute__((used)) = 0;

const uint8_t thread_id = COMPILE_FOR_TRISC;

#define GET_TRISC_RUN_EVAL(x, t) x##t
#define GET_TRISC_RUN(x, t) GET_TRISC_RUN_EVAL(x, t)
volatile tt_l1_ptr uint8_t *const trisc_run =
    &GET_TRISC_RUN(((tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE))->slave_sync.trisc, COMPILE_FOR_TRISC);
tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);
}  // namespace ckernel

volatile tt_l1_ptr uint32_t l1_buffer[16] __attribute__((section("l1_data"))) __attribute__((aligned(16)))
__attribute__((used));

#if !defined(UCK_CHLKC_MATH)
uint32_t tt_l1_ptr *cb_l1_base __attribute__((used));
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));
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

int main(int argc, char *argv[]) {
    conditionally_disable_l1_cache();
    DIRTY_STACK_MEMORY();
    WAYPOINT("I");

    uint tt_l1_ptr *local_l1_start_addr =
        (uint tt_l1_ptr *)PREPROCESSOR_EXPAND(MEM_TRISC, COMPILE_FOR_TRISC, _INIT_LOCAL_L1_BASE);
    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint *)__ldm_data_start, local_l1_start_addr, num_words);

    // Initialize GPRs to all 0s
#pragma GCC unroll 0
    for (int i = 0; i < 64; i++) regfile[i] = 0;

    // Init L1 buffer with 1.0f (used for reduce max)
    union {
        float f;
        uint32_t u;
    } f2u = {.f = 1.0f};

    // Save a little code space.  GCC fails to remove the loop variable so loop with a ptr
#pragma GCC unroll 0
    for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer

    reset_cfg_state_id();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        WAYPOINT("W");
        while (*trisc_run != RUN_SYNC_MSG_GO);
        DeviceZoneScopedMainN("TRISC-FW");

        uint32_t kernel_config_base = mailboxes->launch.kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];

#if !defined(UCK_CHLKC_MATH)
        uint32_t tt_l1_ptr *cb_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch.kernel_config.cb_offset);
        setup_cb_read_write_interfaces(cb_l1_base, 0, max_cb_index, cb_init_read, cb_init_write, cb_init_write);
#endif

        rta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_TENSIX_COMPUTE].rta_offset);
        crta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_TENSIX_COMPUTE].crta_offset);

        WAYPOINT("R");
        kernel_init();
        RECORD_STACK_USAGE();
        WAYPOINT("D");

        // Signal completion
        tensix_sync();
        *trisc_run = RUN_SYNC_MSG_DONE;
    }
}
