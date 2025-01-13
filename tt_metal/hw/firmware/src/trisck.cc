// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


// This c-file's purpose is:
// 1) include the generated list of kernels
//      The files hold run_kernel() definition and inline kernel_main functions for every ckernel
//      Need to make sure no other file includes these lists since it also include global parameter definitions
// 2) instantiate global variables

#include "firmware_common.h"

#include "chlkc_list.h"

#include "tools/profiler/kernel_profiler.hpp"

#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "remote_circular_buffer_api.h"
#endif

// Global vars
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
uint32_t math_sync_tile_dst_index = 0;
uint32_t gl_alu_format_spec_reg = 0;
uint32_t op_info_offset = 0;

namespace ckernel
{
volatile tt_reg_ptr uint * regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile tt_reg_ptr uint * instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile tt_reg_ptr uint * pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile tt_reg_ptr uint * mailbox_base[4] = {
    reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_MAILBOX0_BASE), reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_MAILBOX2_BASE), reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_MAILBOX3_BASE)
};
}

void kernel_launch(uint32_t kernel_base_addr)
{
  DeviceZoneScopedMainChildN("TRISC-KERNEL");
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    ckernel::wait(KERNEL_RUN_TIME);
#endif
#else
  extern uint32_t __kernel_init_local_l1_base[];
  extern uint32_t __fw_export_end_text[];
  do_crt1((
      uint32_t tt_l1_ptr *)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base - (uint32_t)__fw_export_end_text));

#if defined(UCK_CHLKC_UNPACK)
    // Make sure DBG_FEATURE_DISABLE register is cleared before every kernel is executed
    memory_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0);
#endif
#if !defined(UCK_CHLKC_MATH) and defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    run_kernel();
#endif
}
