// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


// This c-file's purpose is:
// 1) include the generated list of kernels
//      The files hold run_kernel() definition and inline kernel_main functions for every ckernel
//      Need to make sure no other file includes these lists since it also include global parameter definitions
// 2) instantiate global variables

#include "ckernel_globals.h"

#include "chlkc_list.h"

#include "tools/profiler/kernel_profiler.hpp"

// Global vars
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
uint32_t math_sync_tile_dst_index = 0;
uint32_t gl_alu_format_spec_reg = 0;

// volatile uint32_t tt_l1_ptr l1_buffer[16] __attribute__ ((section (".text#"))) __attribute__ ((aligned (16)));
// ckernel::operand_u operands[24] = {0};
// ckernel::output_u outputs[16] = {0};

namespace ckernel
{
volatile tt_reg_ptr uint * const regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile tt_reg_ptr uint * const instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile tt_reg_ptr uint * const pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
}

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

void kernel_launch()
{
    tt_l1_ptr uint *local_l1_start_addr = (tt_l1_ptr uint *)PREPROCESSOR_EXPAND(MEM_TRISC, COMPILE_FOR_TRISC, _INIT_LOCAL_L1_BASE);
    firmware_kernel_common_init(local_l1_start_addr);

    run_kernel();
}
