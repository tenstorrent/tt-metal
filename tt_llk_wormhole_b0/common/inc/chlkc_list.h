/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include "fw_debug.h"
#include "ckernel_enum.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"

using namespace ckernel;

// This function is called before each kernel (unpack, math or pack) is called.
// It is defined in hlk_defs.cpp
void setup_kernel();

#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK)
    #if defined(TT_HLK_ALWAYS_INLINE)
        #undef TT_HLK_ALWAYS_INLINE
    #endif

    #define TT_HLK_ALWAYS_INLINE inline __attribute__ ((always_inline))
#endif

#ifdef UCK_CHLKC_MATH
#include "chlkc_math_fidelity.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math_data_format.h"
#include "chlkc_math_tile_dims.h"
#include "chlkc_pack_data_format.h"
#include "chlkc_pack_tile_dims.h"
#include "loop_count.h"
#include "hlk_compile_time_constants.h"
#include "hlk_args_constexpr.h"
#include "hlk.cpp"
#include "hlk_args_struct_init.h"
#include "hlk_defs_wormhole_b0.h"
#endif

#ifdef UCK_CHLKC_PACK
#include "chlkc_pack_data_format.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math_data_format.h"
#include "chlkc_pack_tile_dims.h"
#include "chlkc_math_tile_dims.h"
#include "loop_count.h"
#include "hlk_compile_time_constants.h"
#include "hlk_args_constexpr.h"
#include "hlk.cpp"
#include "hlk_args_struct_init.h"
#include "hlk_defs_wormhole_b0.h"
#endif

#ifdef UCK_CHLKC_UNPACK
#include "chlkc_unpack_data_format.h"
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_pack_tile_dims.h"
#include "loop_count.h"
#include "hlk_compile_time_constants.h"
#include "hlk_args_constexpr.h"
#include "hlk.cpp"
#include "hlk_args_struct_init.h"
#include "hlk_defs_wormhole_b0.h"
#endif

#include "llk_param_structs.h"

uint run_kernel() {

#ifdef UCK_CHLKC_MATH
    FWLOG1("run_kernel = %s", HLKC_MATH);
    regfile[p_gpr::DBG_CKID] = HLKC_MATH;
    trisc_l1_mailbox_write((HLKC_MATH << 16) | KERNEL_IN_PROGRESS);
    zeroacc();
    setup_kernel();
    hlk_process_all_inputs<hlk_args_t>(nullptr, &hlk_args, arg_loop_count);
#endif

#ifdef UCK_CHLKC_PACK
    FWLOG1("run_kernel = %s", HLKC_PACK);
    regfile[p_gpr::DBG_CKID] = HLKC_PACK;
    trisc_l1_mailbox_write((HLKC_PACK << 16) | KERNEL_IN_PROGRESS);
    setup_kernel();
    hlk_process_all_inputs<hlk_args_t>(nullptr, &hlk_args, arg_loop_count);
#endif

#ifdef UCK_CHLKC_UNPACK
    FWLOG1("run_kernel = %s", HLKC_UNPACK);
    regfile[p_gpr::DBG_CKID] = HLKC_UNPACK;
    trisc_l1_mailbox_write((HLKC_UNPACK << 16) | KERNEL_IN_PROGRESS);
    zerosrc();
    setup_kernel();
    hlk_process_all_inputs<hlk_args_t>(nullptr, &hlk_args, arg_loop_count);
#endif

return 0;
}
