#pragma once

#include "fw_debug.h"
#include "ckernel_enum.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "llk_param_structs.h"

using namespace ckernel;

#ifdef UCK_CHLKC_MATH
// #include "chlkc_math_llk_args.h"
#include "chlkc_math_fidelity.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math_data_format.h"
#include "chlkc_math_tile_dims.h"
#include "loop_count.h"
#include "chlkc_math.cpp"
#include "hlk_args_struct_init.h"
void math_main(const struct hlk_args_t* hlk_args=nullptr, const void* llk_args=nullptr, const int loop_count=0);
#endif

#ifdef UCK_CHLKC_PACK
// #include "chlkc_pack_llk_args.h"
#include "chlkc_pack_data_format.h"
#include "loop_count.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math_data_format.h"
#include "chlkc_pack_tile_dims.h"
#include "chlkc_math_tile_dims.h" // needed in case of SFPU executing on pack thread 
#include "chlkc_pack.cpp"
#include "hlk_args_struct_init.h"
void pack_main(const struct hlk_args_t* hlk_args=nullptr, const void* llk_args=nullptr, const int loop_count=0);
#endif

#ifdef UCK_CHLKC_UNPACK
// #include "chlkc_unpack_llk_args.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "loop_count.h"
#include "chlkc_unpack.cpp"
#include "hlk_args_struct_init.h"
void unpack_main(const struct hlk_args_t* hlk_args=nullptr, const void* llk_args=nullptr, const int loop_count=0);
#endif



uint run_kernel() {
    
#ifdef UCK_CHLKC_MATH
    FWLOG1("run_kernel = %s", HLKC_MATH);
    regfile[p_gpr::DBG_CKID] = HLKC_MATH;
    trisc_l1_mailbox_write((HLKC_MATH << 16) | KERNEL_IN_PROGRESS);
    zeroacc();  
    math_main(&hlk_args, arg_loop_count);
#endif

#ifdef UCK_CHLKC_PACK
    FWLOG1("run_kernel = %s", HLKC_PACK);
    regfile[p_gpr::DBG_CKID] = HLKC_PACK;
    trisc_l1_mailbox_write((HLKC_PACK << 16) | KERNEL_IN_PROGRESS);
    pack_main(&hlk_args, arg_loop_count);
#endif

#ifdef UCK_CHLKC_UNPACK
    FWLOG1("run_kernel = %s", HLKC_UNPACK);
    regfile[p_gpr::DBG_CKID] = HLKC_UNPACK;
    trisc_l1_mailbox_write((HLKC_UNPACK << 16) | KERNEL_IN_PROGRESS);
    zerosrc();  
    unpack_main(&hlk_args, arg_loop_count);
#endif

return 0;

}

    
