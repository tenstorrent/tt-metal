#pragma once

#include "fw_debug.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "llk_param_structs.h"

using namespace ckernel;


#ifdef UCK_CHLKC_MATH
// #include "chlkc_math_llk_args.h"
#include "chlkc_math_fidelity.h"
#include "chlkc_math_approx_mode.h"
#include "chlkc_math.cpp"
#endif

#ifdef UCK_CHLKC_PACK
// #include "chlkc_pack_llk_args.h"
#include "chlkc_pack_data_format.h"
#include "chlkc_pack.cpp"
#endif

#ifdef UCK_CHLKC_UNPACK
// #include "chlkc_unpack_llk_args.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack.cpp"
#endif

uint run_kernel() {

#ifdef UCK_CHLKC_MATH
    FWLOG1("run_kernel = %s", HLKC_MATH);
    regfile[p_gpr::DBG_CKID] = HLKC_MATH;
    trisc_l1_mailbox_write((HLKC_MATH << 16) | KERNEL_IN_PROGRESS);
    zeroacc();
    chlkc_math::math_main();
#endif

#ifdef UCK_CHLKC_PACK
    FWLOG1("run_kernel = %s", HLKC_PACK);
    regfile[p_gpr::DBG_CKID] = HLKC_PACK;
    trisc_l1_mailbox_write((HLKC_PACK << 16) | KERNEL_IN_PROGRESS);
    chlkc_pack::pack_main();
#endif

#ifdef UCK_CHLKC_UNPACK
    FWLOG1("run_kernel = %s", HLKC_UNPACK);
    regfile[p_gpr::DBG_CKID] = HLKC_UNPACK;
    trisc_l1_mailbox_write((HLKC_UNPACK << 16) | KERNEL_IN_PROGRESS);
    zerosrc();
    chlkc_unpack::unpack_main();
#endif

return 0;

}
