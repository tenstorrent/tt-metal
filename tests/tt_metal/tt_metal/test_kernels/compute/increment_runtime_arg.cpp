// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {

    // Tests get_arg_val API
    uint32_t arg_a  = get_arg_val<uint32_t>(0);
    uint32_t arg_b = get_arg_val<uint32_t>(1);
    uint32_t common_arg_a = get_common_arg_val<uint32_t>(0);
    uint32_t common_arg_b = get_common_arg_val<uint32_t>(1);
    uint32_t common_arg_c = get_common_arg_val<uint32_t>(2);
    uint32_t common_arg_d = get_common_arg_val<uint32_t>(3);

    // Need pointer as well to modify arg address to test in host
    volatile tt_l1_ptr std::uint32_t* arg_a_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE);
    volatile tt_l1_ptr std::uint32_t* arg_b_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + 4);
    volatile tt_l1_ptr std::uint32_t* common_arg_a_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + COMMON_RT_ARGS_OFFSET);
    volatile tt_l1_ptr std::uint32_t* common_arg_b_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + COMMON_RT_ARGS_OFFSET + 4);
    volatile tt_l1_ptr std::uint32_t* common_arg_c_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + COMMON_RT_ARGS_OFFSET + 8);
    volatile tt_l1_ptr std::uint32_t* common_arg_d_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + COMMON_RT_ARGS_OFFSET + 12);

    UNPACK(arg_a_ptr[0] = arg_a + 87);
    UNPACK(arg_b_ptr[0] = arg_b + 216);
    UNPACK(common_arg_a_ptr[0] = common_arg_a + 123);
    UNPACK(common_arg_b_ptr[0] = common_arg_b + 234);
    UNPACK(common_arg_c_ptr[0] = common_arg_c + 345);
    UNPACK(common_arg_d_ptr[0] = common_arg_d + 456);

}
}
