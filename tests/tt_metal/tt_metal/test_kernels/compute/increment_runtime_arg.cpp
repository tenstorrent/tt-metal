// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {

    // Tests get_arg_val API
    uint32_t arg_a  = get_arg_val<uint32_t>(0);
    uint32_t arg_b = get_arg_val<uint32_t>(1);


    // Need pointer as well to modify arg address to test in host
    volatile tt_l1_ptr std::uint32_t* arg_a_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE);
    volatile tt_l1_ptr std::uint32_t* arg_b_ptr = (volatile tt_l1_ptr uint32_t*)(TRISC_L1_ARG_BASE + 4);


    UNPACK(arg_a_ptr[0] = arg_a + 87);
    UNPACK(arg_b_ptr[0] = arg_b + 216);

}
}
