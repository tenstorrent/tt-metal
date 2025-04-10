// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
    auto* args = reinterpret_cast<tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    args[0] = 0xc0ffee;
    args[1] = 0xdeadbeef;
    args[2] = 0xc0edbabe;
}
}  // namespace NAMESPACE
