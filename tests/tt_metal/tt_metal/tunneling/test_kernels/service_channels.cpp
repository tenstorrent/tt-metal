// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tests/tt_metal/tt_metal/tunneling/lite_fabric.hpp"

void kernel_main() {
    uint32_t* run_signal = (uint32_t*)get_compile_time_arg_val(0);
    do {
        lite_fabric::service_lite_fabric_channels();
        invalidate_l1_cache();
    } while (*run_signal);
}
