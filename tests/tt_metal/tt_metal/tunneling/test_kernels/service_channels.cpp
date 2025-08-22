// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric_lite/hw/inc/kernel_api.hpp"

void kernel_main() {
    uint32_t* run_signal = (uint32_t*)get_compile_time_arg_val(0);
    do {
        fabric_lite::service_fabric_lite_channels();
        invalidate_l1_cache();
    } while (*run_signal);
}
