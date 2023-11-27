// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    cb_reserve_back(cb_id_in0, num_units);
    cb_push_back(cb_id_in0, num_units);
}
