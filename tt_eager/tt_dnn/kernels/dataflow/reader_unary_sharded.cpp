// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    //uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    //uint32_t tile_size = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < num_tiles_per_core; ++ i) {
        cb_push_back(cb_id_in0, onetile);
        DPRINT << "reader Unary_done " << ENDL();
    }
}
