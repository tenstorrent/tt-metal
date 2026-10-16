// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC: cb_w is pinned to this core's slot of the L1 staging tensor that an earlier launch filled
// (proj_prefetch_reader.cpp), so the weights are already in place: publish every tile to compute at once.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    cb_reserve_back(cb_w, tiles);
    cb_push_back(cb_w, tiles);
}
