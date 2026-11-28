// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 1 (tile-engine version; Laguna, greedy_pick.py), reader: unit w = this core (row-major index)
// covers tile row w / S of the logits shard [1, 1, R, V] (bf16 TILE) and its vocabulary tiles (w % S) * per ..;
// pushes a scaler tile (all 1.0, for the max reduce) and then those tiles one by one to compute.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t per = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr uint32_t cb_in = 0, cb_scaler = 1, T = 2048;
    const uint32_t w = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    if (w >= W) {
        return;
    }
    const uint32_t tr = w / S, j0 = (w % S) * per;
    const uint32_t j1 = j0 + per < Vt ? j0 + per : Vt;
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), T);
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < T / 4; ++i) {
        sc[i] = 0x3F803F80u;  // bf16 1.0 pairs
    }
    cb_push_back(cb_scaler, 1);
    for (uint32_t j = j0; j < j1; ++j) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(in.get_noc_addr(tr * Vt + j), get_write_ptr(cb_in), T);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
