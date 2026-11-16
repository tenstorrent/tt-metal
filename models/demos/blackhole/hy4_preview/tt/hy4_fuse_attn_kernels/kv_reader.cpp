// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B20 fused KV append (quantise + pack + row write), reader. Core b of 4 owns the 128-wide scale block b of the
// 512-wide latent. Fills the reduce scaler tile (1.0 in row 0 of each face, as the stock per_token_cast_to_fp8 reader)
// and reads the 4 bf16 tiles 4b..4b+3 of lat [1,1,S,512] TILE. CT: 0 cb_in, 1 cb_scaler, then TensorAccessorArgs(lat).
// Common: lat addr. RT: b.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr auto lat_args = TensorAccessorArgs<2>();
    const uint32_t b = get_arg_val<uint32_t>(0);
    const auto src = TensorAccessor(lat_args, get_common_arg_val<uint32_t>(0));
    Noc noc;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scaler_obj(cb_scaler);
    cb_scaler_obj.reserve_back(1);
    CoreLocalMem<volatile uint32_t> sc(cb_scaler_obj.get_write_ptr());
    noc.async_write_zeros(cb_scaler_obj, get_tile_size(cb_scaler), {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t j = 0; j < 16; ++j) {
            sc[f * 256 + j] = 0x3f800000u;
        }
    }
    cb_scaler_obj.push_back(1);
    const uint32_t tb = get_tile_size(cb_in);
    cb_in_obj.reserve_back(4);
    for (uint32_t t = 0; t < 4; ++t) {
        noc.async_read(src, cb_in_obj, tb, {.page_id = 4 * b + t, .offset_bytes = 0}, {.offset_bytes = t * tb});
    }
    noc.async_read_barrier();
    cb_in_obj.push_back(4);
}
