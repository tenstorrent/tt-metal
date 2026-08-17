// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the per-tile triangle-solve LLK test. Reads tile 0 of l_neg into cb_l and tile 0 of
// rhs into cb_rhs. CT args: [<l_neg TensorAccessorArgs...>, <rhs TensorAccessorArgs...>].
// Runtime args: [l_neg_addr, rhs_addr].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t l_addr = get_arg_val<uint32_t>(0);
    const uint32_t rhs_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_l = tt::CBIndex::c_0;
    constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
    const uint32_t tile_bytes = get_tile_size(cb_l);

    constexpr auto l_args = TensorAccessorArgs<0>();
    const auto l_gen = TensorAccessor(l_args, l_addr, tile_bytes);
    constexpr auto rhs_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    const auto rhs_gen = TensorAccessor(rhs_args, rhs_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_l_o(cb_l);
    CircularBuffer cb_rhs_o(cb_rhs);

    cb_l_o.reserve_back(1);
    noc.async_read(l_gen, cb_l_o, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_l_o.push_back(1);

    cb_rhs_o.reserve_back(1);
    noc.async_read(rhs_gen, cb_rhs_o, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_rhs_o.push_back(1);
}
