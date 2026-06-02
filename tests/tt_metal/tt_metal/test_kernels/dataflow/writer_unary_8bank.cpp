// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    uint32_t num_tiles = get_arg(args::num_tiles);
#else
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(2); // Index 2 to match with regular writer_unary
#endif

    constexpr uint32_t onetile = 1;
#ifdef ARCH_QUASAR
    DataflowBuffer dfb_out(dfb::in);
    uint32_t tile_bytes = dfb_out.get_entry_size();
#else
    constexpr uint32_t out_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_id_out0 = out_id;
    CircularBuffer cb(cb_id_out0);
    uint32_t tile_bytes = get_tile_size(cb_id_out0);
#endif
#ifdef ARCH_QUASAR
    const auto s = TensorAccessor(ta::dst_tensor);
#else
    const auto s = TensorAccessor(dst_args, dst_addr);
#endif

    Noc noc;

    for (uint32_t i = 0; i < num_tiles; i++) {
#ifdef ARCH_QUASAR
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
#else
        cb.wait_front(onetile);
        noc.async_write(cb, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(onetile);
#endif
    }
}
