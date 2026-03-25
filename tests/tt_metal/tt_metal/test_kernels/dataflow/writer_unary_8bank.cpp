// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(2); // Index 2 to match with regular writer_unary

    constexpr uint32_t onetile = 1;
    constexpr auto dst_args = TensorAccessorArgs<0>();
#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    experimental::DataflowBuffer dfb_out(dfb_out_id);
    uint32_t tile_bytes = dfb_out.get_entry_size();
#else
    constexpr uint32_t cb_id_out0 = 16;
    experimental::CircularBuffer cb(cb_id_out0);
    uint32_t tile_bytes = get_tile_size(cb_id_out0);
#endif
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    experimental::Noc noc;

    for (uint32_t i = 0; i < num_tiles; i++) {
#ifdef ARCH_QUASAR
        dfb_out.write_out(noc, s, {.page_id = i});
#else
        cb.wait_front(onetile);
        noc.async_write(cb, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(onetile);
#endif
    }
#ifdef ARCH_QUASAR
    LocalDFBInterface& local_dfb_interface = g_dfb_interface[dfb_out_id];
    for (uint32_t i = 0; i < local_dfb_interface.num_txn_ids; i++) {
        noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(local_dfb_interface.txn_ids[i]);
    }
#endif
}
