// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/tensor.h"
#include "experimental/noc.h"

// #include "api/debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4);  // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t scaler = get_arg_val<uint32_t>(8);

    constexpr auto src_args = TensorAccessorArgs<0>();
#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    experimental::DataflowBuffer dfb_in(dfb_in_id);
#else
    constexpr uint32_t cb_id_in0 = 0;
    experimental::CircularBuffer cb(cb_id_in0);
#endif

    experimental::Noc noc;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
#ifdef ARCH_QUASAR
    uint32_t tile_bytes = dfb_in.get_entry_size();
#else
    uint32_t tile_bytes = cb.get_tile_size();
#endif

#ifndef ARCH_QUASAR
    if (scaler != 0) {
        union {
            float f;
            uint32_t u;
        } u;
        u.u = scaler;
        // DPRINT << "TWH Scaler = " << F32(u.f) << ENDL();
        constexpr uint32_t cb_in_2 = 2;
        experimental::CircularBuffer cb2(cb_in_2);
        cb2.reserve_back(1);
        auto ptr = reinterpret_cast<uint16_t*>(cb2.get_write_ptr());
        for (int j = 0; j < 1024; j++) {
            ptr[j] = uint16_t(0);
        }

        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 16; j++) {
                ptr[k * 256 + j] = uint16_t(u.u >> 16);
            }
        }
        cb2.push_back(1);
    }
#endif

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
#ifdef ARCH_QUASAR
                DPRINT << "reading tile " << i_tile << ENDL();
                dfb_in.read_in(noc, s, {.page_id = i_tile});
#else
                cb.reserve_back(onetile);

                noc.async_read(s, cb, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();

                cb.push_back(onetile);
#endif
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N

#ifdef ARCH_QUASAR
    dfb_in.finish();
#endif
}
