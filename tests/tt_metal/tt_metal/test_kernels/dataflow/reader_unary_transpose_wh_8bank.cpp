// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

// #include "api/debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4);  // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t scaler = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0 = 0;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = cb.get_tile_size();

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

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                cb.reserve_back(onetile);

                noc.async_read(s, cb, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();

                cb.push_back(onetile);
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
