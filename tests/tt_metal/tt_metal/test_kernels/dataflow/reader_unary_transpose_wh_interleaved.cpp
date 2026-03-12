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
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);

    constexpr auto src_args = TensorAccessorArgs<0>();

    experimental::Noc noc;
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb0(0);
    const uint32_t tile_bytes = dfb0.get_entry_size();
#else
    constexpr uint32_t cb_id_in0 = 0;
    experimental::CircularBuffer cb0(cb_id_in0);
    const uint32_t tile_bytes = cb0.get_tile_size();
#endif

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef REDUCE_SCALER
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb1(1);
    dfb1.reserve_back(1);
#else
    constexpr uint32_t cb_id_in2 = 2;
    experimental::CircularBuffer cb2(cb_id_in2);
    cb2.reserve_back(1);
#endif
    constexpr uint32_t scaler = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    experimental::UnicastEndpoint mem_zero_endpoint;

    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
#ifdef ARCH_QUASAR
        noc.async_read(
            mem_zero_endpoint, dfb1, MEM_ZEROS_SIZE, {.addr = MEM_ZEROS_BASE}, {.offset_bytes = i * MEM_ZEROS_SIZE});
#else
        noc.async_read(
            mem_zero_endpoint,
            experimental::use<experimental::CircularBuffer::AddrSelector::WRITE_PTR>(cb2),
            MEM_ZEROS_SIZE,
            {.addr = MEM_ZEROS_BASE},
            {.offset_bytes = i * MEM_ZEROS_SIZE});
#endif
    }
    noc.async_read_barrier();

#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        dfb1.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);
#else
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id_in2));
#endif
    uint32_t idx = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        uint32_t curr_idx = idx;
        for (uint32_t j = 0; j < 8; ++j) {
            ptr[curr_idx] = scaler;
            curr_idx++;
        }
        idx += 128;
    }
#ifdef ARCH_QUASAR
    dfb1.push_back(1);
#else
    cb2.push_back(1);
#endif
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
                dfb0.reserve_back(onetile);
                noc.async_read(s, dfb0, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();
                dfb0.push_back(onetile);
#else
                cb0.reserve_back(onetile);
                noc.async_read(s, cb0, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();
                cb0.push_back(onetile);
#endif
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
