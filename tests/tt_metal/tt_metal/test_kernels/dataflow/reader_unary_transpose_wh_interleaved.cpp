// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t HtWt = get_arg(args::HtWt);

    Noc noc;
    DataflowBuffer dfb0(dfb::out_data);
    const uint32_t tile_bytes = dfb0.get_entry_size();

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef REDUCE_SCALER
    DataflowBuffer dfb1(dfb::out_scaler);
    dfb1.reserve_back(1);
    constexpr uint32_t scaler = get_arg(args::scaler);
    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    UnicastEndpoint mem_zero_endpoint;

    // Fill tile with zeros by reading from local core's MEM_ZEROS region.
    // On Gen1 the local NOC coordinates must be supplied explicitly: a default-constructed
    // endpoint reads from (0,0), which on Blackhole is a DRAM bank, and the stricter 64-byte
    // DRAM alignment check fails because MEM_ZEROS_BASE (0x32e0) is only 32-byte aligned.
    // Quasar's default endpoint already targets the local core, so no override is needed.
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc.async_read(
            mem_zero_endpoint,
            dfb1,
            MEM_ZEROS_SIZE,
#ifdef ARCH_QUASAR
            {.addr = MEM_ZEROS_BASE},
#else
            {.noc_x = my_x[noc_index], .noc_y = my_y[noc_index], .addr = MEM_ZEROS_BASE},
#endif
            {.offset_bytes = i * MEM_ZEROS_SIZE});
    }
    noc.async_read_barrier();

    // On Quasar, dfb.get_write_ptr() returns a cacheable-alias L1 address; the noncacheable
    // alias (required for NOC-port writes to be visible) is reached by adding
    // MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR. On Gen1 the returned pointer is already
    // usable; the macro doesn't exist there.
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
#ifdef ARCH_QUASAR
        dfb1.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);
#else
        dfb1.get_write_ptr());
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
    dfb1.push_back(onetile);
#endif

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const auto s = TensorAccessor(ta::src_tensor);

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                dfb0.reserve_back(onetile);
                noc.async_read(s, dfb0, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();
                dfb0.push_back(onetile);
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
