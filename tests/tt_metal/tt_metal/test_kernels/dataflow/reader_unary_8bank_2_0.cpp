// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void generate_bcast_scaler() {
    DataflowBuffer dfb1(dfb::out_scaler);
    uint32_t scaler = get_arg(args::scaler);
    union {
        float f;
        uint32_t u;
    } u;
    u.u = scaler;
    constexpr uint32_t onetile = 1;
    dfb1.reserve_back(onetile);
    // On Quasar, dfb.get_write_ptr() returns a cacheable-alias L1 address; the noncacheable
    // alias is reached by adding MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR. On Gen1 the
    // returned pointer is already usable; the macro doesn't exist there.
#ifdef ARCH_QUASAR
    auto ptr = reinterpret_cast<uint16_t*>(dfb1.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);
#else
    auto ptr = reinterpret_cast<uint16_t*>(dfb1.get_write_ptr());
#endif
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(0);
    }

    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 16; j++) {
            ptr[k * 256 + j] = uint16_t(u.u >> 16);
        }
    }
    dfb1.push_back(onetile);
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb0(dfb::out_data);
    const uint32_t tile_bytes = dfb0.get_entry_size();

    const auto src_a = TensorAccessor(tensor::src_tensor);

#if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE;
#else
    constexpr uint32_t blk = 1;  // 1 for correctness for unfused kernels
#endif

#ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
#else
    constexpr uint32_t tile_offset = 0;
#endif

    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        uint32_t rem = blk;
        dfb0.reserve_back(rem);
#ifdef ARCH_QUASAR
        uint32_t l1_write_addr = dfb0.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR;
#else
        uint32_t l1_write_addr = dfb0.get_write_ptr();
#endif

        for (uint32_t r = 0; r < rem; r++) {
            uint64_t src_noc_addr =
                get_noc_addr(i + r + tile_offset, src_a);  // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r * tile_bytes);
            noc_async_read(src_noc_addr, addr, tile_bytes);
        }
        noc.async_read_barrier();
        dfb0.push_back(rem);
    }
}
