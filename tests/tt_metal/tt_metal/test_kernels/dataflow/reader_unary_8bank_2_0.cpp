// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

#if GENERATE_BCAST_SCALER
void generate_bcast_scaler() {
    DataflowBuffer dfb1(dfb::out_scaler);
    std::uint32_t scaler = get_arg(args::scaler);
    union {
        float f;
        std::uint32_t u;
    } u;
    u.u = scaler;
    constexpr std::uint32_t onetile = 1;
    dfb1.reserve_back(onetile);
    // Local L1 fill of the reserved entry (data peek — not a NOC transfer endpoint).
    auto ptr = reinterpret_cast<std::uint16_t*>(dfb1.get_write_ptr());
    for (int j = 0; j < 1024; j++) {
        ptr[j] = std::uint16_t(0);
    }

    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 16; j++) {
            ptr[k * 256 + j] = std::uint16_t(u.u >> 16);
        }
    }
    dfb1.push_back(onetile);
}
#endif

void kernel_main() {
    std::uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr std::uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb0(dfb::out_data);
    const std::uint32_t tile_bytes = dfb0.get_entry_size();

    const auto src_a = TensorAccessor(tensor::src_tensor);

#if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr std::uint32_t blk = BLOCK_SIZE;
#else
    constexpr std::uint32_t blk = 1;  // 1 for correctness for unfused kernels
#endif

#ifdef TILE_OFFSET
    std::uint32_t tile_offset = TILE_OFFSET;
#else
    constexpr std::uint32_t tile_offset = 0;
#endif

    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    for (std::uint32_t i = 0; i < num_tiles; i += blk) {
        std::uint32_t rem = blk;
        dfb0.reserve_back(rem);

        for (std::uint32_t r = 0; r < rem; r++) {
            noc.async_read(src_a, dfb0, tile_bytes, {.page_id = i + r + tile_offset}, {.offset_bytes = r * tile_bytes});
        }
        noc.async_read_barrier();
        dfb0.push_back(rem);
    }
}
