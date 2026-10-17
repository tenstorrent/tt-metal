// Reader for the blocked bcast-col SUB/MUL paths with SrcB reuse.
//
// Unlike reader_binary_2_0.cpp, the two inputs are NOT read in lockstep: src1 (the bcast-col
// operand) is num_bcast_tiles tiles, one per row of a block, read once and pushed once; while
// src0 streams num_tiles column tiles. The compute kernel holds src1 at the front of its buffer for
// the whole run so every block re-reads the same L1 tiles, this reuse is the point of the custom op.
//
// PRIME_FULL_TILES: the same pair once more for the priming block of mul_bcast_col_custom.cpp (src3 =
// num_prime_bcast_tiles bcast tiles into in3, src2 = num_prime_tiles column tiles into in2), read
// before the main inputs because the compute kernel consumes them first.

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

#ifndef PRIME_FULL_TILES
#define PRIME_FULL_TILES 0
#endif

void kernel_main() {
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;

    // num_tiles tiles from DRAM (bank_id, addr) into dfb, one at a time, each pushed as soon as it
    // lands. A buffer that holds one block self-throttles against the compute kernel's per-block
    // pop_front.
    auto read_tiles = [&](DataflowBuffer& dfb, std::uint32_t num_tiles, std::uint32_t bank_id, std::uint32_t addr) {
        const std::uint32_t tile_bytes = dfb.get_entry_size();
        for (std::uint32_t i = 0; i < num_tiles; i++) {
            dfb.reserve_back(1);
            noc.async_read(dram_src, dfb, tile_bytes, {.bank_id = bank_id, .addr = addr}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
            addr += tile_bytes;
        }
    };

#if PRIME_FULL_TILES
    DataflowBuffer dfb2(dfb::in2);
    DataflowBuffer dfb3(dfb::in3);
    read_tiles(dfb3, get_arg(args::num_prime_bcast_tiles), get_arg(args::src3_bank_id), get_arg(args::src3_addr));
    read_tiles(dfb2, get_arg(args::num_prime_tiles), get_arg(args::src2_bank_id), get_arg(args::src2_addr));
#endif

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    // src1: num_bcast_tiles tiles (one per row of a block), read up front and never re-read.
    read_tiles(dfb1, get_arg(args::num_bcast_tiles), get_arg(args::src1_bank_id), get_arg(args::src1_addr));
    // src0: num_tiles column tiles.
    read_tiles(dfb0, get_arg(args::num_tiles), get_arg(args::src0_bank_id), get_arg(args::src0_addr));
}
