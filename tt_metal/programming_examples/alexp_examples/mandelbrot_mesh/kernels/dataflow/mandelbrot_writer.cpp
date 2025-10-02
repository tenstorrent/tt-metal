// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    // DEBUG: Print detailed RISC-V dataflow core startup info
    DPRINT << "=== MANDELBROT WRITER KERNEL STARTED ===" << ENDL();

#if defined(COMPILE_FOR_BRISC)
    DPRINT << "CORE TYPE: BRISC (Binary RISC-V - Reader Core)" << ENDL();
    DPRINT << "BRISC FUNCTION: Data movement from DRAM to L1" << ENDL();
#elif defined(COMPILE_FOR_NCRISC)
    DPRINT << "CORE TYPE: NCRISC (Network RISC-V - Writer Core)" << ENDL();
    DPRINT << "NCRISC FUNCTION: Data movement from L1 to DRAM/NOC" << ENDL();
#elif defined(COMPILE_FOR_ERISC)
    DPRINT << "CORE TYPE: ERISC (Ethernet RISC-V - Network Core)" << ENDL();
    DPRINT << "ERISC FUNCTION: Inter-device communication" << ENDL();
#else
    DPRINT << "CORE TYPE: DATAFLOW RISC-V CORE" << ENDL();
#endif

    DPRINT << "Dst addr: 0x" << HEX() << dst_addr << ENDL();
    DPRINT << "Num tiles: " << num_tiles << ENDL();

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    DPRINT << "Tile size: " << tile_size_bytes << " bytes" << ENDL();

    // Address of the output buffer
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, dst_addr, tile_size_bytes);

    // Loop over all the tiles and write them to the output buffer
    for (uint32_t i = 0; i < num_tiles; i++) {
        // DEBUG: Print detailed tile processing info with core identification - REDUCED TILES
        if (i < 3 || i % 200 == 0 || i == num_tiles - 1) { // Show first 3 tiles + every 200th tile + last tile
#if defined(COMPILE_FOR_BRISC)
            DPRINT << "[BRISC-READER] Writing tile " << i << "/" << num_tiles << ENDL();
#elif defined(COMPILE_FOR_NCRISC)
            DPRINT << "[NCRISC-WRITER] Writing tile " << i << "/" << num_tiles << ENDL();
#elif defined(COMPILE_FOR_ERISC)
            DPRINT << "[ERISC-NETWORK] Writing tile " << i << "/" << num_tiles << ENDL();
#else
            DPRINT << "[DATAFLOW-RISC] Writing tile " << i << "/" << num_tiles << ENDL();
#endif
        }

        // Make sure there is a tile in the circular buffer
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);

        if (i < 3) {
            DPRINT << "CB addr: 0x" << HEX() << cb_out0_addr << ENDL();
        }

        // Write the tile to DRAM
        noc_async_write_tile(i, out0, cb_out0_addr);
        noc_async_write_barrier();

        // Mark the tile as consumed
        cb_pop_front(cb_out0, 1);

        if (i < 3 || i % 200 == 0 || i == num_tiles - 1) {
            DPRINT << "Completed tile " << i << ENDL();
        }
    }

    // DEBUG: Print completion with core identification
#if defined(COMPILE_FOR_BRISC)
    DPRINT << "[BRISC-READER] MANDELBROT WRITER KERNEL COMPLETED - wrote " << num_tiles << " tiles" << ENDL();
#elif defined(COMPILE_FOR_NCRISC)
    DPRINT << "[NCRISC-WRITER] MANDELBROT WRITER KERNEL COMPLETED - wrote " << num_tiles << " tiles" << ENDL();
#elif defined(COMPILE_FOR_ERISC)
    DPRINT << "[ERISC-NETWORK] MANDELBROT WRITER KERNEL COMPLETED - wrote " << num_tiles << " tiles" << ENDL();
#else
    DPRINT << "[DATAFLOW-RISC] MANDELBROT WRITER KERNEL COMPLETED - wrote " << num_tiles << " tiles" << ENDL();
#endif
}
