// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Eltwise Add Compressed kernel
//
// Computes: out[tile] = in0[tile] + decompress(in1_compressed[tile])
//
// in0: bf16 TILE_LAYOUT (normal tensor, one format for all tiles)
// in1: compressed data (mixed bfp8/bfp4/bfp2 tiles, variable size per tile)
// assignment: 2-bit packed format indices in L1
// out: bf16 TILE_LAYOUT
//
// The compressed data CB is backed by the sharded compressed tensor.
// The compute kernel reconfigures the unpacker per tile based on the assignment.
// Variable tile sizes are handled by manually advancing the CB read pointer.

#include "../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
using namespace ckernel;
#include "../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_compressed.h"
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    constexpr uint32_t assign_l1_addr = get_named_compile_time_arg_val("assign_l1_addr");

#if defined(COMPILE_FOR_NCRISC)
    // NCRISC: Setup sharded buffers — data is already in L1
    constexpr uint32_t cb_in0_num_pages = get_named_compile_time_arg_val("cb_in0_num_pages");
    constexpr uint32_t cb_in1_num_pages = get_named_compile_time_arg_val("cb_in1_num_pages");

    unified_kernels::setup_sharded_buffer(cb_in0, cb_in0_num_pages);
    unified_kernels::setup_sharded_buffer(cb_in1, cb_in1_num_pages);

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: no-op

#elif defined(COMPILE_FOR_TRISC)
    // TRISC: Per-tile add with compressed format reconfig
    deepseek_compute_kernel_init();

    // Assignment reading helpers (inline, no LLK dependency)
    constexpr uint32_t ASSIGN_BITS = 2;
    constexpr uint32_t ASSIGN_MASK = (1 << ASSIGN_BITS) - 1;
    constexpr uint32_t TILES_PER_BYTE = 4;
    constexpr uint32_t FMT_BFP0 = 3;
    constexpr uint32_t TILE_SIZES[] = {1088, 576, 320, 0};  // bfp8, bfp4, bfp2, bfp0

    volatile uint8_t* assign_ptr = reinterpret_cast<volatile uint8_t*>(assign_l1_addr);

    auto get_fmt = [&](uint32_t tile_id) -> uint32_t {
        return (assign_ptr[tile_id / TILES_PER_BYTE] >> ((tile_id % TILES_PER_BYTE) * ASSIGN_BITS)) & ASSIGN_MASK;
    };

    // Init: reads tile shape from cb_in0 internally
    compressed::add_tiles_init_in1_compressed(cb_in0);

    // Wait for all input data to be ready (sharded, already in L1)
    cb_wait_front(cb_in0, num_tiles);
    cb_wait_front(cb_in1, 1);

    // Get base L1 addresses and page size (in >> cb_addr_shift units)
    uint32_t in0_base_addr = 0;
    uint32_t in0_page_size = 0;
    uint32_t in1_base_addr = 0;
    UNPACK(({
        uint32_t in0_id = get_operand_id(cb_in0);
        in0_base_addr = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
        in0_page_size = get_local_cb_interface(in0_id).fifo_page_size;
        in1_base_addr = get_local_cb_interface(get_operand_id(cb_in1)).fifo_rd_ptr - 1;
    }));

    uint32_t addr_a = in0_base_addr;
    uint32_t addr_b = in1_base_addr;

    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        uint32_t fmt = get_fmt(tile_id);

        cb_reserve_back(cb_out, 1);

        if (fmt == FMT_BFP0) {
            // bfp0: just copy in0 tile to output (add zero = identity)
            tile_regs_acquire();
            copy_tile(cb_in0, tile_id, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        } else {
            compressed::reconfig_unpack_srcb(fmt);

            tile_regs_acquire();
            compressed::add_tiles_in1_compressed(cb_in0, addr_a, addr_b, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }

        cb_push_back(cb_out, 1);
        addr_a += in0_page_size;
        addr_b += TILE_SIZES[fmt] >> cb_addr_shift;
    }

    // Pop inputs
    cb_pop_front(cb_in0, num_tiles);

#endif
}
