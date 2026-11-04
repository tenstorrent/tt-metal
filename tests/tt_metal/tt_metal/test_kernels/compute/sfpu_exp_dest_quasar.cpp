// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Example: exp on Dest through the SFPU op-type compute API. UNPACK copies a block into Dest, MATH runs exp in
// place on each tile, PACK writes the block out.
//
// Compile-time args: 0 = input DFB id, 1 = output DFB id, 2 = tiles per block, 3 = number of blocks,
// 4 = DataFormat of the tiles in Dest.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr std::uint32_t in_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t out_id = get_compile_time_arg_val(1);
    constexpr std::uint32_t tiles_per_block = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr DataFormat dest_format = static_cast<DataFormat>(get_compile_time_arg_val(4));

    DataflowBuffer dfb_in(static_cast<std::uint16_t>(in_id));
    DataflowBuffer dfb_out(static_cast<std::uint16_t>(out_id));

    compute_kernel_hw_startup(in_id, out_id);
    copy_init(in_id);
    sfpu_init<op::Exp, SfpuReg::Dest, dest_format>();

    for (std::uint32_t block = 0; block < num_blocks; ++block) {
        tile_regs_acquire();
        dfb_in.wait_front(tiles_per_block);

        copy_block(in_id, /* start_in_tile_index */ 0, /* start_dst_tile_index */ 0, tiles_per_block);
        for (std::uint32_t tile = 0; tile < tiles_per_block; ++tile) {
            sfpu_tile<op::Exp, SfpuReg::Dest, dest_format>(tile);  // exp(Dest[tile]) in place
        }

        tile_regs_commit();
        tile_regs_wait();

        for (std::uint32_t tile = 0; tile < tiles_per_block; ++tile) {
            dfb_out.reserve_back(1);
            pack_tile(tile, out_id);
            dfb_out.push_back(1);
        }

        dfb_in.pop_front(tiles_per_block);
        tile_regs_release();
    }
}
