// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"            // legacy CB-id pack_tile (reference path)
#include "api/compute/tile_move_copy.h"  // legacy copy_tile OP (the copy-into-DST op stays CB-id)
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"  // id-free copy_init
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// One DEST window, packed once: copies `copy_tiles` input tiles into DST[0..copy_tiles) and packs `pack_tiles`
// tiles out of DST[0..pack_tiles), through the id-free experimental::pack_block (id_free == 1) or the legacy
// CB-id pack_tile loop (id_free == 0). With pack_tiles above the packer's DEST capacity (fp32 dest-acc +
// SyncHalf with a Float32 output: 4 tiles of 32x32), the pack of tile `capacity` must trip the LLK DST-capacity
// assert; without it the packer silently reads stale DEST rows. Deliberately a single window with no loop: by the time
// PACK asserts, UNPACK and MATH have finished (MATH released DEST at tile_regs_commit), so only the asserting thread
// parks and the core can be re-initialized afterwards.
void kernel_main() {
    constexpr std::uint32_t copy_tiles = get_compile_time_arg_val(0);
    constexpr std::uint32_t pack_tiles = get_compile_time_arg_val(1);
    constexpr std::uint32_t id_free = get_compile_time_arg_val(2);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    experimental::copy_init(InOp(in_cb.read_address()));

    tile_regs_acquire();
    cb0.wait_front(copy_tiles);
    cb16.reserve_back(pack_tiles);
    for (std::uint32_t i = 0; i < copy_tiles; ++i) {
        copy_tile(tt::CBIndex::c_0, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();

    if constexpr (id_free == 1) {
        experimental::pack_block(OutOp(out_cb.write_address()), 0 /*ifrom_dst*/, pack_tiles);
    } else {
        for (std::uint32_t i = 0; i < pack_tiles; ++i) {
            pack_tile(i, tt::CBIndex::c_16);
        }
    }

    cb0.pop_front(copy_tiles);
    cb16.push_back(pack_tiles);
    tile_regs_release();
}
