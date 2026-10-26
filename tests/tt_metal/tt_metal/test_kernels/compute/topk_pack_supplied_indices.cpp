// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr std::uint32_t scores_tile = 0;
    constexpr std::uint32_t ids_tile = 2;
    constexpr std::uint32_t block = 4;  // two score tiles, then two ID tiles
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (std::uint32_t tile = 0; tile < count; tile += block) {
        input.wait_front(block);
        output.reserve_back(block);
        tile_regs_acquire();
        copy_init(tt::CBIndex::c_0);
        for (std::uint32_t i = 0; i < block; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }
        // Alternate complete size/half sweeps between score and ID aliases.
        const std::uint32_t packed_dst = (tile / block / 4 % 2) * 2;
        topk_xl_pack_supplied_indices_init();
        switch (tile / block % 4) {
            case 0: topk_xl_pack_supplied_indices<2048>(scores_tile, ids_tile, packed_dst); break;
            case 1: topk_xl_pack_supplied_indices<1024>(scores_tile, ids_tile, packed_dst); break;
            case 2: topk_xl_pack_supplied_indices<512>(scores_tile, ids_tile, packed_dst, 0 /* score_half */); break;
            case 3: topk_xl_pack_supplied_indices<512>(scores_tile, ids_tile, packed_dst, 1 /* score_half */); break;
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t i = 0; i < block; ++i) {
            pack_tile(i, tt::CBIndex::c_16);
        }
        tile_regs_release();
        input.pop_front(block);
        output.push_back(block);
    }
}
