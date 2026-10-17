// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

// Tilize with a packer data-format reconfig between tilize_init and tilize_block, the order the
// pack_reconfig_data_format documentation recommends ("right after other op-specific initialization").
// The packer is first set up for a scratch output CB of a different format (c_17, never written), then
// reconfigured for the real output (c_16) and the block is tilized into c_16. The tilize pack MOP armed by
// tilize_init needs the two-face Z stride; a reconfig that resets the strides to Default mode leaves the
// second half of every output tile holding faces 1,2 instead of 2,3.
void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t scratch_out_cb = tt::CBIndex::c_17;

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);

    compute_kernel_hw_startup(in_cb, scratch_out_cb);
    tilize_init(in_cb, per_core_block_tile_cnt, scratch_out_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_in.wait_front(per_core_block_tile_cnt);
        cb_out.reserve_back(per_core_block_tile_cnt);

        tilize_block(in_cb, per_core_block_tile_cnt, out_cb);

        cb_in.pop_front(per_core_block_tile_cnt);
        cb_out.push_back(per_core_block_tile_cnt);
    }

    tilize_uninit(in_cb, out_cb);
}
