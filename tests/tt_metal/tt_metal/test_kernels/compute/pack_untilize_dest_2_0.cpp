// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/2_0/pack_untilize.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/dataflow/circular_buffer.h"

// Id-free (2.0) pack-untilize-DEST kernel: per tile, copy c_0 -> DST, then experimental::pack_untilize_dest
// packs (untilizes) the tile straight out of the DEST register to c_16. The pack ops take only an OUTPUT
// LLKOperand (data format + tile geometry as NTTPs; L1 address the only runtime state) -- there is NO input
// operand because the source is the DEST register. copy_tile stays the legacy CB-id API so the differential
// against pack_untilize_dest_legacy.cpp isolates the pack_untilize_dest[_init] calls. Output must be
// bit-for-bit identical to the legacy kernel. block_ct_dim = full_ct_dim = block_rt_dim = 1.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_to_dst_init_short(tt::CBIndex::c_0);
    experimental::pack_untilize_dest_init<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(OutOp(out_cb.write_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        experimental::pack_untilize_dest<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(
            OutOp(out_cb.write_address()), 1 /*block_rt_dim*/);
        tile_regs_release();

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    experimental::pack_untilize_uninit(OutOp(out_cb.write_address()));
}
