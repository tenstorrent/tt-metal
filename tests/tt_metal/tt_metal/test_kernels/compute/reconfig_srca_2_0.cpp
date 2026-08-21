// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/2_0/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) reconfig kernel: IDENTICAL to reconfig_srca_legacy.cpp except the per-tile SrcA reconfig
// uses experimental::reconfig_data_format_srca (no CB id) instead of the legacy CB-id call. copy_tile /
// pack_tile stay the legacy CB-id API so the differential isolates reconfig_data_format_srca. The operand
// passed in carries c_0's OWN format/geometry (matched-format reconfig, same as the legacy kernel) -- this
// must reprogram SrcA without corrupting the copy that follows. Output must be bit-for-bit identical to the
// legacy kernel.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        experimental::reconfig_data_format_srca(InOp(in_cb.read_address()));
        copy_tile(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
