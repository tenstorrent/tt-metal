// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_in = get_compile_time_arg_val(0);
constexpr uint32_t cb_out = get_compile_time_arg_val(1);
constexpr uint32_t repeats = get_compile_time_arg_val(2);
constexpr bool runtime_output = get_compile_time_arg_val(3);
constexpr uint32_t outputs_per_section = 4 + 3 * repeats;

template <ckernel::ReduceDim dim>
ALWI void pack_reduced(uint32_t output_index) {
    const uint32_t output_cb = runtime_output ? get_arg_val<uint32_t>(0) : cb_out;
    PACK((llk_pack_reduce_mask_config<ckernel::PoolType::SUM, dim, ckernel::PackMode::Default>(output_cb)));
    for (uint32_t i = 0; i < repeats; ++i) {
        pack_tile<true>(i % 2, output_cb, output_index + i);
    }
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    // Both CBs are resident sharded tensors. Copy two nonzero tiles into DEST and test the
    // packer without doing a reduction, so unused DEST values cannot accidentally hide a bad mask.
    // Two sections exercise the DEST-half flip with half sync, and reuse with full sync.
    for (uint32_t section = 0; section < 2; ++section) {
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        copy_tile(cb_in, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        const uint32_t base = section * outputs_per_section;
        pack_tile<true>(0, cb_out, base);
        pack_tile<true>(1, cb_out, base + 1);
        pack_reduced<ckernel::ReduceDim::REDUCE_ROW>(base + 2);
        pack_reduced<ckernel::ReduceDim::REDUCE_COL>(base + 2 + repeats);
        pack_reduced<ckernel::ReduceDim::REDUCE_SCALAR>(base + 2 + 2 * repeats);
        PACK((llk_pack_reduce_mask_clear()));
        pack_tile<true>(0, cb_out, base + outputs_per_section - 2);
        pack_tile<true>(1, cb_out, base + outputs_per_section - 1);
        tile_regs_release();
    }
}
