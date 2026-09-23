// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/sdpa.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_recip_test_helpers.hpp"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr std::uint32_t fidelity = get_compile_time_arg_val(1);
    constexpr std::uint32_t granularity = get_compile_time_arg_val(2);
    constexpr std::uint32_t output_tiles = 2;
    constexpr std::uint32_t recip_offset = 64;  // scratch tile 1, in DEST rows
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);

    static_assert(sdpa_fidelity_from_sel<0>() == MathFidelity::LoFi);
    static_assert(sdpa_fidelity_from_sel<2>() == MathFidelity::HiFi2);
    static_assert(sdpa_fidelity_from_sel<3>() == MathFidelity::HiFi3);
    static_assert(sdpa_fidelity_from_sel<4>() == MathFidelity::HiFi4);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (std::uint32_t tile = 0; tile < count; ++tile) {
        input.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        ckernel::test_helpers::seed_sdpa_recip_cached_sums();
        compute_sdpa_recip<output_tiles, false, 0x3f800000, granularity, fidelity>(
            tt::CBIndex::c_1, 0, recip_offset, 0);
        ckernel::test_helpers::consume_sdpa_recip_notifications<output_tiles, granularity>();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();
        input.pop_front(1);
        output.push_back(1);
    }
}
