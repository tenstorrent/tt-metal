// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/experimental/sdpa.h"

namespace ckernel::test_helpers {

// Hardware fixture for compute_sdpa_chunk. Keep its PACK-thread setup and
// partial-output handoff here, following sdpa_recip_test_helpers.hpp.
ALWI void init_sdpa_chunk_pack_reduce() {
    // The ordinary sdpa_reduce_row_init wrapper initializes the MATH thread.
    PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
}

// The fixture signals the whole output block once (output_granularity ==
// output_tiles). The caller reserves both CBs and owns the tile-register lifetime.
template <std::uint32_t output_tiles>
ALWI void pack_sdpa_chunk_partials(
    std::uint32_t stats_dst_tile_index,
    std::uint32_t output_dst_tile_index,
    std::uint32_t cb_stats,
    std::uint32_t cb_out) {
    PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU)));
    pack_block_contiguous(stats_dst_tile_index, cb_stats, 1);
    cb_push_back(cb_stats, 1);
    PACK((t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
    pack_block_contiguous(output_dst_tile_index, cb_out, output_tiles);
    PACK((t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU)));
    cb_push_back(cb_out, output_tiles);
}

ALWI void wait_for_sdpa_chunk_pack() { MATH((t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU))); }

}  // namespace ckernel::test_helpers
