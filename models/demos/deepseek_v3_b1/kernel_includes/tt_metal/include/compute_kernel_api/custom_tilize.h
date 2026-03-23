// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_reduce_api.h"
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the only the unpacker for the tilize operation.
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                                   | Type     | Valid Range | Required |
 * |----------- |--------|-----------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier              | uint32_t | 0 to 31     | True     |
 * | Function   | block  | Size of tile block to work on                 | uint32_t | > 0         | True     |
 */
// clang-format on
ALWI void tilize_init_unpack(uint32_t icb, uint32_t block, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA>(icb, call_line);
    UNPACK((llk_unpack_tilize_init(icb, block)));
}

// clang-format off
/**
 * Performs the tilize operation on a block. num_chunks * chunk must be equal to the block size.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                              | Type     | Valid Range | Required |
 * |----------- |------------------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb              | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | num_chunks       | Number of chunks to work on              | uint32_t | > 0         | True     |
 * | Function   | chunk            | Size of each chunk to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb              | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_block_custom(uint32_t icb, uint32_t num_chunks, uint32_t chunk, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_block(icb, num_chunks * chunk, 0)));

    // Acquire dst
    MATH((llk_math_wait_for_dest_available()));
    for (uint32_t c = 0; c < num_chunks; c++) {
        uint32_t dst_index = c * chunk;
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        for (uint32_t t = 0; t < chunk; t++) {
            // Datacopy
            MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                dst_index + t /*dst index*/)));
            PACK((llk_pack<DST_ACCUM_MODE>(dst_index + t /*tile index*/, ocb)));
        }
        PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
    }
    PACK((llk_packer_wait_for_math_done()));
    // Release dest
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}

}  // namespace ckernel
