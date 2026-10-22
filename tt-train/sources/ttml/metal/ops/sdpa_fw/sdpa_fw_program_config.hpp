// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

/**
 * SDPA forward tunables, in elements: multiples of 32 that divide S; 0 = let the program factory pick.
 * q_chunk_size: query rows that share one pass over K/V (fewer K/V reads, more L1).
 * k_chunk_size: K/V rows per inner-loop step (fewer softmax corrections, more L1).
 */
struct SDPAForwardProgramConfig {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
};

}  // namespace ttml::metal
