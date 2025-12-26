// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_NCRISC)
#include "dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Sharded persistent buffer setup utilities
// ============================================================================

#if defined(COMPILE_FOR_NCRISC)

// Setup a sharded persistent buffer by reserving and pushing tiles
// This makes the buffer available for compute to read from
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}

#endif

}  // namespace deepseek_b1_ops
