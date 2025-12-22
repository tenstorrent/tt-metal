// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_lib/llk_io_pack.h"
#include "tools/profiler/kernel_profiler.hpp"

inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    DeviceZoneScopedSumN2("CB-COMPUTE-RESERVE-BACK");
    _llk_wait_for_free_tiles_(operand, num_tiles);
}

inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    _llk_push_tiles_(operand, num_tiles);
}
