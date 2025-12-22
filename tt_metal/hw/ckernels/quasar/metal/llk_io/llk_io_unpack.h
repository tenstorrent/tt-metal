// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_lib/llk_io_unpack.h"
#include "tools/profiler/kernel_profiler.hpp"

inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    _llk_wait_tiles_(operand, num_tiles);
}

inline void llk_pop_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    _llk_pop_tiles_(operand, num_tiles);
}
