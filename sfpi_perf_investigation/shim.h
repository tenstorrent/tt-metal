// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel {
extern volatile std::uint32_t instrn_buffer[];
}

#include "sfpi.h"
