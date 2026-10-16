// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared constants for MiMo attention decode kernels (tt/attention.py).
#pragma once

#include <cstdint>

namespace mimo_attn {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kHidden = 4096;

}  // namespace mimo_attn
