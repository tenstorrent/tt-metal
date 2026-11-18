// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared constants for MiMo decode projection kernels (tt/proj_stream.py).
#pragma once

#include <cstdint>

namespace mimo_proj {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kFaceDim = 16;

}  // namespace mimo_proj
