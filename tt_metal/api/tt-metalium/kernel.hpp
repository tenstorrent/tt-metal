// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// 341 = (4096/(3 * sizeof(uint32_t)), where
// - 4096 - packet size in dispatch
// - 3 - number of kernels per tensix
constexpr uint32_t max_runtime_args = 341;

}  // namespace tt::tt_metal
