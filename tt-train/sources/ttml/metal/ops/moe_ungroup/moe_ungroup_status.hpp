// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal::moe_ungroup_validation {

constexpr uint32_t kOffsetsNonzeroStart = 1U << 0U;
constexpr uint32_t kOffsetsMisaligned = 1U << 1U;
constexpr uint32_t kOffsetsDecreasing = 1U << 2U;
constexpr uint32_t kOffsetsExceedCapacity = 1U << 3U;

}  // namespace ttml::metal::moe_ungroup_validation
