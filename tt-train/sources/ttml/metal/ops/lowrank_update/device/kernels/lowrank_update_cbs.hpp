// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace lowrank_update_cb {

constexpr uint32_t a = 0;    // [Mt] bf16: the a column of the current image, resident
constexpr uint32_t w = 1;    // [2] bf16: the w tile of the current output column
constexpr uint32_t acc = 2;  // [2] bf16: the acc tile of the item
constexpr uint32_t out = 3;  // [2] bf16

}  // namespace lowrank_update_cb
