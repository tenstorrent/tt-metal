// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace tt::tt_metal::pinned_upload {

// Tensor uploads at or below this size copy through the command queue; pinning costs more than it saves.
inline constexpr size_t k_pin_write_threshold_bytes = 32 * 1024 * 1024;

}  // namespace tt::tt_metal::pinned_upload
