// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Returns the list of logical Tensix cores reserved as fast-dispatch cores on user-facing chips.
const std::vector<CoreCoord>& get_logical_dispatch_cores_on_user_chips();

}  // namespace tt::tt_metal
