// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::ttnn_fixed::distributed {

void enable_fabric_config(uint32_t num_devices);

}  // namespace ttml::ttnn_fixed::distributed
