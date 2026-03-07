// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <host_api.hpp>
#include <tt-metalium/device_types.hpp>
#include <vector>

namespace tt {
void ClearNocData(tt_metal::ChipId device_id);
void DumpNocData(const std::vector<tt_metal::ChipId>& devices);
}  // namespace tt
