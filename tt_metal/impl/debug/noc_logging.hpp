// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <host_api.hpp>
#include <tt-metalium/device_types.hpp>
#include <vector>

namespace tt {
void ClearNocData(ChipId device_id);
void DumpNocData(const std::vector<ChipId>& devices);
}  // namespace tt
