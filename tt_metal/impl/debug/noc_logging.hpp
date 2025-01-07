// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/device.hpp"

namespace tt {
void ClearNocData(tt_metal::IDevice* device);
void DumpNocData(const std::vector<tt_metal::IDevice*>& devices);
}  // namespace tt
