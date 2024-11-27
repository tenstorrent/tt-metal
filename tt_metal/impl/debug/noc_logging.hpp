// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/device/device.hpp"

namespace tt {
void ClearNocData(tt_metal::Device* device);
void DumpNocData(const std::vector<tt_metal::Device*>& devices);
}  // namespace tt
