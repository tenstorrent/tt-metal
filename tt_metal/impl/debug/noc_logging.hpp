// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <host_api.hpp>
#include <vector>

namespace tt {
void ClearNocData(chip_id_t device_id);
void DumpNocData(const std::vector<chip_id_t>& device_ids);
}  // namespace tt
