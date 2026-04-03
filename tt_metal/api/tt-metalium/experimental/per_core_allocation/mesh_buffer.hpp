// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal::distributed {
class MeshBuffer;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::per_core_allocation {

DeviceAddr get_per_core_address(const distributed::MeshBuffer& mesh_buffer, const CoreCoord& core);

}  // namespace tt::tt_metal::experimental::per_core_allocation
