// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

DeviceAddr get_per_core_address(const distributed::MeshBuffer& mesh_buffer, const CoreCoord& core) {
    auto* buffer = mesh_buffer.get_reference_buffer();
    TT_FATAL(is_per_core_allocation(*buffer), "Buffer does not use per-core allocation");
    return get_per_core_address(*buffer, core);
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
