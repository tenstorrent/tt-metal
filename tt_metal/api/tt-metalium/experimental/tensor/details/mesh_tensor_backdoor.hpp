// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * This header contains backdoor support for accumulation op to allow update of the underlying storage of a MeshTensor.
 * This is a temporary solution for #37807 and should be quickly refactored out.
 */

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::do_not_use {

// Update the underlying storage of a MeshTensor
inline void do_not_use_update_mesh_tensor_storage(MeshTensor& mesh_tensor, const DeviceStorage& storage) {
    mesh_tensor.impl->storage_ = storage;
}

}  // namespace tt::tt_metal::do_not_use
