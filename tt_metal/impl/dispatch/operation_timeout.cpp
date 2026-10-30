// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/operation_timeout.hpp>

#include <tt-metalium/mesh_device.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {

std::chrono::duration<float> GetOperationTimeout(const distributed::MeshDevice& mesh_device) {
    return mesh_device.impl().metal_env().get_rtoptions().get_timeout_duration_for_operations();
}

void SetOperationTimeout(distributed::MeshDevice& mesh_device, std::chrono::duration<float> timeout) {
    mesh_device.impl().metal_env().get_rtoptions().set_timeout_duration_for_operations(timeout);
}

}  // namespace tt::tt_metal::experimental
