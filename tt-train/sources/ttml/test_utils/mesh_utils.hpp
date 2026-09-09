// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "autograd/auto_context.hpp"

namespace ttml::test_utils {

// Number of devices this host owns, taken from the local extent of the system mesh.
//
// Probed through a short-lived MetalEnv rather than GetNumAvailableDevices() or
// SystemMesh::instance(), since both implicitly create a process-wide MetalContext and
// leave it in place.
inline size_t host_device_count() {
    // Inline and static so the count is cached between calls so only the first call probes.
    static const size_t count = [] {
        if (autograd::ctx().is_device_open()) {
            throw std::runtime_error(
                "host_device_count() must be called before opening a device: probing builds a second MetalEnv for the "
                "silicon cluster, which hangs in UMD rather than reporting an error. Move the mesh-support check ahead "
                "of ttml::autograd::ctx().open_device().");
        }
        return tt::tt_metal::MetalEnv{}.get_system_mesh().local_shape().mesh_size();
    }();
    return count;
}

inline bool host_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    return host_device_count() >= shape.mesh_size();
}

}  // namespace ttml::test_utils

// GTEST_SKIP() expands into a return statement, so this also has to be a macro.
// Simple check to skip if the host does not have enough devices for the mesh.
// Any failures with mesh open should actually fail the test elsewhere.
#define SKIP_UNLESS_MESH_SUPPORTED(shape)                                      \
    do {                                                                       \
        const auto& s = (shape);                                               \
        if (!ttml::test_utils::host_supports_mesh(s)) {                        \
            GTEST_SKIP() << fmt::format(                                       \
                "Skipping test: a {} mesh needs {} devices, this host has {}", \
                s,                                                             \
                s.mesh_size(),                                                 \
                ttml::test_utils::host_device_count());                        \
        }                                                                      \
    } while (0)
