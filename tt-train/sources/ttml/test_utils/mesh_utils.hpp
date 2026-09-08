// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttml::test_utils {

// Number of devices available on the host.
//
// Probed through a short-lived MetalEnv rather than GetNumAvailableDevices(), which would
// implicitly create the process-wide silicon MetalContext and leave it in place; core::MeshDevice
// builds its own MetalEnv when it opens a device, and only one environment for the silicon cluster
// may exist at a time. Cached because constructing a MetalEnv is expensive.
inline uint32_t host_device_count() {
    static const uint32_t count = tt::tt_metal::MetalEnv{}.get_num_available_devices();
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
