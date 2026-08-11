// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttml::test_utils {

// Number of devices available on the host.
inline bool host_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    return tt::tt_metal::GetNumAvailableDevices() >= shape.mesh_size();
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
                tt::tt_metal::GetNumAvailableDevices());                       \
        }                                                                      \
    } while (0)
