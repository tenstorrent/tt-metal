// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace ttml::test_utils {

// Global device capacity of the system mesh which spans all hosts in a multi-host run.
inline size_t system_mesh_size() {
    return tt::tt_metal::distributed::SystemMesh::instance().shape().mesh_size();
}

inline bool system_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    return system_mesh_size() >= shape.mesh_size();
}

}  // namespace ttml::test_utils

// GTEST_SKIP() expands into a return statement, so this also has to be a macro.
// Simple check to skip if the system does not have enough devices for the mesh.
// Any failures with mesh open should actually fail the test elsewhere.
#define SKIP_UNLESS_MESH_SUPPORTED(shape)                                       \
    do {                                                                        \
        const auto& s = (shape);                                                \
        if (!ttml::test_utils::system_supports_mesh(s)) {                       \
            GTEST_SKIP() << fmt::format(                                        \
                "Skipping test: a {} mesh needs {} devices, the system has {}", \
                s,                                                              \
                s.mesh_size(),                                                  \
                ttml::test_utils::system_mesh_size());                          \
        }                                                                       \
    } while (0)
