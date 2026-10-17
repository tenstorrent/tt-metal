// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ONE mesh per binary: an inline function's local static is a single object across every TU,
// where a static in an anonymous namespace gives each TU its own and the fixture reads null.
#pragma once

#include <memory>

#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_fabric::erisc_bridge::bench {

inline std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& the_mesh() {
    static std::shared_ptr<tt::tt_metal::distributed::MeshDevice> m;
    return m;
}

}  // namespace tt::tt_fabric::erisc_bridge::bench
