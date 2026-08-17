// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_event.hpp>

namespace tt::tt_metal::distributed {

// Internal: no MeshCommandQueue equivalent yet (#26591).
bool EventQuery(const MeshEvent& event);

}  // namespace tt::tt_metal::distributed
