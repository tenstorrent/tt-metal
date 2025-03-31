// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/strong_type.hpp>
#include <cstdint>

namespace tt::tt_metal::distributed {

// Identifier for a mesh trace.
using MeshTraceId = tt::stl::StrongType<uint32_t, struct MeshTraceIdTag>;

}  // namespace tt::tt_metal::distributed
