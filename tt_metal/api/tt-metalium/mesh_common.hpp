
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/strong_type.hpp>

// Define common types used across TT-Mesh data-structures and APIs

using MeshTraceId = tt::stl::StrongType<uint32_t, struct MeshTraceIdTag>;
