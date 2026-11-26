// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED-FORWARDING-HEADER: <tt-metalium/mesh_device_view.hpp> moved to <tt-metalium/mesh/mesh_device_view.hpp>.
// This forwarding header will be removed after 2027-03-05.
#ifndef TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS
// NOLINTNEXTLINE(clang-diagnostic-#warnings,clang-diagnostic-#pragma-messages)
#warning \
    "<tt-metalium/mesh_device_view.hpp> is deprecated and will be removed after 2027-03-05; include <tt-metalium/mesh/mesh_device_view.hpp> instead (define TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS to silence this message)"
#endif

#include <tt-metalium/mesh/mesh_device_view.hpp>  // IWYU pragma: export
