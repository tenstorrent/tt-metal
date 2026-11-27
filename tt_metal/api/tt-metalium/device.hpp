// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED-FORWARDING-HEADER: <tt-metalium/device.hpp> moved to <tt-metalium/mesh/legacy/device.hpp>.
// This forwarding header will be removed after 2027-03-05.
#ifndef TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS
// NOLINTNEXTLINE(clang-diagnostic-#warnings,clang-diagnostic-#pragma-messages)
#warning \
    "<tt-metalium/device.hpp> is deprecated and will be removed after 2027-03-05; include <tt-metalium/mesh/legacy/device.hpp> instead (define TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS to silence this message)"
#endif

#include <tt-metalium/mesh/legacy/device.hpp>  // IWYU pragma: export
