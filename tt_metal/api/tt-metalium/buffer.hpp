// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED-FORWARDING-HEADER: <tt-metalium/buffer.hpp> moved to <tt-metalium/memory/buffer/buffer.hpp>.
// This forwarding header will be removed after 2027-03-05.
#ifndef TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS
// NOLINTNEXTLINE(clang-diagnostic-#warnings,clang-diagnostic-#pragma-messages)
#warning \
    "<tt-metalium/buffer.hpp> is deprecated and will be removed after 2027-03-05; include <tt-metalium/memory/buffer/buffer.hpp> instead (define TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS to silence this message)"
#endif

#include <tt-metalium/memory/buffer/buffer.hpp>  // IWYU pragma: export
