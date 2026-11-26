// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED-FORWARDING-HEADER: <tt-metalium/tensor_accessor_args.hpp> moved to
// <tt-metalium/workload/legacy/tensor_accessor_args.hpp>. This forwarding header will be removed after 2027-03-05.
#ifndef TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS
// NOLINTNEXTLINE(clang-diagnostic-#warnings,clang-diagnostic-#pragma-messages)
#warning \
    "<tt-metalium/tensor_accessor_args.hpp> is deprecated and will be removed after 2027-03-05; include <tt-metalium/workload/legacy/tensor_accessor_args.hpp> instead (define TT_METALIUM_SUPPRESS_DEPRECATED_HEADER_WARNINGS to silence this message)"
#endif

#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>  // IWYU pragma: export
