// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header has been deprecated and merged into kernel_types.hpp
// All types (DataMovementProcessor, NOC, NOC_MODE) are now in kernel_types.hpp
// Please update your includes to: #include <tt-metalium/kernel_types.hpp>

#if defined(__GNUC__) || defined(__clang__)
#pragma message("data_types.hpp is deprecated. Include kernel_types.hpp instead.")
#elif defined(_MSC_VER)
#pragma message("data_types.hpp is deprecated. Include kernel_types.hpp instead.")
#endif

#include <tt-metalium/kernel_types.hpp>
