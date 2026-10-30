// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// The status line every CCE test kernel ABI shares. Each kernel ABI header includes this and re-exports, in its own
// namespace, the names it uses; the kernels and the host tests both include it.
namespace cce_test_status_abi {

inline constexpr uint32_t kLineBytes = 64;

// The host fills a status line with kStatusPoison before launch; the kernel writes its done magic to kStatusMagic last.
inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1;
inline constexpr uint32_t kStatusPoison = 0xBADD0E00u;

}  // namespace cce_test_status_abi
