// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// The status line and the operand kinds every CCE test kernel ABI shares. Each kernel ABI header includes this and
// re-exports, in its own namespace, the names it uses; the kernels and the host tests both include it.
namespace cce_test_status_abi {

inline constexpr uint32_t kLineBytes = 64;

// The host fills a status line with kStatusPoison before launch; the kernel writes its done magic to kStatusMagic last.
inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1;
inline constexpr uint32_t kStatusPoison = 0xBADD0E00u;

// Operand kinds. kKindOwnTl1: the kernel's own TL1; the selector is the kernel's own CCE index, so the iDMA operand is
// that CCE's package SPA and the hart checks the data through its uncached alias. kKindSramSpa: another CCE's TL1 at
// its package SPA; the selector is that CCE's index. kKindGddr: GDDR at a chiplet offset; the selector is the target
// Mimir, and the CCE's host-written route table picks the D2D port.
inline constexpr uint32_t kKindOwnTl1 = 0, kKindSramSpa = 1, kKindGddr = 2;

}  // namespace cce_test_status_abi
