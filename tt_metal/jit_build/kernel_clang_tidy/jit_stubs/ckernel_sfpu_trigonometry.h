// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KCT stub for ckernel_sfpu_trigonometry.h
//
// The real header calls __builtin_rvtt_sfpmad, a Tensix SFPU hardware
// intrinsic only available in the custom SFPI GCC fork (riscv-tt-elf-g++).
// Host clang-20 doesn't know this builtin, so we define it as a variadic
// macro that returns a default-constructed sfpi::vFloat before including
// the real header.
//
// sfpi::vFloat is default-constructible (= default) so this is valid.
// Our jit_stubs/sfpi.h stub is already active at this point (it shadowed
// the real sfpi.h), so sfpi::vFloat is in scope.

#pragma once

#ifndef KCT_SFPI_RVTT_BUILTINS_DEFINED
#define KCT_SFPI_RVTT_BUILTINS_DEFINED

// __builtin_rvtt_sfpmad(a, b, c, mod) → vFloat (fused multiply-add on SFPU)
#define __builtin_rvtt_sfpmad(...) (::sfpi::vFloat{})
// __builtin_rvtt_sfpload(dreg, mod, addr_mode) → vFloat (load from SFPU register)
#define __builtin_rvtt_sfpload(...) (::sfpi::vFloat{})
// __builtin_rvtt_sfpstore(src, dreg, mod, addr_mode) → void (store to SFPU register)
#define __builtin_rvtt_sfpstore(...) ((void)0)

#endif  // KCT_SFPI_RVTT_BUILTINS_DEFINED

#include_next "ckernel_sfpu_trigonometry.h"
