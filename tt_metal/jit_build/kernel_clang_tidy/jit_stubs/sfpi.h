// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stub for sfpi.h (clang-tidy analysis only)
//
// The real sfpi.h (from the SFPI toolchain at runtime/sfpi/include/ or
// /opt/tenstorrent/sfpi/include/) requires TT RISC-V compiler builtins
// (__builtin_rvtt_*) that host clang does not have.  Providing this empty
// stub allows clang-tidy to continue past the #include and analyse the
// rest of the compute kernel, consistent with the -ferror-limit=0 approach.
//
// In CI, the real sfpi.h is found via KCT_SFPI_ROOT and will produce
// compiler errors on the builtin calls; those errors are expected and
// documented in CMakeLists.txt.

#pragma once
