// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-TRISC gating macros, from scratch (no api/compute).
//
// A compute kernel .cpp is compiled three times, once per Tensix RISC:
// UNPACK (TRISC0), MATH (TRISC1), PACK (TRISC2).

#ifndef SST_COMPUTE_DEFS_H
#define SST_COMPUTE_DEFS_H

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

#ifndef MATH
#ifdef TRISC_MATH
#define MATH(x) x
#else
#define MATH(x)
#endif
#endif

#ifndef PACK
#ifdef TRISC_PACK
#define PACK(x) x
#else
#define PACK(x)
#endif
#endif

#ifndef UNPACK
#ifdef TRISC_UNPACK
#define UNPACK(x) x
#else
#define UNPACK(x)
#endif
#endif

#endif  // SST_COMPUTE_DEFS_H
