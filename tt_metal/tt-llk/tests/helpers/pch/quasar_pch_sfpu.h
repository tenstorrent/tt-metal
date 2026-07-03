// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCH prefix for the Quasar SFPU TRISC role (compiled with -DLLK_TRISC_ISOLATE_SFPU).
//
// Base prefix only, by design. Unlike unpack/math/pack, the SFPU role has NO
// consistent role-common header across tests: the isolated-SFPU tests pull in
// cmath_common.h / llk_math_common.h / llk_srcs.h, whereas the common case
// (tests without an explicit SFPU pipeline) resolves `run_kernel` through the
// `sfpu_stub.h` stub, which — under LLK_TRISC_ISOLATE_SFPU — includes the
// per-variant `params.h`/`build.h`. There is therefore no non-per-variant SFPU
// header safe to bake in beyond the universal base, so we precompile only the
// base prefix (still capturing the dominant ckernel.h / ckernel_ops.h cost) for
// this role.
#pragma once

#include "quasar_pch_common.h"
