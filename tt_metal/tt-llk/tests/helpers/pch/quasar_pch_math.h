// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCH prefix for the Quasar MATH TRISC role (compiled with -DLLK_TRISC_MATH).
// Base prefix + the role-common math header. `llk_math_common.h` (which pulls in
// `cmath_common.h`) is included in every math-role branch before `params.h`; its
// closure is static (never reaches build.h/params.h), so it is PCH-safe.
// Operation-specific math headers (e.g. llk_math_eltwise_unary_datacopy.h) vary
// per test and are intentionally excluded.
#pragma once

// clang-format off
// Order matters: the base prefix must precede the role-common header, matching
// the include order in every kernel .cpp (universal prefix before the role
// branch). Do not let include sorting reorder these.
#include "quasar_pch_common.h"
#include "llk_math_common.h"
// clang-format on
