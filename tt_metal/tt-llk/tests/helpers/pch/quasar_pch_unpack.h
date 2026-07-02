// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCH prefix for the Quasar UNPACK TRISC role (compiled with -DLLK_TRISC_UNPACK).
// Base prefix + the role-common unpack header. `llk_unpack_common.h` is included
// in every unpack-role branch before the per-variant `params.h`, and its include
// closure is static (reaches cfg_defines.h but never build.h/params.h), so it is
// PCH-safe. Operation-specific unpack headers (e.g. llk_unpack_unary_operand.h)
// vary per test and are intentionally excluded.
#pragma once

// clang-format off
// Order matters: the base prefix must precede the role-common header, matching
// the include order in every kernel .cpp (universal prefix before the role
// branch). Do not let include sorting reorder these.
#include "quasar_pch_common.h"
#include "llk_unpack_common.h"
// clang-format on
