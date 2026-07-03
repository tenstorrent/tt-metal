// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCH prefix for the Quasar PACK TRISC role (compiled with -DLLK_TRISC_PACK).
// Base prefix + the role-common pack headers. `llk_pack.h` and
// `llk_pack_common.h` appear in every pack-role branch before `params.h`; both
// have static include closures (never reach build.h/params.h), so they are
// PCH-safe.
#pragma once

// clang-format off
// Order matters: the base prefix must precede the role-common headers, matching
// the include order in every kernel .cpp (universal prefix before the role
// branch). Do not let include sorting reorder these.
#include "quasar_pch_common.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
// clang-format on
