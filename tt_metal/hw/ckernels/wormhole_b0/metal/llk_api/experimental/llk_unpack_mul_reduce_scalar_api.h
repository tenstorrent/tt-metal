// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK MUL REDUCE SCALAR
 *************************************************************************/

inline void llk_unpack_mul_reduce_scalar_switch_to_reduce() {
    SAN_HOOK(unsupported());
    _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
}
