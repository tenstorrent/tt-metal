// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_mul_reduce_scalar.h"

/*************************************************************************
 * LLK UNPACK MUL REDUCE SCALAR
 *************************************************************************/

inline void llk_unpack_mul_reduce_scalar_switch_to_reduce() { _llk_unpack_mul_reduce_scalar_switch_to_reduce_(); }
