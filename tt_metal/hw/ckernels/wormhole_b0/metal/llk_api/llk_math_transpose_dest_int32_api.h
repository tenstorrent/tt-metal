// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

inline void llk_math_transpose_dest_int32(uint dst_index) {
  _llk_math_transpose_dest_<true>(dst_index);
}

inline void llk_math_transpose_dest_int32_init() { _llk_math_transpose_dest_init_(); }
