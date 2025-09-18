// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

template <bool transpose_of_faces = true, bool is_32bit = false>
inline void llk_math_transpose_dest(uint dst_index) { _llk_math_transpose_dest_<transpose_of_faces, is_32bit>(dst_index); }

template <bool transpose_of_faces = true, bool is_32bit = false>
inline void llk_math_transpose_dest_init() { _llk_math_transpose_dest_init_<transpose_of_faces, is_32bit>(); }
