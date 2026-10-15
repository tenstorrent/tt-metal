// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK MUL REDUCE SCALAR
 *************************************************************************/

inline void llk_unpack_mul_reduce_scalar_switch_to_reduce() {
    SAN_HOOK(unsupported());
    _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
}

inline bool llk_unpack_mul_reduce_scalar_two_face_banks(const std::uint32_t icb0) {
    return _llk_unpack_mul_reduce_scalar_two_face_banks_(get_operand_tensor_shape(get_operand_id(icb0)));
}

inline void llk_unpack_mul_reduce_scalar_two_face_banks_init() {
    SAN_HOOK(unsupported());
    _llk_unpack_mul_reduce_scalar_two_face_banks_init_();
}

inline void llk_unpack_mul_reduce_scalar_two_face_banks_uninit() {
    SAN_HOOK(unsupported());
    _llk_unpack_mul_reduce_scalar_two_face_banks_uninit_();
}
