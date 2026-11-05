// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_math_dest_srca_mm.h"
#include "llk_math_common_api.h"

// Math side of DEST_SRCA_MM (experimental/llk_math_dest_srca_mm.h): D[dst] += in0 * a Float32 DEST tile moved to SrcA.

template <std::uint32_t in0_rows>
inline void llk_math_dest_srca_mm_init() {
    _llk_math_dest_srca_mm_init_<MATH_FIDELITY, in0_rows>();
}

inline void llk_math_dest_srca_mm_uninit() { _llk_math_dest_srca_mm_uninit_(); }

inline void llk_math_dest_srca_mm_page_post(const std::uint32_t dst_index) {
    _llk_math_dest_srca_mm_unpack_post_(dst_index);
}

inline void llk_math_dest_srca_mm_page_wait(const std::uint32_t page_operand, const std::uint32_t dst_index) {
    _llk_math_dest_srca_mm_unpack_wait_(dst_index, get_operand_num_faces(get_operand_id(page_operand)));
}

template <std::uint32_t in0_rows>
inline void llk_math_dest_srca_mm(
    const std::uint32_t src_index, const std::uint32_t dst_index, const std::uint32_t src_rows) {
    _llk_math_dest_srca_mm_<MATH_FIDELITY, in0_rows>(src_index, dst_index, src_rows);
}
