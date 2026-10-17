// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_pack_custom_mm.h"
#include "llk_outputs.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK CUSTOM_MM
 *
 * The packer tile stride of the custom_mm family's dense layout (output tiles
 * 32 DEST rows apart), sized from the output operand's configured pack-source
 * format. The init and the uninit take the same output operand.
 *
 * Uses experimental/llk_pack_custom_mm.h as the low-level implementation.
 *************************************************************************/

// Program the dense tile stride for `output`; no-op when dense_packing is false.
// Call after the pack init that programmed the default strides.
template <bool dense_packing = false>
inline void llk_pack_custom_mm_init(const std::uint32_t output) {
    SAN_HOOK(unsupported());
    _llk_pack_custom_mm_init_<dense_packing>(pack_src_format[get_output_id(output)]);
}

// Restore the default tile stride for `output` after the last dense pack; no-op when
// dense_packing is false.
template <bool dense_packing = false>
inline void llk_pack_custom_mm_uninit(const std::uint32_t output) {
    SAN_HOOK(unsupported());
    _llk_pack_custom_mm_uninit_<dense_packing>(pack_src_format[get_output_id(output)]);
}
