// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Id-free (2.0) Quasar LLK API for the datacopy op set: hw_configure, unpack A, math datacopy, pack.
// New functions only -- the legacy DFB-id LLK API (llk_*_api.h) is untouched. Every function takes the
// operand's LLKMemDescriptor (L1 format + tile shape) as an NTTP and, where the op reads/writes L1, the
// operand's runtime L1 address (16B units). No DFB id, no host format tables.
//
// Register formats are the L1 format (no Quasar format-derivation table exists on device yet), so the
// compute layer restricts 2.0 operands to 16-bit float formats.
//
// Buffer descriptors: each init allocates a BFD id from this TRISC's partition and programs it with the
// operand's L1 base address (llk_bfd_alloc.h); the execute calls index tiles relative to that base. So an
// init must be (re)called whenever the operand's address changes (Quasar rule: init before every op).

#include <cstdint>

#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

#ifdef TRISC_UNPACK
#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_PACK
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif

namespace llk_2_0 {

using ckernel::experimental::LLKMemDescriptor;

// Register format == L1 format holds for the 16-bit float formats only (see the header comment).
constexpr bool is_supported_format(DataFormat f) { return f == DataFormat::Float16_b || f == DataFormat::Float16; }

#ifdef TRISC_UNPACK
/** @brief Program the SrcA/SrcB register formats from the operand descriptors (once per kernel). */
template <LLKMemDescriptor A, LLKMemDescriptor B>
inline void unpack_hw_configure() {
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(A.format, B.format);
}

/**
 * @brief Allocate + program an Unp0 BFD at `l1_address` and build the one-tile unpack-to-SrcA MOP.
 * @param l1_address: Operand L1 base address (16B units); tile 0 of the following unpack_A calls.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
inline void unpack_A_init(const std::uint32_t l1_address) {
    const std::uint8_t bfd_id = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        D.shape, l1_address, static_cast<std::uint32_t>(D.format));
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*TRANSPOSE_EN*/, EN_32BIT_DEST>(bfd_id, D.shape, 1);
}

/** @brief Unpack tile `tile_index` (relative to the unpack_A_init address) into SrcA. */
template <LLKMemDescriptor D>
inline void unpack_A(const std::uint32_t tile_index) {
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(tile_index, D.shape);
}
#endif

#ifdef TRISC_MATH
/** @brief Program the ALU source formats from the operand descriptors (once per kernel). */
template <LLKMemDescriptor A, LLKMemDescriptor B, bool EN_32BIT_DEST>
inline void math_hw_configure() {
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(A.format, B.format);
}

/** @brief Build the SrcA -> Dest datacopy MOP for one tile of shape D.shape. */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
inline void math_datacopy_init() {
    _configure_default_alu_data_format_state_<false /*IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(D.format, D.format);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, EN_32BIT_DEST>(
        D.shape.total_num_faces() * D.shape.face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/** @brief Copy the SrcA tile into Dest tile `dst_index`. */
inline void math_datacopy(const std::uint32_t dst_index) { _llk_math_eltwise_unary_datacopy_(dst_index); }
#endif

#ifdef TRISC_PACK
/** @brief Program the packer input (Dest) format from the output descriptor (once per kernel). */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
inline void pack_hw_configure() {
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(D.format, ckernel::ReluConfig::none());
}

/**
 * @brief Allocate + program a Pack0 BFD at `l1_address` and build the one-tile pack MOP.
 * @param l1_address: Output L1 base address (16B units); tile 0 of the following pack calls.
 */
template <LLKMemDescriptor D>
inline void pack_init(const std::uint32_t l1_address) {
    const std::uint8_t bfd_id = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        D.shape, l1_address, static_cast<std::uint32_t>(D.format));
    _llk_pack_init_(bfd_id, D.shape, 1);
}

/** @brief Pack Dest tile `dst_index` to output tile `tile_index` (relative to the pack_init address). */
template <LLKMemDescriptor D>
inline void pack(const std::uint32_t dst_index, const std::uint32_t tile_index) {
    _llk_pack_(dst_index, tile_index, D.shape);
}
#endif

}  // namespace llk_2_0
