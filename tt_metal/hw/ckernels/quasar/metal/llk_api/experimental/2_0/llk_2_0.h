// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Id-free (2.0) Quasar LLK API: copy (FPU datacopy and unpack-to-dest), eltwise binary, matmul, pack.
// New functions only -- the legacy DFB-id LLK API (llk_*_api.h) is untouched. Every function takes the operand's
// LLKMemDescriptor (L1 format + tile shape) as an NTTP and, where the op reads/writes L1, a buffer-descriptor base
// (16B units) at init and L1 tile indices relative to it at execute. No DFB id, no host format tables.
//
// Init functions are noinline (one copy per descriptor set) and program ALL state their op reads: buffer descriptor
// (allocated from this TRISC's partition, llk_bfd_alloc.h), register formats, MOP. So no reconfig and no uninit:
// on Quasar every op is re-initialised before it runs anyway. Execute functions are inline.
//
// Register formats are the L1 format (no Quasar format-derivation table exists on device yet), so the compute layer
// restricts 2.0 operands to 16-bit float formats. With a 32-bit Dest the packer reads Float32.

#include <cstdint>

#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "llk_bfd_alloc.h"
#include "llk_tdma_guard.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "llk_unpack_matmul.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif

namespace llk_2_0 {

using ckernel::experimental::LLKMemDescriptor;

// Register format == L1 format holds for the 16-bit float formats only (see the header comment).
constexpr bool is_supported_format(DataFormat f) { return f == DataFormat::Float16_b || f == DataFormat::Float16; }

#define LLK_2_0_NOINLINE __attribute__((noinline))

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
/**
 * @brief TEN-4746 guard for id-free ops (debug only): a TDMA on the buffer descriptor based at `bd_base` disarms every
 * waited/reserved DFB of this thread whose ring base is `bd_base`. The guard arms per DFB id in the wait/reserve and
 * checks it in the pop/push; an id-free op knows only its descriptor, and a DFB's descriptor is based at its ring
 * base (tc_slots[0]), so the base address is the key that joins the two.
 * @param bd_base: Buffer-descriptor base (16B units) of the descriptor the TDMA used.
 */
inline void tdma_guard_note_bd([[maybe_unused]] const std::uint32_t bd_base) {
#if defined(ENABLE_LLK_ASSERT_ONLY) || defined(ENABLE_LLK_ASSERT)
    std::uint32_t armed = llk_tdma_guard::armed_mask();
    for (std::uint32_t dfb = 0; armed != 0; ++dfb, armed >>= 1) {
        if ((armed & 1) && get_local_dfb_interface(dfb).tc_slots[0].base_addr == bd_base) {
            llk_tdma_guard::note_tdma(dfb);
        }
    }
#endif
}
#endif

#ifdef TRISC_UNPACK
/**
 * @brief Kernel-wide unpack setup: SrcA/SrcB register formats and the unpack-to-dest bank offset reset.
 * @tparam A, B: Descriptors of the operands unpacked to SrcA / SrcB.
 */
template <LLKMemDescriptor A, LLKMemDescriptor B>
LLK_2_0_NOINLINE void unpack_hw_configure() {
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(A.format, B.format);
    _llk_unpack_dest_init_();
}

/**
 * @brief Init for unpacking one operand to SrcA: Unp0 descriptor at `bd_base`, SrcA format, one-tile MOP.
 * @tparam D: Operand descriptor.
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 * @param bd_base: Buffer-descriptor base (16B units); tile indices of @ref unpack_A are relative to it.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void unpack_A_init(const std::uint32_t bd_base) {
    const std::uint8_t bfd = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(D.format);
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*TRANSPOSE_EN*/, EN_32BIT_DEST>(bfd, D.shape, 1);
}

/**
 * @brief Unpack L1 tile `l1_tile` (relative to the init's base `bd_base`) into SrcA.
 * @tparam D: Operand descriptor, same as the init.
 */
template <LLKMemDescriptor D>
inline void unpack_A(const std::uint32_t bd_base, const std::uint32_t l1_tile) {
    tdma_guard_note_bd(bd_base);
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile, D.shape);
}

/**
 * @brief Init for unpacking one operand straight to Dest: Unp0 descriptor at `bd_base`, UNP_DEST MOP. Full 32x32
 * tiles only (the to-dest placer steps Dest in whole tiles).
 * @tparam D: Operand descriptor.
 * @param bd_base: Buffer-descriptor base (16B units); tile indices of @ref unpack_to_dest are relative to it.
 */
template <LLKMemDescriptor D>
LLK_2_0_NOINLINE void unpack_to_dest_init(const std::uint32_t bd_base) {
    static_assert(
        D.shape.face_r_dim == ckernel::MAX_FACE_R_DIM && D.shape.num_faces_r_dim == ckernel::MAX_NUM_FACES_R_DIM &&
            D.shape.num_faces_c_dim == ckernel::MAX_NUM_FACES_C_DIM,
        "unpack-to-dest supports only 32x32 tiles");
    const std::uint8_t bfd = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(D.format);
    _llk_unpack_unary_operand_to_dest_init_(bfd);
}

/** @brief Unpack L1 tile `l1_tile` (relative to the init's base `bd_base`) into Dest tile `dst_index`. */
inline void unpack_to_dest(const std::uint32_t bd_base, const std::uint32_t l1_tile, const std::uint32_t dst_index) {
    tdma_guard_note_bd(bd_base);
    _llk_unpack_unary_operand_to_dest_tile_(l1_tile, dst_index);
}

/**
 * @brief Init for unpacking two operands, A -> SrcA (Unp0) and B -> SrcB (Unp1): descriptors, formats, MOP.
 * @tparam DA, DB: Operand descriptors (same tile shape).
 * @param bd_a, bd_b: Buffer-descriptor bases (16B units) of A and B.
 */
template <LLKMemDescriptor DA, LLKMemDescriptor DB>
LLK_2_0_NOINLINE void unpack_AB_init(const std::uint32_t bd_a, const std::uint32_t bd_b) {
    const std::uint8_t bfd_a = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        DA.shape, bd_a, static_cast<std::uint32_t>(DA.format));
    const std::uint8_t bfd_b = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
        DB.shape, bd_b, static_cast<std::uint32_t>(DB.format));
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(DA.format, DB.format);
    _llk_unpack_binary_operands_init_(bfd_a, bfd_b, DA.shape, 1);
}

/** @brief Unpack A tile `tile_a` -> SrcA and B tile `tile_b` -> SrcB (indices relative to the init's bases). */
template <LLKMemDescriptor DA>
inline void unpack_AB(
    const std::uint32_t bd_a, const std::uint32_t bd_b, const std::uint32_t tile_a, const std::uint32_t tile_b) {
    tdma_guard_note_bd(bd_a);
    tdma_guard_note_bd(bd_b);
    _llk_unpack_binary_operands_(tile_a, tile_b, DA.shape);
}

/**
 * @brief Init for matmul unpack, in0 -> SrcB (Unp1), in1 -> SrcA (Unp0): descriptors, formats, MOP for a
 * [rt_dim, ct_dim] output block.
 * @tparam D0, D1: Descriptors of in0 and in1.
 * @param bd0, bd1: Buffer-descriptor bases (16B units) of in0 and in1.
 * @param ct_dim, rt_dim, kt_dim: Output block columns / rows and the inner dim.
 */
template <LLKMemDescriptor D0, LLKMemDescriptor D1>
LLK_2_0_NOINLINE void unpack_matmul_init(
    const std::uint32_t bd0,
    const std::uint32_t bd1,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    const std::uint8_t bfd0 = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
        D0.shape, bd0, static_cast<std::uint32_t>(D0.format));
    const std::uint8_t bfd1 = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        D1.shape, bd1, static_cast<std::uint32_t>(D1.format));
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(D1.format, D0.format);
    _llk_unpack_matmul_init_<false /*TRANSPOSE_EN*/>(bfd0, bfd1, ct_dim, rt_dim, kt_dim, D0.shape, D1.shape);
}

/** @brief Matmul unpack of one [rt_dim, ct_dim] block step, starting at in0 tile `tile0` and in1 tile `tile1`. */
template <LLKMemDescriptor D0, LLKMemDescriptor D1>
inline void unpack_matmul(
    const std::uint32_t bd0,
    const std::uint32_t bd1,
    const std::uint32_t tile0,
    const std::uint32_t tile1,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    tdma_guard_note_bd(bd0);
    tdma_guard_note_bd(bd1);
    _llk_unpack_matmul_(ct_dim, rt_dim, kt_dim, tile0, tile1, D0.shape, D1.shape);
}
#endif

#ifdef TRISC_MATH
/**
 * @brief Kernel-wide math setup: Dest mode (16/32-bit), ALU formats, Dest section base at bank 0 (the dvalid chain
 * moves banks in hardware).
 * @tparam A, B: Descriptors of the SrcA / SrcB operands.
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 */
template <LLKMemDescriptor A, LLKMemDescriptor B, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void math_hw_configure() {
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(A.format, B.format);
    _reset_dest_register_offset_();
    _set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());
}

/** @brief Init for SrcA -> Dest datacopy of one tile of shape D.shape (ALU formats + MOP). */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void math_datacopy_init() {
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(D.format, D.format);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, EN_32BIT_DEST>(
        D.shape.total_num_faces() * D.shape.face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/** @brief Copy the SrcA tile into Dest tile `dst_index`. */
inline void math_datacopy(const std::uint32_t dst_index) { _llk_math_eltwise_unary_datacopy_(dst_index); }

/**
 * @brief Init for Dest = SrcA [op] SrcB (ALU formats + MOP); mul keeps `FIDELITY`, add/sub run LoFi.
 * @param acc_to_dest: Accumulate into Dest instead of overwriting it.
 */
template <
    EltwiseBinaryType TYPE,
    LLKMemDescriptor DA,
    LLKMemDescriptor DB,
    bool EN_32BIT_DEST,
    ckernel::MathFidelity FIDELITY>
LLK_2_0_NOINLINE void math_binary_init(const bool acc_to_dest) {
    constexpr ckernel::MathFidelity fidelity =
        TYPE == EltwiseBinaryType::ELWMUL ? FIDELITY : ckernel::MathFidelity::LoFi;
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(DA.format, DB.format);
    _llk_math_eltwise_binary_init_<TYPE, fidelity>(DA.format, DB.format, DA.shape, acc_to_dest);
}

/** @brief Dest tile `dst_index` = SrcA [op] SrcB. */
template <EltwiseBinaryType TYPE, LLKMemDescriptor DA>
inline void math_binary(const std::uint32_t dst_index) {
    _llk_math_eltwise_binary_<TYPE>(dst_index, DA.shape);
}

/**
 * @brief Init for matmul Dest += SrcB(in0) x SrcA(in1) over a [rt_dim, ct_dim] block (ALU formats + MOP).
 */
template <LLKMemDescriptor D0, LLKMemDescriptor D1, bool EN_32BIT_DEST, ckernel::MathFidelity FIDELITY>
LLK_2_0_NOINLINE void math_matmul_init(const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, EN_32BIT_DEST>(D1.format, D0.format);
    _llk_math_matmul_init_<FIDELITY>(D1.format, D0.format, ct_dim, rt_dim, D0.shape, D1.shape);
}

/** @brief One-tile matmul into Dest tile `dst_index`. */
inline void math_matmul_tile(const std::uint32_t dst_index) { _llk_math_matmul_tile_(dst_index); }

/** @brief Block matmul step into Dest tiles [0, ct_dim * rt_dim). */
inline void math_matmul_block(const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    _llk_math_matmul_block_(ct_dim, rt_dim);
}
#endif

#ifdef TRISC_PACK
/** @brief Packer input (Dest) format: Float32 with a 32-bit Dest, else the L1 format. */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
constexpr DataFormat pack_src_format() {
    return EN_32BIT_DEST ? DataFormat::Float32 : D.format;
}

/**
 * @brief Kernel-wide pack setup: packer input format and Dest bank 0 (the dvalid chain moves banks in hardware).
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void pack_hw_configure() {
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), ckernel::ReluConfig::none());
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();
}

/**
 * @brief Init for packing one tile: Pack0 descriptor at `bd_base`, packer input format, no ReLU, L1 accumulation
 * off, one-tile MOP.
 * @param bd_base: Buffer-descriptor base (16B units); tile indices of @ref pack are relative to it.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void pack_init(const std::uint32_t bd_base) {
    const std::uint8_t bfd = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), ckernel::ReluConfig::none());
    _llk_pack_set_l1_acc_<p_pacr::PACK0>(false);
    _llk_pack_init_(bfd, D.shape, 1);
}

/** @brief Pack Dest tile `dst_index` to L1 tile `l1_tile` (relative to the init's base `bd_base`). */
template <LLKMemDescriptor D>
inline void pack(const std::uint32_t bd_base, const std::uint32_t dst_index, const std::uint32_t l1_tile) {
    tdma_guard_note_bd(bd_base);
    _llk_pack_(dst_index, l1_tile, D.shape);
}

/** @brief Packer L1 accumulation on/off (the one runtime reconfig of the 2.0 API). */
inline void pack_set_l1_acc(const bool enable) { _llk_pack_set_l1_acc_<p_pacr::PACK0>(enable); }
#endif

#undef LLK_2_0_NOINLINE

}  // namespace llk_2_0
