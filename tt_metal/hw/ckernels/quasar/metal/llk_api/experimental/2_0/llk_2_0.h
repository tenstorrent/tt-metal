// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Id-free (2.0) Quasar LLK API: copy (FPU datacopy and unpack-to-dest), tilize, eltwise binary and broadcast,
// matmul, pack, pack untilize.
// New functions only -- the legacy DFB-id LLK API (llk_*_api.h) is untouched. Every function takes the operand's
// LLKMemDescriptor (L1 format + tile shape) as an NTTP and, where the op reads/writes L1, a buffer-descriptor base
// (16B units) at init and L1 tile indices relative to it at execute. No DFB id, no host format tables.
//
// Init functions are noinline (one copy per descriptor set) and program ALL state their op reads: buffer descriptor
// (fixed id per TDMA engine, ckernel_buf_desc.h), register formats, MOP. So no reconfig and no uninit:
// on Quasar every op is re-initialised before it runs anyway. Execute functions and the once-per-kernel hw_configure
// setups (24-52 B) are plain inline.
//
// Register formats are the L1 format (no Quasar format-derivation table exists on device yet), so the compute layer
// restricts 2.0 operands to 16-bit float formats, plus Float32 where it never enters SrcA/SrcB: unpack-to-dest copy and
// pack, at a 32-bit Dest. With a 32-bit Dest the packer reads Float32.

#include <cstdint>

#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "ckernel_buf_desc.h"
#include "llk_tdma_guard.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_binary_broadcast_operands.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "llk_unpack_matmul.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_broadcast_operands.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "llk_unpack_tilize.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_binary_broadcast.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"
#include "llk_math_unary_broadcast.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#endif

namespace llk_2_0 {

using ckernel::experimental::LLKMemDescriptor;

// Register format == L1 format holds for the 16-bit float formats only (see the header comment).
constexpr bool is_supported_format(DataFormat f) { return f == DataFormat::Float16_b || f == DataFormat::Float16; }

// Copy (unpack-to-dest) and pack operands also take Float32 at a 32-bit Dest: it never enters SrcA/SrcB.
constexpr bool is_supported_format(DataFormat f, bool en_32bit_dest) {
    return is_supported_format(f) || (en_32bit_dest && f == DataFormat::Float32);
}

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
inline void unpack_hw_configure() {
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
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    constexpr std::uint8_t bfd = ckernel::trisc::BFD_ID_UNP0;
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
        "copy: a Float32 source at a 32-bit Dest takes the unpack-to-dest path, whose Dest placer steps whole 32x32 "
        "tiles, so it supports only 32x32 tiles. Use a 32x32 tile, or a 16-bit L1 format (FPU datacopy path).");
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    constexpr std::uint8_t bfd = ckernel::trisc::BFD_ID_UNP0;
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
 * @tparam BCAST: SrcB broadcast (NONE / COL / ROW / SCALAR).
 * @param bd_a, bd_b: Buffer-descriptor bases (16B units) of A and B.
 */
template <LLKMemDescriptor DA, LLKMemDescriptor DB, BroadcastType BCAST = BroadcastType::NONE>
LLK_2_0_NOINLINE void unpack_AB_init(const std::uint32_t bd_a, const std::uint32_t bd_b) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(DA.shape, bd_a, static_cast<std::uint32_t>(DA.format));
    constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP0;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(DB.shape, bd_b, static_cast<std::uint32_t>(DB.format));
    constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP1;
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(DA.format, DB.format);
    if constexpr (BCAST == BroadcastType::NONE) {
        _llk_unpack_binary_operands_init_(bfd_a, bfd_b, DA.shape, 1);
    } else {
        _llk_unpack_binary_broadcast_operands_init_<BCAST>(bfd_a, bfd_b, DA.shape, 1);
    }
}

/** @brief Unpack A tile `tile_a` -> SrcA and B tile `tile_b` -> SrcB (indices relative to the init's bases). */
template <LLKMemDescriptor DA, BroadcastType BCAST = BroadcastType::NONE>
inline void unpack_AB(
    const std::uint32_t bd_a, const std::uint32_t bd_b, const std::uint32_t tile_a, const std::uint32_t tile_b) {
    tdma_guard_note_bd(bd_a);
    tdma_guard_note_bd(bd_b);
    if constexpr (BCAST == BroadcastType::NONE) {
        _llk_unpack_binary_operands_(tile_a, tile_b, DA.shape);
    } else {
        _llk_unpack_binary_broadcast_operands_(tile_a, tile_b, DA.shape);
    }
}

/**
 * @brief Init for unpacking one operand to SrcB with a broadcast (Unp1): descriptor at `bd_base`, SrcB format, MOP.
 * Full 32x32 tiles only (the broadcast unpacker indexes L1 in whole tiles).
 * @tparam D: Operand descriptor.
 * @tparam BCAST: Broadcast (COL / ROW / SCALAR).
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 * @param bd_base: Buffer-descriptor base (16B units); tile indices of @ref unpack_B_bcast are relative to it.
 */
template <LLKMemDescriptor D, BroadcastType BCAST, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void unpack_B_bcast_init(const std::uint32_t bd_base) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_B>(D.format);
    _llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, BCAST, EN_32BIT_DEST>(ckernel::trisc::BFD_ID_UNP1, 1);
}

/** @brief Unpack L1 tile `l1_tile` (relative to the init's base `bd_base`) into SrcB, broadcast. */
inline void unpack_B_bcast(const std::uint32_t bd_base, const std::uint32_t l1_tile) {
    tdma_guard_note_bd(bd_base);
    _llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_B>(l1_tile);
}

/**
 * @brief Init for matmul unpack, in0 -> SrcB (Unp1), in1 -> SrcA (Unp0): descriptors, formats, MOP for a
 * [rt_dim, ct_dim] output block.
 * @tparam D0, D1: Descriptors of in0 and in1.
 * @tparam TRANSPOSE: Transpose each in1 (SrcA) tile; needs a full four-face in1.
 * @param bd0, bd1: Buffer-descriptor bases (16B units) of in0 and in1.
 * @param ct_dim, rt_dim, kt_dim: Output block columns / rows and the inner dim.
 */
template <LLKMemDescriptor D0, LLKMemDescriptor D1, bool TRANSPOSE>
LLK_2_0_NOINLINE void unpack_matmul_init(
    const std::uint32_t bd0,
    const std::uint32_t bd1,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(D0.shape, bd0, static_cast<std::uint32_t>(D0.format));
    constexpr std::uint8_t bfd0 = ckernel::trisc::BFD_ID_UNP1;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(D1.shape, bd1, static_cast<std::uint32_t>(D1.format));
    constexpr std::uint8_t bfd1 = ckernel::trisc::BFD_ID_UNP0;
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(D1.format, D0.format);
    _llk_unpack_matmul_init_<TRANSPOSE>(bfd0, bfd1, ct_dim, rt_dim, kt_dim, D0.shape, D1.shape);
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

/**
 * @brief Init for unpack-tilize of a row-major block to SrcA: Unp0 descriptor at `bd_base`, SrcA format, tilize
 * strides and the MOP for `block_ct_dim` tiles (one SrcA tile per datacopy). 32x32 tiles only.
 * @tparam D: Operand descriptor (the shape of one output tile).
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 * @param bd_base: Buffer-descriptor base (16B units); rows of @ref unpack_tilize are relative to it.
 * @param block_ct_dim: Block width in tiles; also the row-major row width.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void unpack_tilize_init(const std::uint32_t bd_base, const std::uint32_t block_ct_dim) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    constexpr std::uint8_t bfd = ckernel::trisc::BFD_ID_UNP0;
    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(D.format);
    _llk_unpack_tilize_init_<p_unpacr::UNP_A, EN_32BIT_DEST>(bfd, block_ct_dim, block_ct_dim, D.shape);
}

/**
 * @brief Unpack-tilize `block_ct_dim` tiles (as the init) to SrcA, one tile per SrcA bank, starting at L1 row
 * `l1_row` (rows of 32 datums, relative to the init's base `bd_base`).
 */
template <LLKMemDescriptor D>
inline void unpack_tilize(const std::uint32_t bd_base, const std::uint32_t l1_row, const std::uint32_t block_ct_dim) {
    tdma_guard_note_bd(bd_base);
    _llk_unpack_tilize_<p_unpacr::UNP_A>(l1_row, block_ct_dim, D.shape);
}
#endif

#ifdef TRISC_MATH
/**
 * @brief Kernel-wide math setup: Dest mode (16/32-bit), ALU formats, and the legacy MATH/PACK Dest semaphores
 * (llk_math_pack_sync_init).
 * @tparam A, B: Descriptors of the SrcA / SrcB operands.
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 */
template <LLKMemDescriptor A, LLKMemDescriptor B, bool EN_32BIT_DEST>
inline void math_hw_configure() {  // once per kernel: inline
    _llk_math_srcAB_hw_configure_<EN_32BIT_DEST>(A.format, B.format);
    _llk_math_pack_sync_init_<DST_SYNC_MODE>();
}

/**
 * @brief Per-op ALU formats, from the op init's operand formats: the format part of _llk_math_srcAB_hw_configure_.
 * Its kernel-wide part (TTSync enables, implied SrcA/SrcB format disables) ran once in math_hw_configure.
 */
template <bool EN_32BIT_DEST>
inline void math_alu_formats(const DataFormat srca, const DataFormat srcb) {
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);
    _configure_alu_cfg_<AluCfgType::DEFAULT, EN_32BIT_DEST>(srca, srcb);
}

/** @brief Init for SrcA -> Dest datacopy of one tile of shape D.shape (ALU formats + MOP). */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void math_datacopy_init() {
    math_alu_formats<EN_32BIT_DEST>(D.format, D.format);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, EN_32BIT_DEST>(
        D.shape.total_num_faces() * D.shape.face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/** @brief Copy the SrcA tile into Dest tile `dst_index`. */
inline void math_datacopy(const std::uint32_t dst_index) { _llk_math_eltwise_unary_datacopy_(dst_index); }

/**
 * @brief Init for Dest = SrcA [op] SrcB (ALU formats + MOP); mul keeps `FIDELITY`, add/sub run LoFi.
 * @tparam BCAST: SrcB broadcast (NONE / COL / ROW / SCALAR), same as the unpack init.
 * @param acc_to_dest: Accumulate into Dest instead of overwriting it.
 */
template <
    EltwiseBinaryType TYPE,
    LLKMemDescriptor DA,
    LLKMemDescriptor DB,
    bool EN_32BIT_DEST,
    ckernel::MathFidelity FIDELITY,
    BroadcastType BCAST = BroadcastType::NONE>
LLK_2_0_NOINLINE void math_binary_init(const bool acc_to_dest) {
    constexpr ckernel::MathFidelity fidelity =
        TYPE == EltwiseBinaryType::ELWMUL ? FIDELITY : ckernel::MathFidelity::LoFi;
    math_alu_formats<EN_32BIT_DEST>(DA.format, DB.format);
    if constexpr (BCAST == BroadcastType::NONE) {
        _llk_math_eltwise_binary_init_<TYPE, fidelity>(DA.format, DB.format, DA.shape, acc_to_dest);
    } else {
        _llk_math_eltwise_binary_broadcast_init_<TYPE, BCAST, fidelity>(DA.format, DB.format, DA.shape, acc_to_dest);
    }
}

/** @brief Dest tile `dst_index` = SrcA [op] SrcB (broadcast per BCAST, same as the init). */
template <EltwiseBinaryType TYPE, LLKMemDescriptor DA, BroadcastType BCAST = BroadcastType::NONE>
inline void math_binary(const std::uint32_t dst_index) {
    if constexpr (BCAST == BroadcastType::NONE) {
        _llk_math_eltwise_binary_<TYPE>(dst_index, DA.shape);
    } else {
        _llk_math_eltwise_binary_broadcast_(dst_index);
    }
}

/** @brief Init for the broadcast SrcB -> Dest copy of one 32x32 tile (ALU formats + MOP). */
template <LLKMemDescriptor D, BroadcastType BCAST, bool EN_32BIT_DEST>
LLK_2_0_NOINLINE void math_unary_bcast_init() {
    math_alu_formats<EN_32BIT_DEST>(D.format, D.format);
    _llk_math_eltwise_unary_broadcast_init_<BCAST, EN_32BIT_DEST, false /*unpack_to_dest*/>(D.shape);
}

/** @brief Copy the broadcast SrcB tile into Dest tile `dst_index`. */
inline void math_unary_bcast(const std::uint32_t dst_index) {
    _llk_math_eltwise_unary_broadcast_<false /*unpack_to_dest*/>(dst_index);
}

/**
 * @brief Init for matmul Dest += SrcB(in0) x SrcA(in1) over a [rt_dim, ct_dim] block (ALU formats + MOP).
 */
template <LLKMemDescriptor D0, LLKMemDescriptor D1, bool EN_32BIT_DEST, ckernel::MathFidelity FIDELITY>
LLK_2_0_NOINLINE void math_matmul_init(const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    math_alu_formats<EN_32BIT_DEST>(D1.format, D0.format);
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
 * @brief Kernel-wide pack setup: packer input format and Dest bank 0 (llk_pack_dest_init).
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
inline void pack_hw_configure() {
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), ckernel::ReluConfig::none());
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();
}

/**
 * @brief Init for packing one tile: Pack0 descriptor at `bd_base`, packer input format, ReLU (zero) or none, L1
 * accumulation off, one-tile MOP.
 * @tparam RELU: true = packer ReLU (max(x, 0)), false = none.
 * @param bd_base: Buffer-descriptor base (16B units); tile indices of @ref pack are relative to it.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST, bool RELU = false>
LLK_2_0_NOINLINE void pack_init(const std::uint32_t bd_base) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    constexpr std::uint8_t bfd = ckernel::trisc::BFD_ID_PACK0;
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), RELU ? ckernel::ReluConfig::zero() : ckernel::ReluConfig::none());
    _llk_pack_set_l1_acc_<p_pacr::PACK0>(false);
    _llk_pack_init_(bfd, D.shape, 1);
}

/** @brief Pack Dest tile `dst_index` to L1 tile `l1_tile` (relative to the init's base `bd_base`). */
template <LLKMemDescriptor D>
inline void pack(const std::uint32_t bd_base, const std::uint32_t dst_index, const std::uint32_t l1_tile) {
    tdma_guard_note_bd(bd_base);
    _llk_pack_(dst_index, l1_tile, D.shape);
}

/**
 * @brief Init for pack untilize: Pack0 descriptor at `bd_base`, packer input format, no ReLU, L1 accumulation off,
 * untilize strides and MOP for BLOCK_CT Dest tiles into a row-major buffer whose rows are FULL_CT tiles wide. 32x32
 * tiles only.
 * @tparam D: Output descriptor.
 * @param bd_base: Buffer-descriptor base (16B units); rows of @ref pack_untilize are relative to it.
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST, std::uint32_t BLOCK_CT, std::uint32_t FULL_CT>
LLK_2_0_NOINLINE void pack_untilize_init(const std::uint32_t bd_base) {
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(D.shape, bd_base, static_cast<std::uint32_t>(D.format));
    constexpr std::uint8_t bfd = ckernel::trisc::BFD_ID_PACK0;
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), ckernel::ReluConfig::none());
    _llk_pack_set_l1_acc_<p_pacr::PACK0>(false);
    _llk_pack_untilize_init_<FULL_CT, BLOCK_CT>(bfd, D.shape);
}

/**
 * @brief Pack-untilize Dest tiles dst_index .. dst_index + BLOCK_CT - 1 (BLOCK_CT as the init) into the row-major
 * buffer, starting at L1 row `l1_row` (rows of 32 datums, relative to the init's base `bd_base`).
 */
template <LLKMemDescriptor D>
inline void pack_untilize(const std::uint32_t bd_base, const std::uint32_t dst_index, const std::uint32_t l1_row) {
    tdma_guard_note_bd(bd_base);
    _llk_pack_untilize_(dst_index, l1_row, D.shape);
}

/** @brief Packer L1 accumulation on/off (the one runtime reconfig of the 2.0 API). */
inline void pack_set_l1_acc(const bool enable) { _llk_pack_set_l1_acc_<p_pacr::PACK0>(enable); }
#endif

#undef LLK_2_0_NOINLINE

}  // namespace llk_2_0
