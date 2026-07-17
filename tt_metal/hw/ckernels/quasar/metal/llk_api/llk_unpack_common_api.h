// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_unpack_common.h"
#include "llk_sync.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * @brief Programs l1 info & source register format for both UNP_A and UNP_B
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const std::uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const std::uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id.
    // Skip non-participating DFBs via entry_size==0 (g_dfb_interface[] is zero-init,
    // so non-populated entries naturally fall out). Loop bound is dfb::NUM_DFBS because
    // g_dfb_interface[] is sized NUM_DFBS (=32) and NUM_CIRCULAR_BUFFERS resolves to 64
    // on Quasar — GCC -Werror=aggressive-loop-optimizations rejects the OOB.
    for (std::uint32_t i = 0; i < dfb::NUM_DFBS; ++i) {
        if (g_dfb_interface[i].entry_size == 0) {
            continue;
        }
        const DataFormat l1_data_format = static_cast<DataFormat>(unpack_src_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        // TODO: with multiple TCs are there multiple descriptors?
        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = get_local_dfb_interface(i).tc_slots[0].base_addr;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.y_dim = unpack_tile_face_r_dim[i];
        // [Local LLK bring-up fix — issue filed] Mirror of the pack-side fix in llk_pack_common_api.h:
        // map {nf_r,nf_c} to a valid z via construct_tdma_desc's logic instead of copying total num_faces
        // raw (z must be {1,4}). Input CBs are usually reprogrammed to z=1 by tilizeA_B_init so this side
        // is normally masked, but fix it for correctness/consistency.
        const ckernel::TensorShape unpack_ts = get_operand_tensor_shape(i);
        bd_val.f.z_dim =
            (unpack_ts.num_faces_r_dim == unpack_ts.num_faces_c_dim)
                ? unpack_ts.total_num_faces()
                : ckernel::trisc::compute_square_of_min(unpack_ts.num_faces_r_dim, unpack_ts.num_faces_c_dim);

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val_A, td_val_B;
    td_val_A.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpA_operand_id]);
    td_val_B.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpB_operand_id]);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
}

/**
 * @brief Programs l1 info & source register format for UNP_A
 *
 * @param operandA: The input operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure(unpA_operand, unpA_operand);
}

inline bool should_reconfig_src_reg_df(std::uint32_t old_operand, std::uint32_t new_operand) {
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

/**
 * @brief Signals that the unpacker has finished writing a DEST section in the unpack-to-dest (UNP_DEST) path,
 *        making it available to PACK. Used when the unpacker writes DEST directly and the math thread is
 *        bypassed (no MATH<->PACK semaphore); pairs with llk_pack_dest_dvalid_section_done on T2.
 *
 * @tparam DST: Destination register buffering mode, values = <DstSync::SyncHalf/DstSync::SyncFull>
 */
template <DstSync DST>
inline void llk_unpack_dest_dvalid_section_done() {
    _llk_unpack_dest_dvalid_section_done_<DST>();
}

/**
 * @brief Arms the UNPACK_TO_DEST -> PACK per-bank DEST-dvalid handshake on the UNPACK thread
 *        (programs UNPACK_TO_DEST_DVALID_CTRL so a refill waits for PACK to drain the bank).
 *
 * WHY THIS IS REQUIRED: UNPACR_TILIZE for the unpack-to-dest tilize is built with SET_DVALID=0 and delegates
 * ALL cross-thread ordering to the *_DEST_DVALID_CTRL wait masks. Those masks are ONLY programmed by
 * set_up_dest_dvalid_per_thread(), and the tt-metal compute stack never called it (the tilize_init math-init
 * that claimed to "set up the dvalid client scheme" actually skips it for unpack_to_dest — see
 * llk_math_unary_datacopy_api.h). With the masks unarmed there is no per-bank back-pressure, so the unpacker
 * laps the packer: fused tilize corrupts DEST data (PCC~0) and tilize-only intermittently deadlocks. Producer =
 * UNPACK, consumer = PACK (MATH is bypassed on this path). Call ONCE in init, before any section_done.
 * Mirrors tt-llk/tests/sources/quasar/unpack_tilize_quasar_test.cpp:29-30.
 */
inline void llk_unpack_setup_dest_dvalid() {
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
}

/**
 * @brief Restores UNPACK_TO_DEST_DVALID_CTRL to the no-wait default (0) — the inverse of
 *        llk_unpack_setup_dest_dvalid(). Call when the unpack-to-dest tilize is done so a FOLLOWING op in the
 *        same kernel (e.g. the fused conv's matmul, whose DEST producer is FPU and which syncs via the
 *        MATH<->PACK semaphore, NOT dvalid) is not left gated on the tilize's stale UNPACK dvalid bit.
 */
inline void llk_unpack_teardown_dest_dvalid() {
    auto cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE);
    cfg[UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
}

// =====================================================================================================
// Direct UNPACK<->PACK batched-tilize DEST double-buffer handshake (replaces the dvalid section scheme,
// which desynced because the unpack side never flipped its DEST bank id -> lapped the packer).
// UNPACK is the sole DEST producer; a single semaphore (UNPACK_MATH, reused purely as the unpack->pack
// token) bounds it to <=1 DEST bank ahead of PACK; each thread flips its OWN DEST section base in lockstep.
// MATH issues NO DEST op (no MOVA2D -> no 0x19). SEMINIT is on MATH (llk_math_tilize_dest_sync_init).
// =====================================================================================================

/** @brief UNPACK-side init: SEMINIT the UNPACK_MATH token AND point UNPACK's DEST section base at bank 0.
 *  The SEMINIT lives HERE (on the producer thread), NOT on MATH: UNPACK's acquire (SEMWAIT STALL_ON_MAX) latches
 *  the semaphore's max at issue time, so the SEMINIT (max=N) MUST be ordered before it — guaranteed only when
 *  both run on the same thread. (SEMINIT-on-MATH raced the acquire -> acquire latched max=0 -> first-section
 *  deadlock, dprint_tr10.) PACK only reads the value (starts 0 at reset), so it needs no ordering vs SEMINIT. */
template <DstSync DST>
inline void llk_unpack_tilize_dest_sync_init() {
    constexpr std::uint32_t N = (DST == DstSync::SyncFull) ? 1 : 2;  // one token per DEST bank
    _llk_sync_init_(semaphore::UNPACK_MATH, N, 0);
    if constexpr (DST == DstSync::SyncHalf) {
        _reset_dest_register_offset_();
        _set_dest_section_base_<ckernel::unpack::TRISC_ID>(_get_dest_buffer_base_());
    }
}

/** @brief Block the unpacker until a DEST bank is free (token < max). Call before filling a bank. */
inline void llk_unpack_tilize_dest_acquire() {
    _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::UNPACK_MATH);
}

/** @brief Publish the just-filled DEST bank to PACK (drain the fill first via STALLWAIT UNPACK0) and flip
 *         UNPACK to the other bank. @tparam EN_32BIT_DEST must equal the pack side's (== DST_ACCUM_MODE). */
template <DstSync DST, bool EN_32BIT_DEST>
inline void llk_unpack_tilize_dest_release() {
    _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);
    if constexpr (DST == DstSync::SyncHalf) {
        _llk_sync_advance_dest_section_<ckernel::unpack::TRISC_ID, EN_32BIT_DEST, p_stall::UNPACK0>();
    }
}

/**
 * Reprograms unpacker THCON OUT_DATA_FORMAT only (gasket); L1 format stays in buffer descriptors.
 */
template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, EN_32BIT_DEST>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id]);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    const std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, EN_32BIT_DEST>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id]);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    // Silent no-op if old/new operands already share both src and dst formats.
    if (!should_reconfig_src_reg_df(srca_old_operand, srca_new_operand)) {
        return;
    }
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srca_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    // Silent no-op if old/new operands already share both src and dst formats.
    if (!should_reconfig_src_reg_df(srcb_old_operand, srcb_new_operand)) {
        return;
    }
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srcb_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srcb_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(
        srcb_old_operand, srcb_new_operand);
}

/**
 * @brief Issues a dummy SrcB dvalid so the math thread can satisfy its SRCB_VLD
 * stall in transpose-dest. Used by the transpose_wh_dest compute API.
 */
inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcB_dummy_valid_(); }
