// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "llk_sync.h"
#include "api/debug/waypoint.h"

namespace llk_math_detail {
template <auto...>
inline constexpr bool always_false_v = false;
}  // namespace llk_math_detail

// Defined in the Quasar TRISC firmware (trisc.cc); reset per program in llk_math_pack_sync_init.
extern thread_local bool unpack_math_seeded;

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/

/**
 *
 * @brief Configures math hardware.
 * Sets up ALU formats for math destination register and source registers.
 *
 * @tparam EN_32BIT_DEST_FORMAT: Set to true to use 32bit math dest in Float32 or Int32 format
 * @param srca_operand: The srcA input operand circular buffer, used to infer the srcA data_format
 * @param srcb_operand: The srcB input operand circular buffer, used to infer the srcB data_format
 */
template <bool EN_32BIT_DEST_FORMAT>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    const std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);

    const DataFormat srca_format = static_cast<DataFormat>(unpack_dst_format[srca_operand_id]);
    const DataFormat srcb_format = static_cast<DataFormat>(unpack_dst_format[srcb_operand_id]);

    _llk_math_srcAB_hw_configure_<EN_32BIT_DEST_FORMAT>(srca_format, srcb_format);
}

inline void llk_math_reconfig_remap(const bool /*remap_enable*/) {}

/**
 * @brief Returns the effective math fidelity for an eltwise binary operation.
 * Math fidelity only applies to ELWMUL; for all other binary ops (ELWADD/ELWSUB), LoFi is used.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam math_fidelity: The requested math fidelity
 * @return The requested math_fidelity for ELWMUL, MathFidelity::LoFi otherwise.
 */
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity>
inline constexpr MathFidelity get_effective_math_fidelity() {
    return (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? math_fidelity : MathFidelity::LoFi;
}

/**
 * @brief Sets the dest dvalid for FPU/SFPU
 *
 * @tparam SET_DEST_DVALID: which client to set data valid for, values = p_cleardvalid::FPU/SFPU
 * @tparam DST: Destination register banking mode: SyncHalf = double banked (math/pack overlap), SyncFull = one bank
 *(serialized)
 *
 * @warning SYNC SCHEME: dest-dvalid. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 **/
template <std::uint8_t SET_DEST_DVALID, DstSync DST>
inline void llk_math_set_dvalid() {
    static_assert(
        llk_math_detail::always_false_v<SET_DEST_DVALID, DST>,
        "llk_math_set_dvalid belongs to the dest-dvalid sync scheme, should not be mixed with semaphores which are "
        "currently used in tt-metal.");
    _llk_math_set_dvalid_<SET_DEST_DVALID, DST>();
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on destination register using semaphores.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until destination register space is available.
 * Blocks on the MATH_PACK semaphore until the packer gets the semaphore.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_math_wait_for_dest_available() {
    _llk_math_wait_for_dest_available_();

    if constexpr (UnpackToDestEn) {
        // First section only! No wait on UNPACK_MATH may be pending when the unpacker's SEMINIT of it lands
        // and its first post follows that SEMINIT. A program with no sections should never get here.
        if (!unpack_math_seeded) {
            WAYPOINT("MHBW");
            _llk_sync_spin_until_posted_(semaphore::UNPACK_MATH);
            WAYPOINT("MHBD");
            unpack_math_seeded = true;
        }
        _llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(
            semaphore::UNPACK_MATH);
        // The UNPACK_MATH get is in llk_math_dest_section_done: the tile stays counted until MATH_PACK holds it.
    }
}

/**
 * @brief Signals that the current destination section is done.
 * After math is done, posts to the MATH_PACK semaphore so the packer can proceed; on the unpack-to-dest path it then
 * gets UNPACK_MATH, releasing the unpacker's count for this section. SyncHalf: flips this thread's section base in
 * both paths, so under unpack-to-dest the SFPU follows the bank the unpacker wrote.
 * @tparam EN_32BIT_DEST: Set to true to use 32bit math dest in Float32 or Int32 format
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
template <bool EN_32BIT_DEST>
inline void llk_math_dest_section_done() {
    // Always post MATH_PACK, the math thread is in the chain for every op, including the
    // no-real-work unpack-to-dest forwarder.
    _llk_sync_post_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
    if constexpr (UnpackToDestEn) {
        // Release the unpacker's count for this section. This comes after the MATH_PACK post on purpose: the section
        // is then always held by one of the two semaphores, from the unpacker's write until pack releases it.
        // No stall before the get: the post above already waited for MATH and SFPU, and nothing ran in between.
        _llk_sync_get_(semaphore::UNPACK_MATH);
    }
    if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
        _llk_sync_advance_dest_section_<ckernel::TRISC_ID, EN_32BIT_DEST, p_stall::WAIT_SFPU, p_stall::MATH>();
    }
}

/**
 * @brief Initializes math–pack synchronization for the destination register.
 * Waits for any previous packs to finish, resets the dest bank id, initializes the MATH_PACK semaphore. In the
 * unpack-to-dest path it also re-arms the first-section spin in @ref llk_math_wait_for_dest_available.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<DST_SYNC_MODE>();

    if constexpr (UnpackToDestEn) {
        unpack_math_seeded = false;
    }
}

// Implied SrcA/SrcB formats are disabled on Quasar (_llk_math_srcAB_hw_configure_), so the ALU reads its source
// formats from ALU_FORMAT_SPEC_REG*, not from what the unpacker wrote. hw_configure programs them for the startup
// operands and the op inits keep them (_configure_default_alu_data_format_state_ returns early once DEFAULT is
// latched), so a reconfig has to reprogram them here; otherwise e.g. a Tf32 operand is decoded with the startup
// operand's Float16_b format, and the init's fidelity schedule (derived from Tf32) runs phases a Float16_b SrcA lacks.
// The ALU format words cannot be read back, so they are rewritten whole: the side that does not change comes from
// alu_src_format_state, and the result is the DEFAULT config shape (an op that needs a special shape, such as a
// Dstacc override, re-establishes it at its init).
namespace llk_math_detail {
template <bool EN_32BIT_DEST, bool skip_int8>
inline void set_alu_src_formats(const DataFormat srca_format, const DataFormat srcb_format) {
    const bool en_int32_dest_format = skip_int8 ? alu_src_format_state.en_int32_dest_format
                                                : (_is_src_fmt_int32_dest_compatible_(srca_format) &&
                                                   _is_src_fmt_int32_dest_compatible_(srcb_format) && EN_32BIT_DEST);
    _configure_alu_formats_<EN_32BIT_DEST>(srca_format, srcb_format, en_int32_dest_format, DataFormat::Invalid);
    data_format_config_set = DataFormatConfigSet::DEFAULT;
}

inline DataFormat operand_src_reg_format(const std::uint32_t operand) {
    return static_cast<DataFormat>(unpack_dst_format[get_operand_id(operand)]);
}
}  // namespace llk_math_detail

template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    llk_math_detail::set_alu_src_formats<EN_32BIT_DEST, skip_int8>(
        llk_math_detail::operand_src_reg_format(srca_new_operand), alu_src_format_state.srcb_format);
}

template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    llk_math_detail::set_alu_src_formats<EN_32BIT_DEST, skip_int8>(
        alu_src_format_state.srca_format, llk_math_detail::operand_src_reg_format(srcb_new_operand));
}

template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_math_detail::set_alu_src_formats<EN_32BIT_DEST, skip_int8>(
        llk_math_detail::operand_src_reg_format(srca_new_operand),
        llk_math_detail::operand_src_reg_format(srcb_new_operand));
}

// The (old, new) overloads skip a side whose old and new operands share a source register format.
template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    if (llk_math_detail::operand_src_reg_format(srca_old_operand) !=
        llk_math_detail::operand_src_reg_format(srca_new_operand)) {
        llk_math_reconfig_data_format_srca<EN_32BIT_DEST, skip_int8>(srca_new_operand);
    }
}

template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    if (llk_math_detail::operand_src_reg_format(srcb_old_operand) !=
        llk_math_detail::operand_src_reg_format(srcb_new_operand)) {
        llk_math_reconfig_data_format_srcb<EN_32BIT_DEST, skip_int8>(srcb_new_operand);
    }
}

template <bool EN_32BIT_DEST, bool skip_int8 = false>
inline void llk_math_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    const bool reconfig_a = llk_math_detail::operand_src_reg_format(srca_old_operand) !=
                            llk_math_detail::operand_src_reg_format(srca_new_operand);
    const bool reconfig_b = llk_math_detail::operand_src_reg_format(srcb_old_operand) !=
                            llk_math_detail::operand_src_reg_format(srcb_new_operand);
    if (reconfig_a && reconfig_b) {
        llk_math_reconfig_data_format<EN_32BIT_DEST, skip_int8>(srca_new_operand, srcb_new_operand);
    } else if (reconfig_a) {
        llk_math_reconfig_data_format_srca<EN_32BIT_DEST, skip_int8>(srca_new_operand);
    } else if (reconfig_b) {
        llk_math_reconfig_data_format_srcb<EN_32BIT_DEST, skip_int8>(srcb_new_operand);
    }
}
