// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "llk_bfd_alloc.h"
#include "llk_outputs.h"
#include "llk_pack_common.h"
#include "llk_sync.h"
#include "llk_defs.h"
#include "api/dataflow/dataflow_buffer.h"

namespace llk_pack_detail {
template <auto...>
inline constexpr bool always_false_v = false;
}  // namespace llk_pack_detail

// The pack engine this TRISC owns: TRISC2 -> Pack0, TRISC3 -> Pack1. These pack headers are only
// compiled under TRISC_PACK (COMPILE_FOR_TRISC=2) today, so the Pack1 arm is forward-compat. The
// owning-TRISC guard itself lives in bfd_alloc<E>/bfd_current<E> (static_assert), so it need not be
// repeated at each use site.
inline constexpr ckernel::trisc::BfdResource pack_bfd_resource =
    (ckernel::TRISC_ID == 2) ? ckernel::trisc::BfdResource::Pack0 : ckernel::trisc::BfdResource::Pack1;

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

/**
 * @brief Allocate a BFD id from the pack partition, program its table entry from the output's
 * DFB info (shape, L1 base, formats), and record the id in the pack engine's current slot. The DFB
 * id is used only to fetch buffer info — it never doubles as the BFD id.
 *
 * @tparam MODE: L1 access mode for the descriptor; Strided collapses y/z dims to 1 for the
 * PACR_STRIDE untilize sequences.
 */
template <ckernel::trisc::L1AccessMode MODE = ckernel::trisc::L1AccessMode::Continuous>
inline void llk_pack_program_bfd(const std::uint32_t output_id) {
    // TODO: multi-TC not handled — only tc_slots[0]'s L1 base is programmed. When a DFB is mapped
    // across multiple TCs this must program one descriptor per active tc_slot (same gap in
    // llk_unpack_program_bfd). Tied to the DFB<->buffer-descriptor decouple work.
    ckernel::trisc::bfd_alloc_and_program<pack_bfd_resource, MODE>(
        get_output_tensor_shape(output_id),
        get_local_dfb_interface(output_id).tc_slots[0].base_addr,
        static_cast<std::uint32_t>(pack_dst_format[output_id]));
}

/**
 * @brief Programs packer0 math destination register format
 *
 * Buffer descriptors are no longer programmed here: BFD ids are an LLK-internal resource
 * allocated from the per-TRISC partition (see llk_bfd_alloc.h) and each op's llk_pack_*_init
 * programs its own table entry. DFB ids never double as BFD ids.
 *
 * @tparam EN_32BIT_DEST: Set to true to use 32bit Destination register mode
 * @param pack_output The output DataFlow Buffer identifier
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_hw_configure(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        static_cast<DataFormat>(pack_src_format[output_id]), ckernel::ReluConfig::none());
}

inline bool should_reconfig_pack_in_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    const std::uint32_t old_output_id = get_output_id(old_output);
    const std::uint32_t new_output_id = get_output_id(new_output);
    return (pack_src_format[old_output_id] != pack_src_format[new_output_id]) ||
           (pack_dst_format[old_output_id] != pack_dst_format[new_output_id]);
}

/**
 * Reprograms packer THCON IN_DATA_FORMAT only (gasket); L1 format stays in buffer descriptors.
 */
template <[[maybe_unused]] bool EN_32BIT_DEST>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    _llk_pack_reconfig_data_format_<p_pacr::PACK0>(pack_src_format[output_id], pack_dst_format[output_id]);
}

template <bool EN_32BIT_DEST>
inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    if (!should_reconfig_pack_in_data_format(old_output, new_output)) {
        return;
    }
    llk_pack_reconfig_data_format<EN_32BIT_DEST>(new_output);
}

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 *
 * @tparam DST: Destination register banking mode: SyncHalf = double banked (math/pack overlap), SyncFull = one bank
 *(serialized)
 * @tparam EN_32BIT_DEST: flag to show if Destination register is set to 32-bit mode
 *
 * @warning SYNC SCHEME: dest-dvalid. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 **/
template <DstSync DST, bool EN_32BIT_DEST>
inline void llk_pack_dest_dvalid_section_done() {
    static_assert(
        llk_pack_detail::always_false_v<DST, EN_32BIT_DEST>,
        "llk_pack_dest_dvalid_section_done belongs to the dest-dvalid sync scheme, should not be mixed with "
        "semaphores which are currently used in tt-metal.");
    _llk_pack_dest_dvalid_section_done_<DST, EN_32BIT_DEST>();
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on the destination register using semaphores.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until math has finished producing data for the current Destination Register section.
 * Blocks on the math–pack semaphore so the packer does not read dest before math has written it.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 * Posts to the math–pack semaphore and clears/zeros the dest bank(s) used by the packer;
 *
 * @tparam EN_32BIT_DEST True if math destination is in 32-bit mode, false for 16-bit mode.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_dest_section_done() {
    if constexpr (UnpackToDestEn) {
        _llk_sync_get_<p_stall::PACK0>(semaphore::MATH_PACK);
        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
#ifdef TT_UNPACK_TO_DEST_SECTION_SYNC
            _llk_sync_advance_dest_section_<ckernel::TRISC_ID, EN_32BIT_DEST, p_stall::PACK0>();
#else
            _llk_sync_advance_dest_section_<ckernel::TRISC_ID, true /*EN_32BIT_DEST*/, p_stall::PACK0>();
#endif
        }
#ifdef TT_UNPACK_TO_DEST_SECTION_SYNC
        // Return the physical-bank credit only after PACK0 has drained and
        // this thread has advanced its private section base.
        _llk_sync_post_(semaphore::PACK_UNPACK);
#endif
    } else {
        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DST_SYNC_MODE, EN_32BIT_DEST>();
    }
}

/**
 * @brief Reset packer dest-bank parity to bank 0 at program start (pack-side mirror of llk_math_pack_sync_init).
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_pack_dest_init() {
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();

    // Unpack-to-dest PACR addresses destination through the pack thread's SEC2 base.
    // Initialize it once here; later llk_pack_init() calls may retarget the output DFB
    // mid-kernel and must preserve the current SyncHalf bank parity.
#ifdef TT_UNPACK_TO_DEST_SECTION_SYNC
    if constexpr (UnpackToDestEn && DST_SYNC_MODE == DstSync::SyncHalf) {
        _set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());
    }
#endif
}

/**
 * @brief Configure packer ReLU at runtime from a packed uint32.
 * @param config Packed uint32: bits [1:0] = ReluType, bits [31:16] = threshold.
 */
TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false /* EN_32B_DEST */>(ckernel::ReluConfig::from_packed(config));
}

TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false /* EN_32B_DEST */>(relu_config);
}

/**
 * @brief: Configure packer0 to enable or disable l1 accumulation
 * @param l1_acc_en: if false -> l1 acc is disabled, true -> l1 acc enabled
 **/
inline void llk_pack_reconfig_l1_acc(const std::uint32_t l1_acc_en) { _llk_pack_set_l1_acc_<p_pacr::PACK0>(l1_acc_en); }
