// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_unpack_common.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * All the following functions are added to enable Unpack <-> Math synchronization
 * on the destination register using semaphores, for the unpack-to-dest path.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until destination register space is available.
 * Blocks on the DEST_FREE semaphore (UNPACK side of the three-semaphore protocol)
 * until PACK has released a DEST bank.
 */
inline void llk_unpack_wait_for_dest_available() {
    WAYPOINT("UWDW");
    _llk_unpack_wait_for_dest_available_<DST_SYNC_MODE>();
    WAYPOINT("UWDD");
}

/**
 * @brief Signals that the current destination section has been filled by UNPACK.
 * Posts to the UNPACK_MATH semaphore so the math trisc can proceed.
 */
inline void llk_unpack_dest_section_done() { _llk_unpack_dest_section_done_(); }

/**
 * @brief Initializes unpack–math synchronization for the destination register.
 * Bootstraps DEST_FREE and initializes UNPACK_MATH; called once at compute-kernel
 * init from the UNPACK trisc, alongside llk_unpack_to_dest_hw_configure.
 */
inline void llk_unpack_pack_sync_init() { _llk_unpack_pack_sync_init_<DST_SYNC_MODE>(); }

/**
 * @brief Configures the unpack-to-dest path hardware (ALU dest mode, dvalid CTRL
 *        registers, and SEC<N> base offsets for the three-semaphore protocol).
 *
 * Per-CB dispatch: derives EN_FP32 / EN_INT32 from the operand's unpack source
 * format. Float32 -> EN_FP32, Int32 -> EN_INT32, otherwise both off (16-bit dest).
 *
 * @param icb: The input operand circular buffer used to drive 32-bit dest mode.
 */
inline void llk_unpack_to_dest_hw_configure(const std::uint32_t icb) {
    const std::uint32_t operand_id = get_operand_id(icb);
    const DataFormat fmt = static_cast<DataFormat>(unpack_src_format[operand_id]);
    if (fmt == DataFormat::Float32) {
        _llk_unpack_to_dest_hw_configure_<true /*EN_FP32*/, false /*EN_INT32*/, DST_SYNC_MODE>();
    } else if (fmt == DataFormat::Int32) {
        _llk_unpack_to_dest_hw_configure_<false /*EN_FP32*/, true /*EN_INT32*/, DST_SYNC_MODE>();
    } else {
        _llk_unpack_to_dest_hw_configure_<false /*EN_FP32*/, false /*EN_INT32*/, DST_SYNC_MODE>();
    }
}

/**
 * @brief Programs l1 info & source register format for both UNP_A and UNP_B
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const std::uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const std::uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id
    for (std::uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
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
        bd_val.f.z_dim = unpack_tile_num_faces[i];

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
