// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR)

#include "api/compute/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#include "llk_unpack_unary_broadcast_operands.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_unary_broadcast.h"
#endif

namespace ckernel {

template <BroadcastType bcast_type>
ALWI void unary_bcast_init(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

#ifdef TRISC_UNPACK
    const std::uint32_t operand_id_u = get_operand_id(icb);
    const std::uint32_t dst_format_u = get_operand_dst_format(operand_id_u);
    const bool enable_unpack_to_dest_u = (dst_format_u == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                         (dst_format_u == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                         (dst_format_u == static_cast<std::uint32_t>(DataFormat::Int32));

    UNPACK((llk_unpack_hw_configure(icb)));
    if constexpr (bcast_type == BroadcastType::NONE) {
        if (enable_unpack_to_dest_u) {
            UNPACK((_llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, DST_ACCUM_MODE>(operand_id_u, 1)));
        } else {
            UNPACK((_llk_unpack_unary_operand_init_<p_unpacr::UNP_B, false, DST_ACCUM_MODE>(operand_id_u, 1)));
        }
    } else {
        if (enable_unpack_to_dest_u) {
            UNPACK((_llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_A, bcast_type, true, false>(
                operand_id_u, 1)));
        } else {
            UNPACK((_llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, bcast_type, false, DST_ACCUM_MODE>(
                operand_id_u, 1)));
        }
    }
#endif

#ifdef TRISC_MATH
    const std::uint32_t operand_id_m = get_operand_id(icb);
    const std::uint32_t dst_format_m = get_operand_dst_format(operand_id_m);
    const bool enable_unpack_to_dest_m = (dst_format_m == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                         (dst_format_m == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                         (dst_format_m == static_cast<std::uint32_t>(DataFormat::Int32));

    if constexpr (bcast_type == BroadcastType::NONE) {
        if (enable_unpack_to_dest_m) {
            MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
        } else {
            MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::B2D, DST_ACCUM_MODE>(icb)));
        }
    } else if (!enable_unpack_to_dest_m) {
        const TileShape tile_shape{
            .num_faces = get_operand_num_faces(operand_id_m),
            .face_r_dim = get_operand_face_r_dim(operand_id_m),
            .face_c_dim = static_cast<std::uint32_t>(ckernel::trisc::FACE_C_DIM),
            .narrow_tile = get_operand_narrow_tile(operand_id_m) != 0,
        };
        MATH((_llk_math_eltwise_unary_broadcast_init_<bcast_type, false, DST_ACCUM_MODE>(tile_shape)));
    }
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif

#ifdef TRISC_PACK
    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
#endif
}

template <BroadcastType bcast_type>
ALWI void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifdef TRISC_UNPACK
    const std::uint32_t operand_id_u = get_operand_id(icb);
    const std::uint32_t dst_format_u = get_operand_dst_format(operand_id_u);
    const bool enable_unpack_to_dest_u = (dst_format_u == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                         (dst_format_u == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                         (dst_format_u == static_cast<std::uint32_t>(DataFormat::Int32));
    const std::uint32_t l1_tile_index = g_dfb_interface[operand_id_u].rd_entry_idx + in_tile_index;

    if constexpr (bcast_type == BroadcastType::NONE) {
        if (enable_unpack_to_dest_u) {
            UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_index)));
        } else {
            UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_B>(l1_tile_index)));
        }
    } else {
        if (enable_unpack_to_dest_u) {
            UNPACK((_llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_A, true>(l1_tile_index)));
        } else {
            UNPACK((_llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_B, false>(l1_tile_index)));
        }
    }
#endif

#ifdef TRISC_MATH
    const std::uint32_t operand_id_m = get_operand_id(icb);
    const std::uint32_t dst_format_m = get_operand_dst_format(operand_id_m);
    const bool enable_unpack_to_dest_m = (dst_format_m == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                         (dst_format_m == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                         (dst_format_m == static_cast<std::uint32_t>(DataFormat::Int32));

    if constexpr (bcast_type == BroadcastType::NONE) {
        MATH((llk_math_eltwise_unary_datacopy(dst_tile_index, icb)));
    } else if (!enable_unpack_to_dest_m) {
        const TileShape tile_shape{
            .num_faces = get_operand_num_faces(operand_id_m),
            .face_r_dim = get_operand_face_r_dim(operand_id_m),
            .face_c_dim = static_cast<std::uint32_t>(ckernel::trisc::FACE_C_DIM),
            .narrow_tile = get_operand_narrow_tile(operand_id_m) != 0,
        };
        MATH((_llk_math_eltwise_unary_broadcast_<bcast_type, false, DST_ACCUM_MODE>(dst_tile_index, tile_shape)));
    }
#endif
}

template <BroadcastType old_bcast_type, BroadcastType new_bcast_type>
void reconfigure_unary_bcast(uint32_t old_icb, uint32_t new_icb, uint32_t old_ocb, uint32_t new_ocb) {
#ifdef TRISC_UNPACK
    const std::uint32_t new_operand_id = get_operand_id(new_icb);
    const std::uint32_t old_operand_id = get_operand_id(old_icb);
    const std::uint32_t dst_format = get_operand_dst_format(new_operand_id);
    const bool enable_unpack_to_dest = (dst_format == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                       (dst_format == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                       (dst_format == static_cast<std::uint32_t>(DataFormat::Int32));
    const bool unpacker_src_format_change = unpack_src_format[new_operand_id] != unpack_src_format[old_operand_id];
    const bool unpacker_dst_format_change = unpack_dst_format[new_operand_id] != unpack_dst_format[old_operand_id];
    const bool bcast_type_change = (old_bcast_type != new_bcast_type);

    if (unpacker_src_format_change || unpacker_dst_format_change) {
        UNPACK((llk_unpack_hw_configure(new_icb)));
    }

    if (unpacker_src_format_change || unpacker_dst_format_change || bcast_type_change) {
        if constexpr (new_bcast_type == BroadcastType::NONE) {
            if (enable_unpack_to_dest) {
                UNPACK((_llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, DST_ACCUM_MODE>(new_operand_id, 1)));
            } else {
                UNPACK((_llk_unpack_unary_operand_init_<p_unpacr::UNP_B, false, DST_ACCUM_MODE>(new_operand_id, 1)));
            }
        } else {
            if (enable_unpack_to_dest) {
                UNPACK((_llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_A, new_bcast_type, true, false>(
                    new_operand_id, 1)));
            } else {
                UNPACK((_llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, new_bcast_type, false, DST_ACCUM_MODE>(
                    new_operand_id, 1)));
            }
        }
    }
#endif

#ifdef TRISC_MATH
    const std::uint32_t new_operand_id = get_operand_id(new_icb);
    const std::uint32_t old_operand_id = get_operand_id(old_icb);
    const std::uint32_t dst_format = get_operand_dst_format(new_operand_id);
    const bool enable_unpack_to_dest = (dst_format == static_cast<std::uint32_t>(DataFormat::Float32)) ||
                                       (dst_format == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
                                       (dst_format == static_cast<std::uint32_t>(DataFormat::Int32));
    const bool unpacker_dst_format_change = unpack_dst_format[new_operand_id] != unpack_dst_format[old_operand_id];
    const bool bcast_type_change = (old_bcast_type != new_bcast_type);

    if (unpacker_dst_format_change) {
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(new_icb, new_icb)));
    }

    if (unpacker_dst_format_change || bcast_type_change) {
        if constexpr (new_bcast_type == BroadcastType::NONE) {
            if (enable_unpack_to_dest) {
                MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE>(new_icb)));
            } else {
                MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::B2D, DST_ACCUM_MODE>(new_icb)));
            }
        } else if (!enable_unpack_to_dest) {
            const TileShape tile_shape{
                .num_faces = get_operand_num_faces(new_operand_id),
                .face_r_dim = get_operand_face_r_dim(new_operand_id),
                .face_c_dim = static_cast<std::uint32_t>(ckernel::trisc::FACE_C_DIM),
                .narrow_tile = get_operand_narrow_tile(new_operand_id) != 0,
            };
            MATH((_llk_math_eltwise_unary_broadcast_init_<new_bcast_type, false, DST_ACCUM_MODE>(tile_shape)));
        }
    }
#endif

#ifdef TRISC_PACK
    // api/compute/pack.h pack_reconfig_data_format is a no-op on Quasar (see TODO there). WH uses it to
    // point the packer at the second output; we must reprogram pack for new_ocb when operand or formats differ.
    const std::uint32_t old_out_id = get_output_id(old_ocb);
    const std::uint32_t new_out_id = get_output_id(new_ocb);
    const bool pack_operand_change = (old_ocb != new_ocb);
    const bool pack_format_change =
        (pack_src_format[old_out_id] != pack_src_format[new_out_id]) ||
        (pack_dst_format[old_out_id] != pack_dst_format[new_out_id]);
    if (pack_operand_change || pack_format_change) {
        PACK((llk_pack_hw_configure(new_ocb)));
        PACK((llk_pack_init(new_ocb)));
    }
#endif
}

}  // namespace ckernel

#endif  // ARCH_QUASAR
