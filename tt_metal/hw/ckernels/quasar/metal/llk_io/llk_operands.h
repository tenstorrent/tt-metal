// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "tensix_types.h"
#include "tensor_shape.h"

inline std::uint32_t get_operand_id(std::uint32_t operand) { return (operand); }

inline const std::uint32_t get_operand_src_format(const std::uint32_t operand_id) {
    return unpack_src_format[operand_id];
}

inline const std::uint32_t get_operand_dst_format(const std::uint32_t operand_id) {
    return unpack_dst_format[operand_id];
}

// Per-operand unpack-to-dest route under the kernel-wide UnpackToDestEn flag. UNP_DEST writes the unpacker output
// format into DEST as is: the Quasar unpacker conversion table (is_quasar_unpack_reconfig_pair_supported) has no
// 16-bit to 32-bit widening, Float32 / Int32 output is Dest-only and Float16 / Float16_b output is Src-only. So an
// operand may take the to-dest path only when its datum width matches the DEST width; any other operand takes
// SrcA / SrcB, where the FPU widens it. Mirrors WH/BH should_unpack_to_dest / is_32bit_input.
inline bool operand_fits_unpack_to_dest(const std::uint32_t operand_id) {
    const auto fmt = static_cast<DataFormat>(unpack_dst_format[operand_id]);
    const bool is_32bit = (fmt == DataFormat::Float32) || (fmt == DataFormat::Int32);
    return is_32bit == DST_ACCUM_MODE;
}

inline const std::uint32_t get_operand_num_faces(const std::uint32_t operand_id) {
    return static_cast<std::uint32_t>(unpack_tile_num_faces[operand_id]);
}

inline const std::uint32_t get_operand_face_r_dim(const std::uint32_t operand_id) {
    return static_cast<std::uint32_t>(unpack_tile_face_r_dim[operand_id]);
}

inline const std::uint32_t get_operand_narrow_tile(const std::uint32_t operand_id) {
    return static_cast<std::uint32_t>(unpack_narrow_tile[operand_id]);
}

inline ckernel::TensorShape get_operand_tensor_shape(const std::uint32_t operand_id) {
    return ckernel::TensorShape{
        unpack_tile_face_r_dim[operand_id],
        ckernel::MAX_FACE_C_DIM,
        unpack_num_faces_r_dim[operand_id],
        unpack_num_faces_c_dim[operand_id]};
}

// Compile-time version for future support when operands are passed as compile-time constants.
template <std::uint32_t operand_id>
constexpr ckernel::TensorShape get_operand_tensor_shape() {
    return ckernel::TensorShape{
        unpack_tile_face_r_dim[operand_id],
        ckernel::MAX_FACE_C_DIM,
        unpack_num_faces_r_dim[operand_id],
        unpack_num_faces_c_dim[operand_id]};
}
