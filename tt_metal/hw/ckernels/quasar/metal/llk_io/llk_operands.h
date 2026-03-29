// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "tensor_shape.h"

inline std::uint32_t get_operand_id(std::uint32_t operand) { return (operand); }

inline const std::uint32_t get_operand_src_format(const std::uint32_t operand_id) {
    return unpack_src_format[operand_id];
}

inline const std::uint32_t get_operand_dst_format(const std::uint32_t operand_id) {
    return unpack_dst_format[operand_id];
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
