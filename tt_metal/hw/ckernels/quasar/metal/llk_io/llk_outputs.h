// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "tensor_shape.h"

inline std::uint32_t get_output_id(std::uint32_t output) { return (output); }

inline const unsigned char get_output_src_format(const std::uint32_t output_id) { return pack_src_format[output_id]; }

inline const unsigned char get_output_dst_format(const std::uint32_t output_id) { return pack_dst_format[output_id]; }

inline const std::uint32_t get_output_num_faces(const std::uint32_t output_id) {
    return static_cast<std::uint32_t>(pack_tile_num_faces[output_id]);
}

inline const std::uint32_t get_output_face_r_dim(const std::uint32_t output_id) {
    return static_cast<std::uint32_t>(pack_tile_face_r_dim[output_id]);
}

inline const std::uint32_t get_output_narrow_tile(const std::uint32_t output_id) {
    return static_cast<std::uint32_t>(pack_narrow_tile[output_id]);
}

inline ckernel::TensorShape get_output_tensor_shape(const std::uint32_t output_id) {
    return ckernel::TensorShape{
        pack_tile_face_r_dim[output_id],
        ckernel::MAX_FACE_C_DIM,
        pack_num_faces_r_dim[output_id],
        pack_num_faces_c_dim[output_id]};
}
