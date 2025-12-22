// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

inline std::uint32_t get_output_id(std::uint32_t output) { return (output); }

inline const unsigned char get_output_src_format(const std::uint32_t output_id) { return pack_src_format[output_id]; }

inline const unsigned char get_output_dst_format(const std::uint32_t output_id) { return pack_dst_format[output_id]; }

inline const std::uint32_t get_output_num_faces(const std::uint32_t output_id) {
    return static_cast<std::uint32_t>(pack_tile_num_faces[output_id]);
}

inline const std::uint32_t get_output_face_r_dim(const std::uint32_t output_id) {
    return static_cast<std::uint32_t>(pack_tile_face_r_dim[output_id]);
}
