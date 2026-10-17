// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// Metal specific overrides -- No support for partial tiles so hard-code to fixed 32x32 sizes
inline uint32_t get_output_id(uint32_t output) { return (output); }

inline const uint32_t get_output_base_id() {
    const uint32_t OUTPUT_BASE_ID = 16;
    return (OUTPUT_BASE_ID);
}

inline const unsigned char get_output_src_format(const std::uint32_t output_id) { return pack_src_format[output_id]; }

inline const unsigned char get_output_dst_format(const std::uint32_t output_id) { return pack_dst_format[output_id]; }

inline const uint32_t get_output_num_faces(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_num_faces[output_id];
}

inline const uint32_t get_output_face_r_dim(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_face_r_dim[output_id];
}

// Packer partial-face flag: the output's faces are shorter than FACE_R_DIM rows, so a tiny tile of
// one or two such faces carries at most 16 exponent bytes (one 16-byte exponent chunk). This is NOT
// the circular buffer's partial_face flag (pack_partial_face[]: tile height < TILE_HEIGHT), which is
// also set for a 16x32 tile whose two full-height faces carry 32 exponent bytes; feeding that flag to
// the packer lays every 16x32 block-float tile out one chunk early, so it has no accessor here.
inline bool is_output_face_partial(const std::uint32_t output_id) {
    return get_output_face_r_dim(output_id) < ckernel::FACE_R_DIM;
}

inline const uint32_t get_output_narrow_tile(const std::uint32_t output_id) {
    return (uint32_t)pack_narrow_tile[output_id];
}

inline const uint32_t get_output_tile_r_dim(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_r_dim[output_id];
}

inline const uint32_t get_output_tile_c_dim(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_c_dim[output_id];
}
