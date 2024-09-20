// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>

// Metal specific overrides -- No support for partial tiles so hard-code to fixed 32x32 sizes
inline uint32_t get_output_id(uint32_t output)
{
   return (output);
}

inline const uint32_t get_output_base_id()
{
   const uint32_t OUTPUT_BASE_ID = 16;
   return (OUTPUT_BASE_ID);
}

inline const unsigned char get_output_src_format(const std::uint32_t output_id)
{
   return pack_src_format[output_id];
}

inline const unsigned char get_output_dst_format(const std::uint32_t output_id)
{
   return pack_dst_format[output_id];
}

inline const uint32_t get_output_num_faces(const std::uint32_t output_id)
{
   return (uint32_t)pack_tile_num_faces[output_id];
}

inline const uint32_t get_output_partial_face(const std::uint32_t output_id)
{
   return (uint32_t)pack_partial_face[output_id];
}

inline const uint32_t get_output_face_r_dim(const std::uint32_t output_id)
{
   return (uint32_t)pack_tile_face_r_dim[output_id];
}

inline const uint32_t get_output_narrow_tile(const std::uint32_t output_id)
{
   return (uint32_t)pack_narrow_tile[output_id];
}

inline const uint32_t get_output_tile_r_dim(const std::uint32_t output_id)
{
   return (uint32_t)pack_tile_r_dim[output_id];
}

inline const uint32_t get_output_tile_c_dim(const std::uint32_t output_id)
{
   return (uint32_t)pack_tile_c_dim[output_id];
}
