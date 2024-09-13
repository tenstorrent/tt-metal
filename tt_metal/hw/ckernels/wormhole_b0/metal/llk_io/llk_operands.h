// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>

inline uint32_t get_operand_id(uint32_t operand)
{
   return (operand);
}

inline const uint32_t get_operand_src_format(const std::uint32_t operand_id)
{
   return unpack_src_format[operand_id];
}

inline const uint32_t get_operand_dst_format(const std::uint32_t operand_id)
{
   return unpack_dst_format[operand_id];
}

inline const uint32_t get_operand_num_faces(const std::uint32_t operand_id)
{
   return unpack_tile_num_faces[operand_id];
}

inline const uint32_t get_operand_partial_face(const std::uint32_t operand_id)
{
   return unpack_partial_face[operand_id];
}

inline const uint32_t get_operand_face_r_dim(const std::uint32_t operand_id)
{
   return unpack_tile_face_r_dim[operand_id];
}

inline const uint32_t get_operand_narrow_tile(const std::uint32_t operand_id)
{
   return unpack_narrow_tile[operand_id];
}

inline const uint32_t get_operand_tile_r_dim(const std::uint32_t operand_id)
{
   return unpack_tile_r_dim[operand_id];
}

inline const uint32_t get_operand_tile_c_dim(const std::uint32_t operand_id)
{
   return unpack_tile_c_dim[operand_id];
}
