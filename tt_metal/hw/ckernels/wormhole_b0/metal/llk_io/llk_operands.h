
#pragma once
#include <cstdint>
#include <vector>

inline uint32_t get_operand_id(uint32_t operand)
{
    const int INTERMEDIATE_BASE_ID = 24;
    const int OPERAND_BASE_ID = 0;
    return (operand>=INTERMEDIATE_BASE_ID) ? operand - 8 : operand - OPERAND_BASE_ID;
}

inline const uint32_t get_operand_src_format(const std::uint32_t operand_id)
{
   return unpack_src_format[operand_id];
}

inline const uint32_t get_operand_dst_format(const std::uint32_t operand_id)
{
   return unpack_src_format[operand_id];
}

inline const uint32_t get_operand_num_faces(const std::uint32_t operand_id)
{
   return 4;
}

inline const uint32_t get_operand_partial_face(const std::uint32_t operand_id)
{
   return 0;
}

inline const uint32_t get_operand_face_r_dim(const std::uint32_t operand_id)
{
   return 16;
}

inline const uint32_t get_operand_narrow_tile(const std::uint32_t operand_id)
{
   return 0;
}

inline const std::vector<uint32_t> get_operand_tile_dims(const std::uint32_t operand_id)
{
   return {32, 32};
}
