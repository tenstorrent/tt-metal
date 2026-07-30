// Stub for IDE indexing only: a snapshot of a real JIT-generated
// chlkc_descriptors.h. The real one is generated per kernel at build time and
// is not on any static include path, so indexers cannot find it.

#pragma once

#if defined(UCK_CHLKC_MATH)
#include "llk_defs.h"
constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(255);
constexpr bool APPROX = true;
#endif

#if defined(UCK_CHLKC_PACK)
#include "llk_defs.h"
constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(255);
constexpr bool APPROX = true;
#endif

#if !defined(UCK_CHLKC_PACK)
constexpr uint8_t unpack_src_format[32] = {
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};
constexpr uint8_t unpack_dst_format[32] = {
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};
constexpr uint8_t unpack_tile_num_faces[32] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
};
constexpr uint8_t unpack_partial_face[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
constexpr uint8_t unpack_tile_face_r_dim[32] = {
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16
};
constexpr uint8_t unpack_narrow_tile[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
constexpr uint8_t unpack_tile_r_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32
};
constexpr uint8_t unpack_tile_c_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32
};
constexpr uint16_t unpack_tile_size[32] = {
    2048,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088
};
constexpr uint8_t unpack_num_faces_r_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};
constexpr uint8_t unpack_num_faces_c_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};
#endif

#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_UNPACK)
constexpr unsigned char pack_src_format[32] = {
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};
constexpr unsigned char pack_dst_format[32] = {
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};
constexpr uint8_t pack_tile_num_faces[32] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
};
constexpr uint8_t pack_partial_face[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
constexpr uint8_t pack_tile_face_r_dim[32] = {
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16
};
constexpr uint8_t pack_narrow_tile[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
constexpr uint8_t pack_tile_r_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32
};
constexpr uint8_t pack_tile_c_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32
};
constexpr uint16_t pack_tile_size[32] = {
    2048,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088
};
constexpr uint8_t pack_num_faces_r_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};
constexpr uint8_t pack_num_faces_c_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};
#if defined(UCK_CHLKC_PACK)
constexpr uint8_t unpack_src_format[32] = {
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};
#endif
#endif

#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_ISOLATE_SFPU)
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf
#endif
