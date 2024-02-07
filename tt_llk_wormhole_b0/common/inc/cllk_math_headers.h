
#include "ckernel_defs.h"
#include "tensix_types.h"


//
// LLK math common
//

template <DstSync Dst>
inline void llk_math_wait_for_dest_available();

inline void llk_math_dest_section_done();

template <DstClear Dst>
inline void llk_math_clear_dst<Dst>(uint tile_index);

template <DstStart Dst>
inline void llk_math_set_dest_section_base();

template <DstStart Dst>
inline void llk_math_set_dest_section_flip();

//
// LLK matrix multiplication
// 
inline void llk_math_mmul_init();
inline void llk_math_mmul();

//
// LLK Eltwise binary
//
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_broadcast_type>
inline void llk_math_eltwise_binary();
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_broadcast_type>
inline void llk_math_eltwise_binary_init();

//
// LLK Eltwise unary sfpu
//
template <SfpuType sfpu_type, bool approximation_mode>
inline void llk_math_eltwise_unary_sfpu();
template <SfpuType sfpu_type, bool approximation_mode>
inline void llk_math_eltwise_unary_sfpu_init();

//
// LLK Eltwise unary datacopy
//
template <DataCopyType datacopy_type>
inline void llk_math_eltwise_unary_datacopy();
template <DataCopyType datacopy_type>
inline void llk_math_eltwise_unary_datacopy_init();

