#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_format_conversions.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"

using namespace ckernel;
template <SfpuType sfpu_type>
void static_assert_sfpu_type_dependent() {
    static_assert(sfpu_type == SfpuType::unused, "sfpu_type exception");
}
// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);

}
inline void eltwise_unary_sfpu_configure_mop();

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu(
    const uint operand,
    uint dst_index, 
    int vector_mode = (int)Dim::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu<{}, {}, {}>({}, {}, {}, {}, {}, {}, {}, {}, {})", (int)sfpu_op, APPROXIMATE, Dst, operand, dst_index, vector_mode, param0, param1, param2, param3, param4, param5);
    
    constexpr int ITERATIONS = 8;
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    if (vector_mode == (int)Dim::R) {
        const std::uint32_t operand_id = get_operand_id(operand);
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
        const int iterations = (math::get_num_faces(operand_id) < 4) ? 
                                    ((math::get_face_r_dim(operand_id) <= 2) ? 2 : math::get_face_r_dim(operand_id)/2) : 2; // At least 2 iterations for odd and even columns
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(iterations, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    } else if (vector_mode == (int)Dim::C) {
        // Do a column vector, Face0 + Face2 if tile is 32x32 or Face0+Face1 if tiles is 32x16 -- All iterations for full face
        const std::uint32_t operand_id = get_operand_id(operand);
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(ITERATIONS, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            if (math::get_num_faces(operand_id)>2) { // Skip next 2 faces if tile is 32x32
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }
        if (math::get_num_faces(operand_id)<=2) { 
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }    
    } else {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(ITERATIONS, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    math::clear_dst_reg_addr(); 

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(
    const uint operand, uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_init<{}, {}>({}, {}, {}, {}, {}, {})", sfpu_op, APPROXIMATE, param0, param1, param2, param3, param4, param5);
    eltwise_unary_sfpu_configure_addrmod< sfpu_op >();
    if constexpr (sfpu_op == SfpuType::dropout) {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op, param2);
    } else {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// New LLK SFPU APIs

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_exponential(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_exponential<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::exponential, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sqrt(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_sqrt<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::sqrt, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_gelu<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::gelu, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_gelu_derivative<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::gelu_derivative, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_derivative, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_reciprocal(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_reciprocal<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::reciprocal, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_log(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_log<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::log, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_tanh(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_tanh<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::tanh, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>(operand);
}

template <DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_dropout(const uint operand, uint dst_index, int vector_mode, int integer_dropout, int scale_factor) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_dropout<{}>({}, {}, {}, {})", dst_sync, dst_index, vector_mode, integer_dropout, scale_factor);
    constexpr bool dont_care = false;
    llk_math_eltwise_unary_sfpu<SfpuType::dropout, dont_care, dst_sync>(operand, dst_index, vector_mode, integer_dropout, scale_factor);
}

inline void llk_math_eltwise_unary_sfpu_dropout_init(const uint operand, uint seed = 0) {
    constexpr bool dont_care = false;
    constexpr uint dont_care_param = 0;

    llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, dont_care>(operand, dont_care_param, dont_care_param, seed);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sigmoid(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_sigmoid<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::sigmoid, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_max(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_max<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::max, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::max, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_square(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_square<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::square, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_square_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::square, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_power(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC, int pow = 0) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_power<{}, {}>({}, {}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode, pow);
    llk_math_eltwise_unary_sfpu<SfpuType::power, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode, pow);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::power, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sine(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_sine<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::sine, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sine_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sine, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_cosine(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_cosine<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::cosine, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cosine_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cosine, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_lrelu(const uint operand, uint dst_index, int vector_mode, uint uint_slope) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_lrelu<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::lrelu, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode, uint_slope);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lrelu_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lrelu, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_max(const uint operand, uint dst_index, int vector_mode, uint uint_threshold) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_relu_max<{}, {}>({}, {}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode, uint_threshold);
    llk_math_eltwise_unary_sfpu<SfpuType::relu_max, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode, uint_threshold);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_max_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::relu_max, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_min(const uint operand, uint dst_index, int vector_mode, uint uint_threshold) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_relu_min<{}, {}>({}, {}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode, uint_threshold);
    llk_math_eltwise_unary_sfpu<SfpuType::relu_min, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode, uint_threshold);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_min_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_abs(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    TT_LLK_DUMP("llk_math_eltwise_unary_sfpu_abs<{}, {}>({}, {})", APPROXIMATE, dst_sync, dst_index, vector_mode);
    llk_math_eltwise_unary_sfpu<SfpuType::abs, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_cast_fp32_to_fp16a(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::cast_fp32_to_fp16a, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cast_fp32_to_fp16a_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cast_fp32_to_fp16a, APPROXIMATE>(operand);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_requant_int32(const uint operand, uint dst_index, int vector_mode = (int)Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::requant_int32, APPROXIMATE, dst_sync>(operand, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_requant_int32_init(const uint operand) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::requant_int32, APPROXIMATE>(operand);
}
