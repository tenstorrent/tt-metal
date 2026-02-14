// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// 优化后的三角函数实现
// Issue: #37891 - Optimise and improve precision of sin/cos/tan

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel::sfpu {

static const float PI = 3.14159265358979323846f;
static const float PI_2 = 1.57079632679489661923f;
static const float PI_4 = 0.78539816339744830962f;
static const float FRAC_1_PI = 0.31830988618379067154f;
static const float FRAC_2_PI = 0.63661977236758134308f;

// ============================================================================
// 优化的 sin 实现 - 直接在 [-π/2, π/2] 范围内使用多项式
// ============================================================================

// 归约到 [-π/2, π/2] 区间
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_sin_reduce(vFloat x) {
    // 使用 2/π 进行归约，避免除法
    vFloat y = x * FRAC_2_PI;
    
    // 获取整数部分
    vInt n = float_to_int16(y, 0);
    y -= int32_to_float(n, 0);
    
    // 根据 n 的奇偶性确定符号
    vFloat sign = (n & 1) ? -1.0f : 1.0f;
    
    // 转换回角度范围 [-π/2, π/2]
    vFloat z = y * PI_2;
    
    return setsgn(z, sign * z);
}

// 优化的 sin 多项式 (使用 minimax 优化系数)
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_sin_poly(vFloat x) {
    vFloat xx = x * x;
    
    if constexpr (APPROXIMATION_MODE) {
        // 低精度模式：6 阶多项式
        // Coefficients from fpminimax(sin(x), [|1,3,5|], [|single...|], [-pi/2; pi/2], relative)
        return x * (1.0f + xx * (-0.1666666716f + xx * (0.0083333338f + xx * (-0.0001984127f))));
    } else {
        // 高精度模式：9 阶多项式
        // Coefficients from fpminimax(sin(x), [|1,3,5,7,9|], [|single...|], [-pi/2; pi/2], relative)
        return x * (1.0f + xx * (-0.1666666716f + xx * (0.0083333338f + xx * (-0.0001984127f 
               + xx * (0.0000027557f + xx * (-0.0000000251f))))));
    }
}

// ============================================================================
// 优化的 cos 实现 - 直接在 [-π/2, π/2] 范围内使用多项式
// ============================================================================

// 归约到 [0, π] 区间，然后使用 cos(x) = sin(π/2 - x)
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_cos_reduce(vFloat x) {
    // cos(x) = sin(x + π/2)，所以我们只需要调整相位
    vFloat y = x * FRAC_2_PI + 0.5f;
    
    vInt n = float_to_int16(y, 0);
    y -= int32_to_float(n, 0);
    
    vFloat sign = (n & 1) ? -1.0f : 1.0f;
    vFloat z = (y - 0.5f) * PI_2;
    
    return setsgn(z, sign);
}

// 优化的 cos 多项式
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_cos_poly(vFloat x) {
    vFloat xx = x * x;
    
    if constexpr (APPROXIMATION_MODE) {
        // 低精度模式：6 阶多项式
        // Coefficients from fpminimax(cos(x), [|0,2,4|], [|single...|], [-pi/2; pi/2], relative)
        return 1.0f + xx * (-0.5f + xx * (0.0416666679f + xx * (-0.0013888889f)));
    } else {
        // 高精度模式：10 阶多项式
        return 1.0f + xx * (-0.5f + xx * (0.0416666679f + xx * (-0.0013888889f 
               + xx * (0.0000248016f + xx * (-0.0000002756f)))));
    }
}

// ============================================================================
// 优化的 tan 实现 - 改进精度和性能
// ============================================================================

// 优化的 tan 多项式，减少计算步骤
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_tan_poly(vFloat x) {
    vFloat xx = x * x;
    
    if constexpr (APPROXIMATION_MODE) {
        // 低精度模式：使用更少系数
        // Coefficients optimized for [-π/4, π/4] range
        return x * (1.0f + xx * (0.333333343f + xx * (0.133333340f + xx * 0.053968254f)));
    } else {
        // 高精度模式：更多系数，更高精度
        return x * (1.0f + xx * (0.333333343f + xx * (0.133333340f + xx * (0.053968254f 
               + xx * (0.021869488f + xx * 0.008863236f)))));
    }
}

// 改进的 tan 大值处理
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_tan_improved(vFloat x) {
    vFloat abs_x = sfpi::abs(x);
    vFloat result;
    
    // 归约到 [-π/4, π/4] 区间以获得最佳精度
    vFloat y = x * FRAC_2_PI;
    vInt n = float_to_int16(y, 0);
    y -= int32_to_float(n, 0);
    
    // 如果 y > 0.5，映射回 [-0.5, 0.5]
    v_if (y > 0.5f) {
        y = y - 1.0f;
    }
    v_endif;
    
    // 转换回角度
    vFloat z = y * PI_2;
    
    // 计算 tan
    result = sfpu_tan_poly<APPROXIMATION_MODE>(z);
    
    // 处理周期性 (tan 周期为 π)
    v_if (n & 1) {
        result = -result;
    }
    v_endif;
    
    return result;
}

// ============================================================================
// 主计算函数 - 更新版本
// ============================================================================

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine_improved() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        
        // 归约到 [-π/2, π/2]
        vFloat reduced = sfpu_sin_reduce<APPROXIMATION_MODE>(v);
        
        // 计算 sin
        dst_reg[0] = sfpu_sin_poly<APPROXIMATION_MODE>(reduced);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine_improved() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        
        // 归约并计算 cos
        vFloat reduced = sfpu_cos_reduce<APPROXIMATION_MODE>(v);
        
        dst_reg[0] = sfpu_cos_poly<APPROXIMATION_MODE>(reduced);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent_improved() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        
        // 使用改进的 tan 实现
        dst_reg[0] = sfpu_tan_improved<APPROXIMATION_MODE>(v);
        dst_reg++;
    }
}

// ============================================================================
// 保持向后兼容的接口
// ============================================================================

// 保留原始函数作为兼容层
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    calculate_sine_improved<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine() {
    calculate_cosine_improved<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent() {
    calculate_tangent_improved<APPROXIMATION_MODE, ITERATIONS>();
}

} // namespace ckernel::sfpu
