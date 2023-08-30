#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

//template <bool APPROXIMATION_MODE>
//inline void sfpu_init(SfpuType operation, uint param0 = 0)


template <bool APPROXIMATION_MODE>
inline void calculate_elu(uint slope)
{
    // SFPU microcode
    Converter c_slope;
    c_slope.u = slope;
    vFloat s = c_slope.f;

    #pragma GCC unroll 0
    for (int d = 0; d < WHB0_ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
	  vFloat v_exp = calculate_exponential_body<true>(v);
	  v = s*(v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}


template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_elu(uint dst_index, uint param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_elu<APPROXIMATE>,
				 dst_index, Dim::RC, param0);
}


}  // namespace sfpu
}  // namespace ckernel
