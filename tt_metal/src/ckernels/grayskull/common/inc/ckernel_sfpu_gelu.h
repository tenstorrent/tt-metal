#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_init.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu_appx()
{
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;
    vUInt l0 = l_reg[LRegs::LReg0];

    // SFPU microcode
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vUInt l1;
        vUInt l2;
        vFloat result;

        if constexpr (APPROXIMATION_MODE)
        {
            l1 = imm1;
            l2 = imm2;
            result = val;
        } else {
            // f = (0.044715*x^3 + x)
            result = (val * val * val) * 0.044715f + val;

            // result = result * sqrt(2/pi)
            result *= 0.7969f;

            // Reload l1, l2 for lut
            l1 = imm1;
            l2 = imm2;
        }

        result = lut(result, l0, l1, l2);

        val = dst_reg[0];

        result = val * result + val;
        result *= 0.5f;

        dst_reg[0] = result;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
}

#define POLYVAL5(coef4,coef3,coef2,coef1,coef0,val) ( (((coef4*val + coef3)*val + coef2)*val + coef1)*val + coef0 )

inline
vFloat calculate_pos_cdf_appx(vFloat val) {
  //(0,2.5) interpolation polynomial coeffs  [ 0.0122792,  -0.05281024, -0.03048313,  0.41314081,  0.49866379]
  //(2.5,5) interpolation polynomial coeffs  [0.44656975,  0.58216001]

  // FIXME:
  // reuse LREG0-3 for storing coefficients and do product computation
  // const float coef_2dot5_to_5[4] = {-0.00221304f, -0.03253934f, -0.18027954f, -0.44656975f };
  // TTI_SFPLOADI(p_sfpu::LREG0, 0, 0xbb1108a6);
  // TTI_SFPLOADI(p_sfpu::LREG1, 0, 0xbd0547f9);
  // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbe389b33);
  // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbee4a4ca);

  vFloat result;
  v_if( val < 2.5f ) {
    result = POLYVAL5(0.0122792f,  -0.05281024f, -0.03048313f,  0.41314081f,  0.49866379f, val);
  } v_else {
    // assume v >= 2.5f - 5
    //result = POLYVAL5(result,-0.00221304f,  0.03253934f, -0.18027954f,  0.44656975f,  0.58216001f, val);
    //result = ((vFloat)l_reg[LRegs::LReg0])*val + (vFloat)l_reg[LRegs::LReg1];
    //result = result*val + (vFloat)l_reg[LRegs::LReg2];
    //result = result*val + (vFloat)l_reg[LRegs::LReg3];
    result = 0.44656975f*val + 0.58216001f;
  }
  v_endif;

  v_if(result > 1.0f) {
    result = 1.0f;
  }
  v_endif;
  return result;
}


// compute the approximate value of CDF of normal distribution
inline
vFloat calculate_cdf_appx(vFloat val,bool scaled = false) {
    vFloat result = 0.0f;
    vFloat val2 = 0.0;
    v_if ( val < 0.0f ) {
         val2 = -val;
    } v_else {
         val2 = val;
    }
    v_endif;

    result = calculate_pos_cdf_appx(val2);

    v_if ( val < 0.0f ) {
        result = 1.0f - result;
    }
    v_endif;

    if ( scaled ) {
      result *= val; //scale
    }
    return result;
}


template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_sfpu_gelu()
{

    if constexpr (APPROXIMATION_MODE) {
	calculate_gelu_appx<APPROXIMATION_MODE,ITERATIONS>();
    } else {
      constexpr bool scaled = true;
      // SFPU microcode
      for (int d = 0; d < ITERATIONS; d++)
	{
	  vFloat val = dst_reg[0];
	  vFloat result = calculate_cdf_appx(val,scaled);
	  dst_reg[0] = result;
	  dst_reg++;
	}
    }
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat val = in * vConst1p4424 + p_exp::C23_73;

        // Remove Exponent of 7 and bias the Mantissa to 127.
        // LREG2 already holds 2's complement value so we simply do REG2 + REG3
        vInt val_short = p_exp::ADJ_EXP + reinterpret<vInt>(val);

        // SHL to move integer bits to exponent
        val_short <<= 10 - p_exp::FRAC_BITS;
        out = reinterpret<vFloat>(val_short);

        // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
        // without using Relu in Packer to clamp -ve Infinity to 0.
        if constexpr (ZERO_NEGATIVE)
        {
            v_if (val_short < 0) {
                out = vConst0;
            }
            v_endif;
        }
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = sfpu_exp_opt(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        dst_reg[0] = exp;

        v_if (val < 0) {
            dst_reg[0] = sfpu_reciprocal_opt<true>(exp);
        }
        v_endif;
    }
    return out;
}


template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_gelu_core(vFloat in)
{
    constexpr uint imm0 = 0x18FF;
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;

    // SFPU microcode:
    // result = (APPROX_MODE == 1)
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = in * in * in;
        result = result * 0.044715f + in;

        result *= 0.79788f;
    }

    result = lut(result, imm0, imm1, imm2);

    result = result * 0.5f + 0.5f;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_sfpu_gelu_derivative()
{
    // SFPU microcode:
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = val * val * vConstNeg0p5;

        // exp = e^(val) * 1/sqrt(2*pi)
        if constexpr(APPROXIMATION_MODE) {
            vFloat exp = calculate_exponential_body<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        } else {
            dst_reg[0] = result;
            calculate_exponential_body<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            vFloat exp = dst_reg[0];
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        }
        result = calculate_gelu_core<APPROXIMATION_MODE>(val);

        dst_reg[0] = dst_reg[0] + result;

        dst_reg++;
    }
}



} // namespace sfpu
} // namespace ckernel
