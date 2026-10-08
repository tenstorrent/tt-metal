#ifndef CKERNEL_SFPUS_GELU_H
#define CKERNEL_SFPUS_GELU_H

#include "llk_math_generic.h"
#include "ckernel_sfpu_utils.h"
#include <cmath>

namespace ckernel
{

inline void calculate_gelu_derivative_simple(const uint32_t input_tile_addr, const uint32_t result_tile_addr)
{
    volatile uint32_t* input = (volatile uint32_t*)input_tile_addr;
    volatile uint32_t* result = (volatile uint32_t*)result_tile_addr;

    constexpr float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    // In FP32, GELU'(x) = Phi(x) + x*phi(x) stays above 1.0 until x >= 5.9607,
    // where it rounds to exactly 1.0 in FP32. The previous threshold of 3.1719
    // was derived from BF16 behavior and caused accuracy loss in FP32.
    constexpr float fp32_derivative_saturation_threshold = 5.9607f;

    for (uint32_t j = 0u; j < TILE_H / 32u; ++j)
    {
        for (uint32_t i = 0u; i < TILE_W / 32u; ++i)
        {
            constexpr uint32_t reg_idx_in = 0;
            constexpr uint32_t reg_idx_out = 1;

            llk_sfpu_pop_input(reg_idx_in, input);

            float x = get_scalar_from_vector_reg<0>(reg_idx_in);
            float result_val;

            if (x >= fp32_derivative_saturation_threshold)
            {
                result_val = 1.0f;
            }
            else if (x <= -4.0f)
            {
                result_val = 0.0f;
            }
            else
            {
                // GELU'(x) = Phi(x) + x * phi(x)
                //         = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
                float x2 = x * x;
                float phi_x = std::exp(-0.5f * x2) * sqrt_2_over_pi;
                // Approximate Phi(x) using a rational approximation
                // For x >= 0: Phi(x) = 1 - phi(x) * (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)
                //             where t = 1/(1 + p*x), p = 0.2316419
                // For x < 0: Phi(x) = 1 - Phi(-x)
                float ax = std::fabs(x);
                float t = 1.0f / (1.0f + 0.2316419f * ax);
                float poly = t * (0.319381530f + t * (-0.356563782f + t * (1.781477936f + t * (-1.821255978f + t * 1.330274429f))));
                float Phi_pos = 1.0f - phi_x * poly;
                float Phi_x = (x >= 0.0f) ? Phi_pos : (1.0f - Phi_pos);
                result_val = Phi_x + x * phi_x;
            }

            set_vector_reg_from_scalar<0>(reg_idx_out, result_val);
            llk_sfpu_push_output(reg_idx_out, result);
        }
        input += TILE_W / 32u;
        result += TILE_W / 32u;
    }
}

inline void calculate_gelu_derivative_approx(const uint32_t input_tile_addr, const uint32_t result_tile_addr)
{
    volatile uint32_t* input = (volatile uint32_t*)input_tile_addr;
    volatile uint32_t* result = (volatile uint32_t*)result_tile_addr;

    constexpr float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float fp32_derivative_saturation_threshold = 5.9607f;

    for (uint32_t j = 0u; j < TILE_H / 32u; ++j)
    {
        for (uint32_t i = 0u; i < TILE_W / 32u; ++i)
        {
            constexpr uint32_t reg_idx_in = 0;
            constexpr uint32_t reg_idx_out = 1;

            llk_sfpu_pop_input(reg_idx_in, input);

            float x = get_scalar_from_vector_reg<0>(reg_idx_in);
            float result_val;

            if (x >= fp32_derivative_saturation_threshold)
            {
                result_val = 1.0f;
            }
            else if (x <= -4.0f)
            {
                result_val = 0.0f;
            }
            else
            {
                float x2 = x * x;
                float phi_x = std::exp(-0.5f * x2) * sqrt_2_over_pi;
                float t = 1.0f / (1.0f + 0.2316419f * std::fabs(x));
                float poly = t * (0.319381530f + t * (-0.356563782f + t * (1.781477936f + t * (-1.821255978f + t * 1.330274429f))));
                float Phi_pos = 1.0f - phi_x * poly;
                float Phi_x = (x >= 0.0f) ? Phi_pos : (1.0f - Phi_pos);
                result_val = Phi_x + x * phi_x;
            }

            set_vector_reg_from_scalar<0>(reg_idx_out, result_val);
            llk_sfpu_push_output(reg_idx_out, result);
        }
        input += TILE_W / 32u;
        result += TILE_W / 32u;
    }
}

inline void calculate_gelu(const uint32_t input_tile_addr, const uint32_t result_tile_addr)
{
    volatile uint32_t* input = (volatile uint32_t*)input_tile_addr;
    volatile uint32_t* result = (volatile uint32_t*)result_tile_addr;

    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float sqrt_half = 0.7071067811865476f;

    for (uint32_t j = 0u; j < TILE_H / 32u; ++j)
    {
        for (uint32_t i = 0u; i < TILE_W / 32u; ++i)
        {
            constexpr uint32_t reg_idx_in = 0;
            constexpr uint32_t reg_idx_out = 1;

            llk_sfpu_pop_input(reg_idx_in, input);

            float x = get_scalar_from_vector_reg<0>(reg_idx_in);
            float result_val;

            if (x >= 6.0f)
            {
                result_val = x;
            }
            else if (x <= -6.0f)
            {
                result_val = 0.0f;
            }
            else
            {
                float x_sqrt_half = x * sqrt_half;
                float phi_x = std::exp(-0.5f * x * x) * sqrt_2_over_pi;
                float t = 1.0f / (1.0f + 0.2316419f * std::fabs(x));
                float poly = t * (0.319381530f + t * (-0.356563782f + t * (1.781477936f + t * (-1.821255978f + t * 1.330274429f))));
                float Phi_pos = 1.0f - phi_x * poly;
                float Phi_x = (x >= 0.0f) ? Phi_pos : (1.0f - Phi_pos);
                result_val = x * Phi_x;
            }

            set_vector_reg_from_scalar<0>(reg_idx_out, result_val);
            llk_sfpu_push_output(reg_idx_out, result);
        }
        input += TILE_W / 32u;
        result += TILE_W / 32u;
    }
}

} // namespace ckernel

#endif // CKERNEL_SFPUS_GELU_H
