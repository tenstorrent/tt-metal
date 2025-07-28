// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_relu.h"

namespace ckernel::sfpu
{

// General template structure to implement activations
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
struct ActivationImpl;

// Specialization for CELU activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Celu>
{
    static inline void apply(sfpi::vFloat& v, uint32_t param0, uint32_t param1)
    {
        // All params are in FP16_B format
        // param0 = alpha
        // param1 = alpha_recip

        sfpi::vFloat alpha       = Converter::as_float(param0);
        sfpi::vFloat alpha_recip = Converter::as_float(param1);

        v_if (v < 0.0f)
        {
            // Compute exp(x / alpha)
            sfpi::vFloat exp_val = _calculate_exponential_body_<APPROXIMATION_MODE>(v * alpha_recip);

            // Compute CELU: alpha * (exp(x / alpha) - 1)
            v = alpha * (exp_val - 1.0f);
        }
        v_endif;
    }
};

// Specialization for HARDSIGMOID activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Hardsigmoid>
{
    static inline void apply(sfpi::vFloat& v)
    {
        sfpi::vFloat tmp = (v * sfpi::vConstFloatPrgm0) + sfpi::vConstFloatPrgm1;
        v                = _relu_max_body_(tmp, 1.0f);
    }
};

// Dispatch wrapper function
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
inline void apply_activation(sfpi::vFloat& v, uint32_t param0, uint32_t param1)
{
    ActivationImpl<APPROXIMATION_MODE, ACTIVATION_TYPE>::apply(v, param0, param1);
}

// Dispatch wrapper function
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
inline void apply_activation(sfpi::vFloat& v)
{
    ActivationImpl<APPROXIMATION_MODE, ACTIVATION_TYPE>::apply(v);
}

template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS = 8>
inline void _calculate_activation_(uint32_t param0, uint32_t param1)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_activation<APPROXIMATION_MODE, ACTIVATION_TYPE>(v, param0, param1);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS>
inline void _calculate_activation_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_activation<APPROXIMATION_MODE, ACTIVATION_TYPE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void _init_hardsigmoid_()
{
    sfpi::vConstFloatPrgm0 = 0.1668f;
    sfpi::vConstFloatPrgm1 = 0.5f;
}

} // namespace ckernel::sfpu
