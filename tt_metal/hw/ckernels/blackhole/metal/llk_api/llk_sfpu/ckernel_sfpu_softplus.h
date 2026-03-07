// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_piecewise_rational.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// =============================================================================
// LUT-based softplus: log(1 + exp(x)) via piecewise rational P(x)/Q(x)
//
// BF16: rational n9/d9, 1 segment, uniform, range [-10, 10]
//   MaxULP=1.00, MeanULP=0.45  (hardware-validated, run_csv.sh + Tracy)
//
// FP32: rational n11/d11, 2 segments, uniform, range [-10, 10]
//   MaxULP=106.66, MeanULP=2.66  (hardware-validated, run_csv.sh + Tracy)
// =============================================================================

#ifdef INP_FLOAT32
constexpr uint32_t SOFTPLUS_NUM_DEGREE = 11;
constexpr uint32_t SOFTPLUS_DEN_DEGREE = 11;
constexpr uint32_t SOFTPLUS_NUM_SEGMENTS = 2;
constexpr uint32_t SOFTPLUS_LUT_SIZE = 51;
constexpr std::array<float, 51> SOFTPLUS_LUT = {
    {-1.0000000000e+01f, 0.0000000000e+00f,  1.0000000000e+01f,  6.9314718246e-01f,  -1.5993961543e+00f,
     -2.5127007067e-01f, -1.3265351206e-01f, -2.9783745878e-02f, -1.5730193118e-03f, 2.8420342278e-04f,
     5.4250589073e-05f,  4.2192577894e-06f,  1.8079879105e-07f,  4.2213262608e-09f,  4.2415547203e-11f,
     1.0000000000e+00f,  -3.0287885070e+00f, 1.6419652700e+00f,  -8.2960790396e-01f, 2.6686237752e-01f,
     -6.7941252142e-02f, 1.3101734454e-02f,  -1.8655313179e-03f, 1.8875836395e-04f,  -2.0982502519e-05f,
     7.1359330178e-07f,  -9.3163990300e-08f, 6.9314718246e-01f,  7.5465214074e+01f,  9.6088657975e+01f,
     6.4373108774e+01f,  2.9242373943e+01f,  9.6345350314e+00f,  2.3334935121e+00f,  4.0709061723e-01f,
     4.8712256554e-02f,  3.5111696889e-03f,  3.2350137371e-04f,  -1.5047511503e-06f, 1.0000000000e+00f,
     1.0815194273e+02f,  6.0431155443e+01f,  2.9775133565e+01f,  9.8190872483e+00f,  2.2600082848e+00f,
     4.1877857578e-01f,  4.7625164822e-02f,  3.5753458696e-03f,  3.2110606327e-04f,  -1.4526502952e-06f,
     -5.0736897138e-10f}};

#else

constexpr uint32_t SOFTPLUS_NUM_DEGREE = 9;
constexpr uint32_t SOFTPLUS_DEN_DEGREE = 9;
constexpr uint32_t SOFTPLUS_NUM_SEGMENTS = 1;
constexpr uint32_t SOFTPLUS_LUT_SIZE = 22;
constexpr std::array<float, 22> SOFTPLUS_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 6.9314640760e-01f,  3.0334073305e-01f,  1.0266214609e-01f,
     2.5587543845e-02f,  4.1804499924e-03f, 4.4794997666e-04f,  3.1875460991e-05f,  1.4791692138e-06f,
     4.1209883506e-08f,  5.2948567753e-10f, 1.0000000000e+00f,  -2.8372007608e-01f, 1.7242786288e-01f,
     -3.6300994456e-02f, 8.6406944320e-03f, -1.1708125239e-03f, 1.2544600759e-04f,  -8.4130733740e-06f,
     3.4111849345e-07f,  -6.1077645164e-09f}};

#endif

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    vFloat x = beta * dst_reg[0];
    v_if(x < threshold) {
        dst_reg[0] =
            beta_reciprocal *
            piecewise_rational_eval<SOFTPLUS_NUM_DEGREE, SOFTPLUS_DEN_DEGREE, SOFTPLUS_NUM_SEGMENTS, SOFTPLUS_LUT_SIZE>(
                SOFTPLUS_LUT, x);
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {
    sfpu_reciprocal_init();
}

}  // namespace sfpu
}  // namespace ckernel
