// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sfpu golden functions
#include <cmath>
#include <numbers>
#include <vector>
#include <cstdint>
#include <functional>
#include <random>
#include <map>

#include <tt-metalium/bfloat16.hpp>

inline float exponential(float x) { return std::exp(x); }

inline float reciprocal(float x) { return 1 / x; }

inline float gelu(float x) {
    static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
    auto x3 = x * x * x;
    return x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
}

inline float relu(float x) { return fmaxf(x, 0.0f); }

inline float ref_sqrt(float x) { return sqrtf(x); }

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float ref_log(float x) { return logf(x); }

inline float ref_log10(float x) { return ref_log(x) * std::numbers::log10e; }

inline float ref_log2(float x) { return ref_log(x) * std::numbers::log2e_v<float>; }

inline float ref_tanh(float x) { return std::tanh(x); }

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16_0_2(size_t num_bytes, int seed) {
    return create_random_vector_of_bfloat16(num_bytes, 2.0f, seed);  // 0.0f..2.0f
}

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16_0_2_plus_1(uint32_t num_bytes, int seed) {
    return create_random_vector_of_bfloat16(num_bytes, 2.0f, seed, 1.0f);  // 0.0f..2.0f
}

namespace helper {
template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
}  // namespace helper

inline float ref_sign(float x) { return helper::sgn(x); }

inline float ref_square(float x) { return x * x; }

inline float ref_abs(float x) { return std::abs(x); }

inline float ref_identity(float x) { return x; }

inline std::vector<uint32_t> sfpu(const std::vector<uint32_t>& src, const std::function<float(float)>& sfpu_func) {
    std::vector<uint32_t> dst;

    for (uint32_t el : src) {
        uint32_t top = el & 0xffff0000;
        uint32_t bottom = el << 16;

        float top_ = *reinterpret_cast<float*>(&top);
        float bottom_ = *reinterpret_cast<float*>(&bottom);

        float exp_top = sfpu_func(top_);
        float exp_bottom = sfpu_func(bottom_);

        bfloat16 bfp16_top = bfloat16(exp_top);
        bfloat16 bfp16_bottom = bfloat16(exp_bottom);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

inline std::vector<uint32_t> create_random_binary_vector_of_bfloat16(size_t num_bytes, int seed) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, 1), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (std::uint32_t& num : vec) {
        float num_1_float = rand_float();
        float num_2_float = rand_float();

        num_1_float = (num_1_float > 0.5);
        num_2_float = (num_2_float > 0.5);

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        num = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
    return vec;
}

// Helper functions
inline std::vector<uint32_t> create_random_ones_and_twos_vector_of_bfloat16(uint32_t num_bytes, int seed) {
    // Used for reciprocal, since binary vectors are filled with 0s and 1s, and recip of 0 is undefined,
    // so then we just generate a vector of ones and twos

    std::vector<uint32_t> src = create_random_binary_vector_of_bfloat16(num_bytes, seed);

    std::vector<uint32_t> dst;

    for (uint32_t el : src) {
        uint32_t top = el & 0xffff0000;
        uint32_t bottom = el << 16;

        float top_ = *reinterpret_cast<float*>(&top);
        float bottom_ = *reinterpret_cast<float*>(&bottom);

        float top_plus_one = 1 + top_;
        float bottom_plus_one = 1 + bottom_;

        bfloat16 bfp16_top = bfloat16(top_plus_one);
        bfloat16 bfp16_bottom = bfloat16(bottom_plus_one);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

// Comparison functions
inline bool equal_within_two_sig_figs(float a, float b) { return equal_within_n_sig_figs(a, b, 2); }

inline bool equal_within_absolute_tolerance(float a, float b, float tol) { return std::abs(a - b) < tol; }

inline bool equal_within_absolute_tolerance_of_0p03(float a, float b) {
    return equal_within_absolute_tolerance(a, b, 0.03);
}

inline bool is_close_0p015(float a, float b) { return is_close(a, b, 0.015f); }

inline bool is_close_rtol_0p06_atol_0p006(float a, float b) { return is_close(a, b, 0.06f, 0.006f); }

inline bool is_close_rtol_0p175_atol_0p1(float a, float b) { return is_close(a, b, 0.175f, 0.1f); }

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16_1_1(size_t num_bytes, int seed) {
    return create_random_vector_of_bfloat16(num_bytes, 2.0f, seed, -1.0f);  // -1.0..1.0
}

// SFPU maps -> relevant kernels, golden functions, comparison functions
static std::vector<std::string> sfpu_op = {
    "relu",
    "exponential",
    "reciprocal",
    "gelu",
    "sqrt",
    "sigmoid",
    "log",
    "log2",
    "log10",
    "tanh",
    "sign",
    "abs",
    "square",
    "identity"};

const std::map<std::string, std::function<float(float)>> sfpu_op_to_function = {
    {"relu", relu},
    {"exponential", exponential},
    {"reciprocal", reciprocal},
    {"gelu", gelu},
    {"sqrt", ref_sqrt},
    {"sigmoid", sigmoid},
    {"log", ref_log},
    {"log2", ref_log2},
    {"log10", ref_log10},
    {"tanh", ref_tanh},
    {"sign", ref_sign},
    {"abs", ref_abs},
    {"square", ref_square},
    {"identity", ref_identity}};

const std::map<std::string, std::function<std::vector<uint32_t>(uint32_t num_bytes, int seed)>> sfpu_op_to_init_func = {
    {"relu", create_random_vector_of_bfloat16_1_1},
    {"exponential", create_random_binary_vector_of_bfloat16},
    {"reciprocal", create_random_ones_and_twos_vector_of_bfloat16},
    {"gelu", create_random_binary_vector_of_bfloat16},
    {"sqrt", create_random_vector_of_bfloat16_0_2},
    {"sigmoid", create_random_vector_of_bfloat16_1_1},
    {"log", create_random_vector_of_bfloat16_0_2_plus_1},
    {"log2", create_random_vector_of_bfloat16_0_2_plus_1},
    {"log10", create_random_vector_of_bfloat16_0_2_plus_1},
    {"tanh", create_random_vector_of_bfloat16_1_1},
    {"sign", create_random_vector_of_bfloat16_1_1},
    {"abs", create_random_vector_of_bfloat16_1_1},
    {"square", create_random_vector_of_bfloat16_1_1},
    {"identity", create_random_vector_of_bfloat16_1_1}};

const std::map<std::string, std::function<bool(float a, float b)>> sfpu_op_to_comparison_function = {
    {"exponential", equal_within_two_sig_figs},
    {"reciprocal", equal_within_absolute_tolerance_of_0p03},
    {"gelu", is_close_0p015},
    {"relu", is_close_0p015},
    {"sqrt", is_close_rtol_0p06_atol_0p006},
    {"sigmoid", is_close_rtol_0p06_atol_0p006},
    {"log", is_close_rtol_0p06_atol_0p006},
    {"log2", is_close_rtol_0p06_atol_0p006},
    {"log10", is_close_rtol_0p06_atol_0p006},
    {"tanh", is_close_rtol_0p175_atol_0p1},
    {"sign", is_close_rtol_0p175_atol_0p1},
    {"abs", is_close_rtol_0p175_atol_0p1},
    {"square", is_close_rtol_0p175_atol_0p1},
    {"identity", is_close_rtol_0p175_atol_0p1}};
