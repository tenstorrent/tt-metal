// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * tanh_bw ULP Precision Tests
 *
 * Validates accuracy of ttnn::tanh_bw (tanh derivative = sech²(x)) across
 * the entire BFloat16 range.
 *
 * MATHEMATICAL FORMULA:
 * d/dx tanh(x) = sech²(x) = 1 - tanh²(x) = 1/cosh²(x)
 *
 * REFERENCE IMPLEMENTATION:
 * Uses fp64 1/cosh²(x) — avoids the catastrophic cancellation in 1 - tanh²(x).
 *
 * HARDWARE MODEL: DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
 * Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"
 * All denormal BF16 values are excluded from test inputs. NaN and Inf are excluded
 * per policy: "all inputs to the device should be filtered from the host not to have
 * any Inf/NaNs".
 *
 * KNOWN BUG (#35885):
 * The current implementation computes 1 - tanh²(x). When tanh(x) saturates to ±1.0
 * in BF16 (|x| > ~3.4), tanh² = 1.0 exactly, so 1 - 1 = 0. The true sech²(x) values
 * (e.g., 0.0013 at x=4) are perfectly representable in BF16, so the implementation
 * produces incorrect zeros. Max ULP = 15,139.
 *
 * BATCHED TESTING PATTERN:
 * Tests that sweep all ~65,000 BF16 values use batched tensor operations:
 *   1. Collect all valid BF16 values into a vector
 *   2. Pad to tile boundary (multiple of 32x32=1024)
 *   3. Create single tensor: Tensor::from_vector(data, TensorSpec).to_device(device)
 *   4. Call operation ONCE on the entire tensor
 *   5. Process results from output vector
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhBwUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <map>
#include <vector>
#include <iomanip>
#include <sstream>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// =============================================================================
// BFloat16 ULP Calculator
// =============================================================================

namespace bf16_ulp_tanh_bw {

constexpr uint16_t BF16_EXP_MASK = 0x7F80;
constexpr uint16_t BF16_MANTISSA_MASK = 0x007F;
constexpr uint16_t BF16_SIGN_MASK = 0x8000;
constexpr uint16_t BF16_POS_INF = 0x7F80;
constexpr uint16_t BF16_NEG_INF = 0xFF80;

inline uint16_t float_to_bf16_bits(float f) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    return static_cast<uint16_t>(f32_bits >> 16);
}

inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

inline bool is_bf16_denormal(uint16_t bits) { return ((bits >> 7) & 0xFF) == 0 && (bits & BF16_MANTISSA_MASK) != 0; }

inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;
    }
    if (bits == 0x8000) {
        return 0x0000;  // -0 -> +0
    }
    return bits;
}

inline float bf16_daz_normalize(float f) { return bf16_bits_to_float(bf16_daz_normalize(float_to_bf16_bits(f))); }

inline int32_t bf16_value_order_index_daz(uint16_t bits) {
    bits = bf16_daz_normalize(bits);
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & BF16_MANTISSA_MASK;
    if (exp == 0xFF && mantissa != 0) {
        return -1;  // NaN
    }
    if (bits == BF16_POS_INF) {
        return 65281;
    }
    if (bits == BF16_NEG_INF) {
        return -1;
    }
    if (bits == 0x0000) {
        return 32640;  // Zero
    }
    if (bits & BF16_SIGN_MASK) {
        return 0x7FFF - (bits & 0x7FFF);
    }
    return 32640 + bits - BF16_MANTISSA_MASK;
}

inline int32_t ulp_distance_bf16_daz(float a, float b) {
    uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
    uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));
    int32_t idx_a = bf16_value_order_index_daz(a_bits);
    int32_t idx_b = bf16_value_order_index_daz(b_bits);
    if (idx_a < 0 || idx_b < 0) {
        return -1;
    }
    return std::abs(idx_a - idx_b);
}

/**
 * Collect all valid (non-NaN, non-Inf, non-denormal) BF16 values.
 */
inline std::vector<float> all_valid_bf16_values() {
    std::vector<float> values;
    values.reserve(65536);
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & BF16_EXP_MASK) == BF16_EXP_MASK) {
            continue;  // NaN/Inf
        }
        if (is_bf16_denormal(bf16_bits)) {
            continue;  // denormals
        }
        values.push_back(bf16_bits_to_float(bf16_bits));
    }
    return values;
}

}  // namespace bf16_ulp_tanh_bw

// =============================================================================
// tanh derivative reference: sech²(x) = 1/cosh²(x)
// =============================================================================

/**
 * Exact sech²(x) using fp64.
 * Uses 1/cosh²(x) to avoid catastrophic cancellation in 1 - tanh²(x).
 */
inline double sech2_exact(double x) {
    double cosh_x = std::cosh(x);
    return 1.0 / (cosh_x * cosh_x);
}

/**
 * Expected BF16 sech²(x) with DAZ+FTZ applied.
 */
inline float sech2_expected_bf16_daz(float x) {
    float x_daz = bf16_ulp_tanh_bw::bf16_daz_normalize(x);
    double result = sech2_exact(x_daz);
    return bf16_ulp_tanh_bw::bf16_daz_normalize(static_cast<float>(result));
}

/**
 * Expected BF16 tanh_bw output: grad * sech²(x), with DAZ+FTZ.
 */
inline float tanh_bw_expected_bf16_daz(float grad, float x) {
    float grad_daz = bf16_ulp_tanh_bw::bf16_daz_normalize(grad);
    float x_daz = bf16_ulp_tanh_bw::bf16_daz_normalize(x);
    double result = static_cast<double>(grad_daz) * sech2_exact(x_daz);
    return bf16_ulp_tanh_bw::bf16_daz_normalize(static_cast<float>(result));
}

// =============================================================================
// Test Fixture
// =============================================================================

class TanhBwUlpTest : public TTNNFixtureWithDevice {};

/**
 * Run tanh_bw on device with a single value, return the output.
 */
float run_tanh_bw_single(tt::tt_metal::distributed::MeshDevice& device, float input_val, float grad_val = 1.0f) {
    ttnn::Shape shape({1, 1, 32, 32});
    auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto grad_tensor = ttnn::full(shape, grad_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    auto results = ttnn::tanh_bw(grad_tensor, input_tensor);
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();
    return static_cast<float>(output_vec[0]);
}

/**
 * Run tanh_bw on device with a batched vector of inputs (grad=1.0), return output vector.
 */
std::vector<::bfloat16> run_tanh_bw_batched(
    tt::tt_metal::distributed::MeshDevice* device, const std::vector<float>& input_values) {
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((input_values.size() + tile_size - 1) / tile_size) * tile_size;

    std::vector<::bfloat16> bf16_inputs;
    std::vector<::bfloat16> bf16_grads;
    bf16_inputs.reserve(padded_size);
    bf16_grads.reserve(padded_size);

    for (size_t i = 0; i < padded_size; ++i) {
        float x = (i < input_values.size()) ? input_values[i] : 0.0f;
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));
    }

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device);

    auto results = ttnn::tanh_bw(grad_tensor, input_tensor);
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    return output_cpu.to_vector<::bfloat16>();
}

// =============================================================================
// Smoke tests: correctness guards for key points
// =============================================================================

// sech²(0) = 1 exactly. Catches broken constant, wrong formula, or kernel crash.
TEST_F(TanhBwUlpTest, DerivativeAtZero) {
    float actual = run_tanh_bw_single(*device_, 0.0f);
    float expected = sech2_expected_bf16_daz(0.0f);
    int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);

    log_debug(tt::LogTest, "x=0: expected={}, actual={}, ULP={}", expected, actual, ulp);

    EXPECT_NEAR(expected, 1.0f, 1e-6) << "Reference sech²(0) should be 1.0";
    EXPECT_LE(ulp, 2) << "sech²(0) should be ~1.0 with low ULP error";
}

// sech²(x) → 0 for large |x|. Catches broken saturation threshold.
TEST_F(TanhBwUlpTest, DerivativeAtPositiveValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {0.5f, 2},
        {1.0f, 2},
        {2.0f, 2},
        {3.0f, 2},
        {5.0f, 2},
        {10.0f, 2},
    };
    for (const auto& [x, max_ulp] : test_cases) {
        float actual = run_tanh_bw_single(*device_, x);
        float expected = sech2_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, max_ulp) << "sech²(" << x << ") ULP too high";
    }
}

// sech²(x) is symmetric: sech²(-x) = sech²(x). Tests negative side independently.
TEST_F(TanhBwUlpTest, DerivativeAtNegativeValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {-0.5f, 2},
        {-1.0f, 2},
        {-2.0f, 2},
        {-3.0f, 2},
        {-4.0f, 2},
        {-5.0f, 2},
        {-8.0f, 2},
    };
    for (const auto& [x, max_ulp] : test_cases) {
        float actual = run_tanh_bw_single(*device_, x);
        float expected = sech2_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, max_ulp) << "sech²(" << x << ") ULP too high";
    }
}

// Near-zero region where sech²(x) ≈ 1 - x². Catches DAZ flush or denormal issues.
TEST_F(TanhBwUlpTest, DerivativeNearZero) {
    std::vector<float> values = {1e-6f, 1e-4f, 0.01f, 0.1f, -0.1f, -0.01f, -1e-4f};
    for (float x : values) {
        float actual = run_tanh_bw_single(*device_, x);
        float expected = sech2_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, 2) << "sech²(" << x << ") near-zero ULP too high";
    }
}

// Tests grad * sech²(x). Catches swapped grad/input or missing gradient multiplication.
TEST_F(TanhBwUlpTest, WithGradientScaling) {
    std::vector<std::tuple<float, float, int32_t>> test_cases = {
        {1.0f, 2.0f, 2},   // grad=2, x=1
        {-1.0f, 0.5f, 2},  // grad=0.5, x=-1
        {0.0f, 1.0f, 2},   // x=0
        {2.0f, -1.0f, 2},  // negative grad
    };
    for (const auto& [x, grad, max_ulp] : test_cases) {
        float actual = run_tanh_bw_single(*device_, x, grad);
        float expected = tanh_bw_expected_bf16_daz(grad, x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}, grad={}: expected={}, actual={}, ULP={}", x, grad, expected, actual, ulp);
        EXPECT_LE(ulp, max_ulp) << "tanh_bw(grad=" << grad << ", x=" << x << ") ULP too high";
    }
}

// =============================================================================
// Full BF16 sweep: per-segment ULP analysis
// =============================================================================

// Precision guard: per-segment analysis across all ~65K BF16 values.
// Segment boundaries match the implementation's piecewise regions and the
// saturation thresholds where the original 1-tanh² bug manifests.
TEST_F(TanhBwUlpTest, PerSegmentULPAnalysis) {
    auto input_values = bf16_ulp_tanh_bw::all_valid_bf16_values();
    const size_t valid_count = input_values.size();

    // Pad to tile boundary
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    auto output_vec = run_tanh_bw_batched(device_, input_values);

    struct RegionStats {
        std::string name;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
    };

    // Regions chosen to highlight the 1-tanh² cancellation zones
    std::vector<RegionStats> regions = {
        {"x < -10"},
        {"-10 <= x < -5"},
        {"-5 <= x < -4"},
        {"-4 <= x < -3"},
        {"-3 <= x < -2"},
        {"-2 <= x < -1"},
        {"-1 <= x < 0"},
        {"x == 0"},
        {"0 < x < 1"},
        {"1 <= x < 2"},
        {"2 <= x < 3"},
        {"3 <= x < 4"},
        {"4 <= x < 5"},
        {"5 <= x < 10"},
        {"x >= 10"},
    };

    int overall_max_ulp = 0;
    float overall_worst_x = 0;
    double overall_ulp_sum = 0;
    int overall_count = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_tanh_bw::bf16_bits_to_float(bf16_ulp_tanh_bw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = sech2_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        if (ulp < 0) {
            continue;
        }

        int idx;
        if (x < -10.0f) {
            idx = 0;
        } else if (x < -5.0f) {
            idx = 1;
        } else if (x < -4.0f) {
            idx = 2;
        } else if (x < -3.0f) {
            idx = 3;
        } else if (x < -2.0f) {
            idx = 4;
        } else if (x < -1.0f) {
            idx = 5;
        } else if (x < 0.0f) {
            idx = 6;
        } else if (x == 0.0f) {
            idx = 7;
        } else if (x < 1.0f) {
            idx = 8;
        } else if (x < 2.0f) {
            idx = 9;
        } else if (x < 3.0f) {
            idx = 10;
        } else if (x < 4.0f) {
            idx = 11;
        } else if (x < 5.0f) {
            idx = 12;
        } else if (x < 10.0f) {
            idx = 13;
        } else {
            idx = 14;
        }

        regions[idx].count++;
        regions[idx].ulp_sum += ulp;
        if (ulp > regions[idx].max_ulp) {
            regions[idx].max_ulp = ulp;
            regions[idx].worst_x = x;
        }
        overall_ulp_sum += ulp;
        overall_count++;
        if (ulp > overall_max_ulp) {
            overall_max_ulp = ulp;
            overall_worst_x = x;
        }
    }

    // Log results
    {
        std::ostringstream oss;
        oss << "\n============================================================\n"
            << "TANH_BW (sech²) PER-SEGMENT ULP ANALYSIS (DAZ+FTZ MODEL)\n"
            << "============================================================\n"
            << std::setw(20) << "Region" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
            << "Max ULP" << std::setw(15) << "Worst x\n"
            << std::string(69, '-') << "\n";
        for (const auto& r : regions) {
            if (r.count > 0) {
                oss << std::setw(20) << r.name << std::setw(10) << r.count << std::setw(12) << std::fixed
                    << std::setprecision(2) << (r.ulp_sum / r.count) << std::setw(12) << r.max_ulp << std::setw(15)
                    << std::scientific << std::setprecision(3) << r.worst_x << "\n";
            }
        }
        oss << std::string(69, '-') << "\n"
            << std::setw(20) << "OVERALL" << std::setw(10) << overall_count << std::setw(12) << std::fixed
            << std::setprecision(2) << (overall_ulp_sum / overall_count) << std::setw(12) << overall_max_ulp
            << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x << "\n"
            << "============================================================\n";
        log_debug(tt::LogTest, "{}", oss.str());
    }

    // Regression guards per region — target: max ULP <= 2 everywhere
    for (const auto& r : regions) {
        EXPECT_LE(r.max_ulp, 2) << "Region '" << r.name << "' max ULP " << r.max_ulp << " exceeds threshold 2"
                                << " (worst at x=" << r.worst_x << ")";
    }
    EXPECT_LE(overall_max_ulp, 2) << "Overall max ULP " << overall_max_ulp << " at x=" << overall_worst_x;
}

// =============================================================================
// Full BF16 sweep: cumulative ULP histogram
// =============================================================================

// Precision guard: ULP histogram across all ~65K BF16 values.
// Catches broad precision degradation that per-segment tests might miss.
TEST_F(TanhBwUlpTest, CumulativeULPDistribution) {
    auto input_values = bf16_ulp_tanh_bw::all_valid_bf16_values();
    const size_t valid_count = input_values.size();

    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    auto output_vec = run_tanh_bw_batched(device_, input_values);

    std::map<int32_t, int> ulp_histogram;
    int max_ulp = 0;
    float worst_x = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_tanh_bw::bf16_bits_to_float(bf16_ulp_tanh_bw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = sech2_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
        if (ulp >= 0) {
            ulp_histogram[ulp]++;
            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_x = x;
            }
        }
    }

    {
        std::ostringstream oss;
        oss << "\n============================================================\n"
            << "TANH_BW CUMULATIVE ULP DISTRIBUTION (DAZ+FTZ MODEL)\n"
            << "============================================================\n"
            << std::setw(10) << "ULP <=" << std::setw(12) << "Count" << std::setw(12) << "Percent\n"
            << std::string(34, '-') << "\n";

        std::vector<int> thresholds = {0, 1, 2, 3, 5, 10, 50, 100, 1000, 15000};
        for (int t : thresholds) {
            int count = 0;
            for (auto& [u, c] : ulp_histogram) {
                if (u <= t) {
                    count += c;
                }
            }
            oss << std::setw(10) << t << std::setw(12) << count << std::setw(11) << std::fixed << std::setprecision(2)
                << (100.0 * count / valid_count) << "%\n";
        }
        oss << std::string(34, '-') << "\n"
            << "Max ULP: " << max_ulp << " at x = " << worst_x << "\n"
            << "============================================================\n";
        log_debug(tt::LogTest, "{}", oss.str());
    }

    // Regression guards
    EXPECT_LE(max_ulp, 2) << "Max ULP " << max_ulp << " at x=" << worst_x << " exceeds threshold 2";

    int count_le_2 = 0;
    for (auto& [u, c] : ulp_histogram) {
        if (u <= 2) {
            count_le_2 += c;
        }
    }
    double pct_le_2 = 100.0 * count_le_2 / valid_count;
    EXPECT_GE(pct_le_2, 99.0) << "Only " << pct_le_2 << "% within 2 ULP (expected >= 99%)";
}

// =============================================================================
// Reference implementation verification
// =============================================================================

// Test infrastructure guard: validates the fp64 golden reference.
// If this fails, all other tests' expected values are wrong.
TEST_F(TanhBwUlpTest, ReferenceImplementationVerification) {
    // sech²(0) = 1
    double s0 = sech2_exact(0.0);
    EXPECT_NEAR(s0, 1.0, 1e-10) << "sech²(0) should be 1.0";

    // sech²(large) → 0
    double s_large = sech2_exact(100.0);
    EXPECT_NEAR(s_large, 0.0, 1e-6) << "sech²(100) should approach 0";

    // sech²(-x) = sech²(x) — symmetry
    double s_pos = sech2_exact(2.0);
    double s_neg = sech2_exact(-2.0);
    EXPECT_NEAR(s_pos, s_neg, 1e-15) << "sech² must be symmetric";

    // sech²(1) = 1/cosh²(1) ≈ 0.4199743...
    double s1 = sech2_exact(1.0);
    EXPECT_NEAR(s1, 0.4199743, 1e-6) << "sech²(1) ≈ 0.4199743";

    log_debug(tt::LogTest, "sech²(0) = {} (expected 1.0)", s0);
    log_debug(tt::LogTest, "sech²(1) = {} (expected ~0.4200)", s1);
    log_debug(tt::LogTest, "sech²(100) = {} (expected ~0.0)", s_large);
    log_debug(tt::LogTest, "sech²(2) = {}, sech²(-2) = {} (symmetry)", s_pos, s_neg);
}

// =============================================================================
// Saturation region analysis — the core of bug #35885
// =============================================================================

// Diagnostic: the transition region |x| ∈ [3, 5] where 1-tanh²(x) fails.
// sech²(3.34) ≈ 0.0049 — perfectly representable in BF16, but the buggy
// implementation returns 0.0 because tanh(3.34) saturates to 1.0 in BF16.
TEST_F(TanhBwUlpTest, SaturationRegionAnalysis) {
    std::vector<float> critical_values = {
        -5.0f,
        -4.5f,
        -4.0f,
        -3.75f,
        -3.5f,
        -3.34375f,
        -3.0f,
        -2.5f,
        2.5f,
        3.0f,
        3.34375f,
        3.5f,
        3.75f,
        4.0f,
        4.5f,
        5.0f,
    };

    int max_ulp_found = 0;
    float worst_x = 0;

    {
        std::ostringstream oss;
        oss << "\n========================================\n"
            << "SATURATION REGION ANALYSIS\n"
            << "(Critical region for bug #35885)\n"
            << "========================================\n"
            << std::setw(10) << "x" << std::setw(15) << "Expected" << std::setw(15) << "Actual" << std::setw(10)
            << "ULP" << std::setw(15) << "Abs Error\n"
            << std::string(65, '-') << "\n";

        for (float x : critical_values) {
            float actual = run_tanh_bw_single(*device_, x);
            float expected = sech2_expected_bf16_daz(x);
            int32_t ulp = bf16_ulp_tanh_bw::ulp_distance_bf16_daz(actual, expected);
            float abs_error = std::abs(actual - expected);

            oss << std::setw(10) << std::fixed << std::setprecision(4) << x << std::setw(15) << std::scientific
                << std::setprecision(4) << expected << std::setw(15) << actual << std::setw(10) << ulp << std::setw(15)
                << abs_error << "\n";

            if (ulp > max_ulp_found) {
                max_ulp_found = ulp;
                worst_x = x;
            }
        }
        oss << std::string(65, '-') << "\n"
            << "Max ULP in saturation region: " << max_ulp_found << " at x=" << worst_x << "\n";
        log_debug(tt::LogTest, "{}", oss.str());
    }

    EXPECT_LE(max_ulp_found, 2) << "Saturation region max ULP " << max_ulp_found << " at x=" << worst_x;
}

}  // namespace ttnn::test
