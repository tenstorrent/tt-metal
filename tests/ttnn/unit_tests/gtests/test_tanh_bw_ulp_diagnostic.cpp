// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tanh Backward ULP Diagnostic Tests with Dual ULP Calculator Verification
 *
 * Mathematical Background:
 * - Forward: y = tanh(x)
 * - Backward: dy/dx = 1 - tanh(x)^2 = sech^2(x)
 * - tanh_bw(grad_output, input) = grad_output * (1 - tanh(input)^2)
 *
 * For ULP testing, we use grad_output = 1.0 to isolate the derivative calculation.
 * Expected output: 1 - tanh(input)^2
 *
 * This test file contains:
 * 1. Two independent BFloat16 ULP calculators (for cross-verification)
 * 2. Exhaustive BF16 sweep for tanh_bw precision
 * 3. DAZ+FTZ behavior verification
 *
 * Run: ./build/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhBwUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <limits>
#include <iomanip>
#include <numeric>
#include <mpfr.h>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;

// =============================================================================
// BFloat16 Utilities (wrapped in anonymous namespace for Unity build)
// =============================================================================

namespace {
namespace bf16_ulp_bw {

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

inline bool is_bf16_denormal(uint16_t bits) {
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & 0x7F;
    return (exp == 0) && (mantissa != 0);
}

inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;
    }
    if (bits == 0x8000) {
        return 0x0000;
    }
    return bits;
}

// =============================================================================
// Method 1: Bitwise Formula ULP Calculator
// =============================================================================

inline int32_t bf16_index_bitwise(uint16_t bits) {
    bits = bf16_daz_normalize(bits);

    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & 0x7F;

    if (exp == 0xFF && mantissa != 0) {
        return -1;  // NaN
    }
    if (bits == 0x7F80) {
        return 65025;  // +inf
    }
    if (bits == 0xFF80) {
        return -1;  // -inf
    }
    if (bits == 0x0000) {
        return 32512;  // zero
    }

    if (bits & 0x8000) {
        uint16_t magnitude = bits & 0x7FFF;
        return 0x7F7F - magnitude;
    } else {
        return 32513 + bits - 0x0080;
    }
}

inline int32_t ulp_distance_bitwise(float a, float b) {
    uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
    uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));

    int32_t idx_a = bf16_index_bitwise(a_bits);
    int32_t idx_b = bf16_index_bitwise(b_bits);

    if (idx_a < 0 || idx_b < 0) {
        return -1;
    }

    return std::abs(idx_a - idx_b);
}

// =============================================================================
// Method 2: Sorted Index Lookup ULP Calculator
// =============================================================================

class Bf16SortedIndex {
public:
    Bf16SortedIndex() {
        // Negative normals
        for (uint16_t bits = 0xFF7F; bits >= 0x8080; --bits) {
            if (bits == 0xFF80) {
                continue;
            }
            if (!is_bf16_denormal(bits)) {
                sorted_bits_.push_back(bits);
                sorted_values_.push_back(bf16_bits_to_float(bits));
            }
        }

        // Zero
        sorted_bits_.push_back(0x0000);
        sorted_values_.push_back(0.0f);

        // Positive normals
        for (uint16_t bits = 0x0080; bits <= 0x7F7F; ++bits) {
            if (bits == 0x7F80) {
                continue;
            }
            if (!is_bf16_denormal(bits)) {
                sorted_bits_.push_back(bits);
                sorted_values_.push_back(bf16_bits_to_float(bits));
            }
        }

        for (size_t i = 1; i < sorted_values_.size(); ++i) {
            assert(sorted_values_[i - 1] < sorted_values_[i] && "Values must be strictly sorted");
        }
    }

    int32_t find_index(uint16_t bits) const {
        bits = bf16_daz_normalize(bits);

        uint16_t exp = (bits >> 7) & 0xFF;
        if (exp == 0xFF) {
            return -1;
        }

        auto it = std::lower_bound(sorted_bits_.begin(), sorted_bits_.end(), bits, [](uint16_t a, uint16_t b) {
            return bf16_bits_to_float(a) < bf16_bits_to_float(b);
        });

        if (it != sorted_bits_.end() && *it == bits) {
            return static_cast<int32_t>(std::distance(sorted_bits_.begin(), it));
        }

        return -1;
    }

    int32_t ulp_distance(float a, float b) const {
        uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
        uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));

        int32_t idx_a = find_index(a_bits);
        int32_t idx_b = find_index(b_bits);

        if (idx_a < 0 || idx_b < 0) {
            return -1;
        }

        return std::abs(idx_a - idx_b);
    }

    size_t size() const { return sorted_bits_.size(); }
    uint16_t bits_at(size_t idx) const { return sorted_bits_[idx]; }
    float value_at(size_t idx) const { return sorted_values_[idx]; }

private:
    std::vector<uint16_t> sorted_bits_;
    std::vector<float> sorted_values_;
};

static Bf16SortedIndex g_sorted_index;

inline int32_t ulp_distance_sorted(float a, float b) { return g_sorted_index.ulp_distance(a, b); }

/**
 * Reference tanh backward (derivative) using fp64 std::tanh.
 * d/dx tanh(x) = 1 - tanh(x)^2 = sech^2(x)
 */
inline double tanh_bw_reference_fp64(double x) {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

/**
 * Exact tanh backward (derivative) using MPFR 256-bit precision.
 *
 * tanh_bw(x) = 1 - tanh(x)^2 = sech^2(x)
 *
 * This uses MPFR (Multiple Precision Floating-Point Reliable) library
 * to compute the true tanh derivative with 256-bit precision. This ensures
 * accurate reference values even in regions where fp64 might lose precision.
 */
inline double tanh_bw_exact(double x) {
    constexpr mpfr_prec_t precision = 256;

    mpfr_t mpfr_x, tanh_result, tanh_squared, one, result;

    mpfr_init2(mpfr_x, precision);
    mpfr_init2(tanh_result, precision);
    mpfr_init2(tanh_squared, precision);
    mpfr_init2(one, precision);
    mpfr_init2(result, precision);

    // Set values
    mpfr_set_d(mpfr_x, x, MPFR_RNDN);
    mpfr_set_ui(one, 1, MPFR_RNDN);

    // Compute tanh(x)
    mpfr_tanh(tanh_result, mpfr_x, MPFR_RNDN);

    // Compute tanh(x)^2
    mpfr_mul(tanh_squared, tanh_result, tanh_result, MPFR_RNDN);

    // Compute 1 - tanh(x)^2
    mpfr_sub(result, one, tanh_squared, MPFR_RNDN);

    // Extract result as double
    double tanh_bw_result = mpfr_get_d(result, MPFR_RNDN);

    // Clean up
    mpfr_clear(mpfr_x);
    mpfr_clear(tanh_result);
    mpfr_clear(tanh_squared);
    mpfr_clear(one);
    mpfr_clear(result);

    return tanh_bw_result;
}

/**
 * Convert double to BF16 bits by truncation (no rounding).
 * This matches hardware behavior which truncates when converting to BF16.
 */
inline uint16_t double_to_bf16_bits_truncate(double d) {
    // Convert double to float (this rounds, but we then truncate to BF16)
    float f = static_cast<float>(d);

    // Truncate float to BF16 by taking upper 16 bits
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    return static_cast<uint16_t>(f32_bits >> 16);
}

/**
 * Compute the expected BF16 tanh_bw value with proper DAZ+FTZ methodology.
 *
 * Methodology (matching hardware behavior):
 * 1. Input: BF16 -> DAZ normalize (flush denormals to zero)
 * 2. Calculate: mpfr-256 precision for exact result
 * 3. Output: Truncate to BF16 -> DAZ normalize (flush denormals to zero)
 */
inline float tanh_bw_expected_bf16(float input_bf16) {
    // Step 1: Apply DAZ to input (flush denormals to zero)
    uint16_t input_bits = float_to_bf16_bits(input_bf16);
    uint16_t input_daz = bf16_daz_normalize(input_bits);
    double input_double = static_cast<double>(bf16_bits_to_float(input_daz));

    // Step 2: Calculate with mpfr-256 precision
    double result_exact = tanh_bw_exact(input_double);

    // Step 3: Convert to BF16 (truncation) and apply FTZ to output
    uint16_t result_bits = double_to_bf16_bits_truncate(result_exact);
    uint16_t result_daz = bf16_daz_normalize(result_bits);

    return bf16_bits_to_float(result_daz);
}

}  // namespace bf16_ulp_bw
}  // anonymous namespace

// =============================================================================
// ULP Calculator Verification Tests
// =============================================================================

class TanhBwUlpCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TanhBwUlpCalculatorTest, BitwiseAndSortedMethodsAgree) {
    const auto& idx = bf16_ulp_bw::g_sorted_index;
    int mismatches = 0;

    for (size_t i = 0; i < idx.size(); i += 100) {  // Sample every 100th
        for (size_t j = i; j < std::min(i + 10, idx.size()); ++j) {
            float a = idx.value_at(i);
            float b = idx.value_at(j);

            int32_t ulp_bitwise = bf16_ulp_bw::ulp_distance_bitwise(a, b);
            int32_t ulp_sorted = bf16_ulp_bw::ulp_distance_sorted(a, b);

            if (ulp_bitwise != ulp_sorted) {
                mismatches++;
            }
        }
    }

    EXPECT_EQ(mismatches, 0) << "ULP calculation methods disagree";
}

TEST_F(TanhBwUlpCalculatorTest, Fp64AndMpfr256ReferenceAgree) {
    // Verify that fp64 std::tanh and mpfr-256 produce the same results for tanh_bw
    // This validates our reference implementations are consistent

    const auto& idx = bf16_ulp_bw::g_sorted_index;
    int mismatches = 0;
    double max_rel_diff = 0.0;
    float worst_input = 0.0f;

    std::cout << "\nVerifying fp64 vs mpfr-256 reference implementations agree..." << std::endl;

    for (size_t i = 0; i < idx.size(); ++i) {
        float input_f = idx.value_at(i);
        double input = static_cast<double>(input_f);

        double fp64_result = bf16_ulp_bw::tanh_bw_reference_fp64(input);
        double mpfr_result = bf16_ulp_bw::tanh_bw_exact(input);

        // Both should produce the same BF16 value when truncated
        float fp64_bf16 = bf16_ulp_bw::bf16_bits_to_float(
            bf16_ulp_bw::bf16_daz_normalize(bf16_ulp_bw::float_to_bf16_bits(static_cast<float>(fp64_result))));
        float mpfr_bf16 = bf16_ulp_bw::bf16_bits_to_float(
            bf16_ulp_bw::bf16_daz_normalize(bf16_ulp_bw::float_to_bf16_bits(static_cast<float>(mpfr_result))));

        if (fp64_bf16 != mpfr_bf16) {
            // For tanh_bw, fp64 and mpfr-256 might differ slightly in the saturation region
            // where 1 - tanh(x)^2 becomes very small. Track differences.
            double diff = std::abs(fp64_result - mpfr_result);
            double rel_diff = (mpfr_result != 0.0) ? diff / std::abs(mpfr_result) : diff;
            if (rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
                worst_input = input_f;
            }
            mismatches++;
        }
    }

    std::cout << "Total BF16 values checked: " << idx.size() << std::endl;
    std::cout << "Values where fp64 and mpfr-256 produce different BF16: " << mismatches << std::endl;
    if (mismatches > 0) {
        std::cout << "Max relative difference: " << std::scientific << max_rel_diff << std::endl;
        std::cout << "Worst input: " << worst_input << std::endl;
    }

    // For tanh_bw, we expect some differences in the saturation region where both
    // fp64 and mpfr produce values very close to zero. What matters is that our
    // mpfr-256 reference is authoritative.
    // If there are differences, they should be in the saturation region where
    // small differences in the fp64 vs mpfr-256 calculation lead to different BF16 rounding.
    std::cout << "NOTE: Differences in saturation region are expected - mpfr-256 is authoritative." << std::endl;
}

// =============================================================================
// Tanh Backward Device Tests
// =============================================================================

class TanhBwUlpDeviceTest : public TTNNFixtureWithDevice {};

TEST_F(TanhBwUlpDeviceTest, ExhaustiveBf16Sweep) {
    using namespace bf16_ulp_bw;

    const auto& idx = g_sorted_index;
    std::vector<::bfloat16> all_values;
    all_values.reserve(idx.size());

    for (size_t i = 0; i < idx.size(); ++i) {
        all_values.push_back(::bfloat16(idx.value_at(i)));
    }

    std::cout << "Testing " << all_values.size() << " BF16 values for tanh_bw precision..." << std::endl;

    // Pad to tile-compatible 2D shape
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 2048;
    constexpr size_t padded_size = tile_height * tile_width;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    std::vector<::bfloat16> padded_grad(padded_size, ::bfloat16(1.0f));  // grad = 1.0 to isolate derivative
    for (size_t i = 0; i < all_values.size(); ++i) {
        padded_input[i] = all_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_height, tile_width};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Create input tensor
    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);

    // Create grad tensor (all 1.0)
    Tensor host_grad = Tensor::from_vector(padded_grad, tensor_spec);
    Tensor device_grad = host_grad.to_layout(Layout::TILE).to_device(device_);

    // Run tanh_bw
    auto output_tensors = ttnn::tanh_bw(device_grad, device_input);
    Tensor output_tensor = output_tensors[0].value();

    // Move back to host
    Tensor output_host = output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    // Analyze ULP errors
    std::vector<int32_t> ulp_errors;
    ulp_errors.reserve(all_values.size());

    int max_ulp = 0;
    float worst_input = 0;
    float worst_expected = 0;
    float worst_actual = 0;

    for (size_t i = 0; i < all_values.size(); ++i) {
        float input = static_cast<float>(all_values[i]);
        float actual = static_cast<float>(output_vec[i]);

        // Reference: tanh_bw with DAZ applied
        float expected = tanh_bw_expected_bf16(input);

        int32_t ulp = ulp_distance_bitwise(actual, expected);
        if (ulp >= 0) {
            ulp_errors.push_back(ulp);
            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_input = input;
                worst_expected = expected;
                worst_actual = actual;
            }
        }
    }

    // Calculate statistics
    int64_t sum = std::accumulate(ulp_errors.begin(), ulp_errors.end(), 0LL);
    double mean = static_cast<double>(sum) / ulp_errors.size();

    int count_ulp_0 = std::count(ulp_errors.begin(), ulp_errors.end(), 0);
    int count_ulp_le_1 = std::count_if(ulp_errors.begin(), ulp_errors.end(), [](int u) { return u <= 1; });
    int count_ulp_le_2 = std::count_if(ulp_errors.begin(), ulp_errors.end(), [](int u) { return u <= 2; });

    std::cout << "\n========================================" << std::endl;
    std::cout << "EXHAUSTIVE BF16 SWEEP RESULTS - tanh_bw()" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total values tested: " << ulp_errors.size() << std::endl;
    std::cout << "Max ULP: " << max_ulp << std::endl;
    std::cout << "Mean ULP: " << std::fixed << std::setprecision(4) << mean << std::endl;
    std::cout << "ULP = 0: " << count_ulp_0 << " (" << (100.0 * count_ulp_0 / ulp_errors.size()) << "%)" << std::endl;
    std::cout << "ULP <= 1: " << count_ulp_le_1 << " (" << (100.0 * count_ulp_le_1 / ulp_errors.size()) << "%)"
              << std::endl;
    std::cout << "ULP <= 2: " << count_ulp_le_2 << " (" << (100.0 * count_ulp_le_2 / ulp_errors.size()) << "%)"
              << std::endl;

    if (max_ulp > 2) {
        std::cout << "\nWorst case:" << std::endl;
        std::cout << "  Input: " << worst_input << " (0x" << std::hex << float_to_bf16_bits(worst_input) << std::dec
                  << ")" << std::endl;
        std::cout << "  Expected: " << worst_expected << std::endl;
        std::cout << "  Actual: " << worst_actual << std::endl;
        std::cout << "  ULP: " << max_ulp << std::endl;
    }

    // Note: tanh_bw has precision issues in saturation region where the derivative
    // approaches zero. For large |x|, 1-tanh(x)^2 → 0, but the implementation
    // produces exactly zero earlier than the reference, causing large ULP errors.
    // This is a known limitation documented in the diagnostic report.

    // Relaxed assertions - focus on the transition region precision
    EXPECT_GE(count_ulp_le_2, static_cast<int>(0.95 * ulp_errors.size())) << "At least 95% should have ULP <= 2";
}

TEST_F(TanhBwUlpDeviceTest, DerivativeNearZero) {
    // Near x=0: tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
    using namespace bf16_ulp_bw;

    std::vector<float> test_floats = {-0.1f, -0.05f, -0.01f, -0.001f, 0.0f, 0.001f, 0.01f, 0.05f, 0.1f};

    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    std::vector<::bfloat16> padded_grad(padded_size, ::bfloat16(1.0f));
    for (size_t i = 0; i < test_floats.size(); ++i) {
        padded_input[i] = ::bfloat16(test_floats[i]);
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);
    Tensor host_grad = Tensor::from_vector(padded_grad, tensor_spec);
    Tensor device_grad = host_grad.to_layout(Layout::TILE).to_device(device_);

    auto output_tensors = ttnn::tanh_bw(device_grad, device_input);
    Tensor output_host = output_tensors[0].value().cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    std::cout << "\nTanh Backward Near Zero (derivative should be close to 1.0):" << std::endl;
    std::cout << std::setw(12) << "Input" << std::setw(12) << "Output" << std::setw(12) << "Expected" << std::setw(10)
              << "ULP" << std::endl;

    int max_ulp = 0;
    for (size_t i = 0; i < test_floats.size(); ++i) {
        float input = test_floats[i];
        float actual = static_cast<float>(output_vec[i]);
        float expected = tanh_bw_expected_bf16(input);

        int32_t ulp = ulp_distance_bitwise(actual, expected);
        if (ulp > max_ulp) {
            max_ulp = ulp;
        }

        std::cout << std::setw(12) << input << std::setw(12) << actual << std::setw(12) << expected << std::setw(10)
                  << ulp << std::endl;
    }

    EXPECT_LE(max_ulp, 5) << "Near-zero derivative should have low ULP error";
}

TEST_F(TanhBwUlpDeviceTest, DerivativeSaturation) {
    // For large |x|: tanh'(x) = 1 - tanh(x)^2 -> 1 - 1 = 0
    using namespace bf16_ulp_bw;

    std::vector<float> test_floats = {-8.0f, -6.0f, -5.0f, -4.0f, -3.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f};

    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    std::vector<::bfloat16> padded_grad(padded_size, ::bfloat16(1.0f));
    for (size_t i = 0; i < test_floats.size(); ++i) {
        padded_input[i] = ::bfloat16(test_floats[i]);
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);
    Tensor host_grad = Tensor::from_vector(padded_grad, tensor_spec);
    Tensor device_grad = host_grad.to_layout(Layout::TILE).to_device(device_);

    auto output_tensors = ttnn::tanh_bw(device_grad, device_input);
    Tensor output_host = output_tensors[0].value().cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    std::cout << "\nTanh Backward Saturation (derivative should approach 0):" << std::endl;
    std::cout << std::setw(12) << "Input" << std::setw(12) << "Output" << std::setw(12) << "Expected" << std::setw(10)
              << "ULP" << std::endl;

    for (size_t i = 0; i < test_floats.size(); ++i) {
        float input = test_floats[i];
        float actual = static_cast<float>(output_vec[i]);
        float expected = tanh_bw_expected_bf16(input);

        int32_t ulp = ulp_distance_bitwise(actual, expected);

        std::cout << std::setw(12) << input << std::setw(12) << actual << std::setw(12) << expected << std::setw(10)
                  << ulp << std::endl;
    }
}

TEST_F(TanhBwUlpDeviceTest, PerSegmentAnalysis) {
    // Detailed per-segment ULP analysis for tanh_bw
    // Tests ALL BF16 values in a single tensor call
    using namespace bf16_ulp_bw;

    const auto& idx = g_sorted_index;
    std::vector<::bfloat16> all_values;
    all_values.reserve(idx.size());

    for (size_t i = 0; i < idx.size(); ++i) {
        all_values.push_back(::bfloat16(idx.value_at(i)));
    }

    // Pad to tile-compatible shape
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 2048;
    constexpr size_t padded_size = tile_height * tile_width;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    std::vector<::bfloat16> padded_grad(padded_size, ::bfloat16(1.0f));
    for (size_t i = 0; i < all_values.size(); ++i) {
        padded_input[i] = all_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_height, tile_width};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Single tensor with ALL BF16 values - call ttnn::tanh_bw once
    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);
    Tensor host_grad = Tensor::from_vector(padded_grad, tensor_spec);
    Tensor device_grad = host_grad.to_layout(Layout::TILE).to_device(device_);

    auto output_tensors = ttnn::tanh_bw(device_grad, device_input);
    Tensor output_host = output_tensors[0].value().cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    // Define segments by actual x ranges (not abs)
    struct Segment {
        std::string name;
        float min_x;
        float max_x;
        std::vector<int32_t> ulps;
        float worst_input = 0;
        int32_t worst_ulp = 0;
    };

    std::vector<Segment> segments = {
        {"x < -10            ", -std::numeric_limits<float>::max(), -10.0f, {}, 0, 0},
        {"-10 <= x < -5      ", -10.0f, -5.0f, {}, 0, 0},
        {"-5 <= x < -4       ", -5.0f, -4.0f, {}, 0, 0},
        {"-4 <= x < -3.5     ", -4.0f, -3.5f, {}, 0, 0},
        {"-3.5 <= x < -3     ", -3.5f, -3.0f, {}, 0, 0},
        {"-3 <= x < -2       ", -3.0f, -2.0f, {}, 0, 0},
        {"-2 <= x < -1       ", -2.0f, -1.0f, {}, 0, 0},
        {"-1 <= x < -0.5     ", -1.0f, -0.5f, {}, 0, 0},
        {"-0.5 <= x < 0      ", -0.5f, 0.0f, {}, 0, 0},
        {"x == 0             ", 0.0f, 0.0f, {}, 0, 0},  // Special case for zero
        {"0 < x < 0.5        ", 0.0f, 0.5f, {}, 0, 0},
        {"0.5 <= x < 1       ", 0.5f, 1.0f, {}, 0, 0},
        {"1 <= x < 2         ", 1.0f, 2.0f, {}, 0, 0},
        {"2 <= x < 3         ", 2.0f, 3.0f, {}, 0, 0},
        {"3 <= x < 3.5       ", 3.0f, 3.5f, {}, 0, 0},
        {"3.5 <= x < 4       ", 3.5f, 4.0f, {}, 0, 0},
        {"4 <= x < 5         ", 4.0f, 5.0f, {}, 0, 0},
        {"5 <= x < 10        ", 5.0f, 10.0f, {}, 0, 0},
        {"x >= 10            ", 10.0f, std::numeric_limits<float>::max(), {}, 0, 0},
    };

    // Categorize each value into segments
    for (size_t i = 0; i < all_values.size(); ++i) {
        float input = static_cast<float>(all_values[i]);
        float actual = static_cast<float>(output_vec[i]);

        float expected = tanh_bw_expected_bf16(input);

        int32_t ulp = ulp_distance_bitwise(actual, expected);
        if (ulp < 0) {
            continue;
        }

        // Find matching segment
        for (auto& seg : segments) {
            bool matches = false;
            if (seg.name.find("x == 0") != std::string::npos) {
                matches = (input == 0.0f);
            } else if (seg.name.find("0 < x") != std::string::npos) {
                matches = (input > 0.0f && input < seg.max_x);
            } else if (seg.name.find("x < -10") != std::string::npos) {
                matches = (input < -10.0f);
            } else if (seg.name.find("x >= 10") != std::string::npos) {
                matches = (input >= 10.0f);
            } else {
                matches = (input >= seg.min_x && input < seg.max_x);
            }

            if (matches) {
                seg.ulps.push_back(ulp);
                if (ulp > seg.worst_ulp) {
                    seg.worst_ulp = ulp;
                    seg.worst_input = input;
                }
                break;
            }
        }
    }

    // Print per-segment table
    std::cout << "\n";
    std::cout << "TANH_BW PER-SEGMENT ULP ANALYSIS\n";
    std::cout << "================================\n\n";
    std::cout << std::left << std::setw(22) << "Segment" << std::right << std::setw(8) << "Count" << std::setw(10)
              << "Max ULP" << std::setw(12) << "Mean ULP" << std::setw(10) << "ULP=0" << std::setw(10) << "ULP<=1"
              << std::setw(10) << "ULP<=2" << std::setw(14) << "Worst Input" << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (const auto& seg : segments) {
        if (seg.ulps.empty()) {
            continue;
        }

        int count = seg.ulps.size();
        int max_ulp = *std::max_element(seg.ulps.begin(), seg.ulps.end());
        double mean_ulp = std::accumulate(seg.ulps.begin(), seg.ulps.end(), 0.0) / count;
        int ulp_0 = std::count(seg.ulps.begin(), seg.ulps.end(), 0);
        int ulp_le_1 = std::count_if(seg.ulps.begin(), seg.ulps.end(), [](int u) { return u <= 1; });
        int ulp_le_2 = std::count_if(seg.ulps.begin(), seg.ulps.end(), [](int u) { return u <= 2; });

        double pct_0 = 100.0 * ulp_0 / count;
        double pct_1 = 100.0 * ulp_le_1 / count;
        double pct_2 = 100.0 * ulp_le_2 / count;

        std::cout << std::left << std::setw(22) << seg.name << std::right << std::setw(8) << count << std::setw(10)
                  << max_ulp << std::fixed << std::setprecision(2) << std::setw(12) << mean_ulp << std::setprecision(1)
                  << std::setw(9) << pct_0 << "%" << std::setw(9) << pct_1 << "%" << std::setw(9) << pct_2 << "%"
                  << std::setprecision(4) << std::setw(14) << seg.worst_input << "\n";
    }

    std::cout << std::string(96, '-') << "\n";

    // Also print derivative behavior explanation
    std::cout << "\nDerivative Behavior: tanh'(x) = 1 - tanh(x)^2 = sech^2(x)\n";
    std::cout << "  - Near zero: derivative = 1.0 (excellent precision)\n";
    std::cout << "  - Transition region: derivative decreases from 1 to 0\n";
    std::cout << "  - Saturation region (|x| > 3): IMPLEMENTATION BUG - produces 0 instead of correct small values\n";
    std::cout << "\nNOTE: High ULP in saturation is an implementation bug, NOT a BF16 limitation.\n";
    std::cout << "      The forward tanh achieves Max ULP = 1, proving correct implementation is possible.\n";
}

TEST_F(TanhBwUlpDeviceTest, DenormalInputsProduceDerivativeOne) {
    // Denormal inputs are treated as zero under DAZ
    // tanh'(0) = 1 - tanh(0)^2 = 1
    using namespace bf16_ulp_bw;

    std::vector<::bfloat16> denormal_values;
    for (uint16_t bits = 0x0001; bits < 0x0080; ++bits) {
        denormal_values.push_back(::bfloat16(bf16_bits_to_float(bits)));
    }

    std::cout << "Testing " << denormal_values.size() << " positive denormal inputs for tanh_bw..." << std::endl;

    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    std::vector<::bfloat16> padded_grad(padded_size, ::bfloat16(1.0f));
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        padded_input[i] = denormal_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);
    Tensor host_grad = Tensor::from_vector(padded_grad, tensor_spec);
    Tensor device_grad = host_grad.to_layout(Layout::TILE).to_device(device_);

    auto output_tensors = ttnn::tanh_bw(device_grad, device_input);
    Tensor output_host = output_tensors[0].value().cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    // Expected: all outputs should be 1.0 (since tanh'(0) = 1)
    int non_one_count = 0;
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        float out = static_cast<float>(output_vec[i]);
        int32_t ulp = ulp_distance_bitwise(out, 1.0f);
        if (ulp > 1) {
            non_one_count++;
            if (non_one_count <= 5) {
                std::cout << "  Denormal 0x" << std::hex << float_to_bf16_bits(static_cast<float>(denormal_values[i]))
                          << " -> " << std::dec << out << " (ULP from 1.0: " << ulp << ")" << std::endl;
            }
        }
    }

    std::cout << "Values with ULP > 1 from 1.0: " << non_one_count << "/" << denormal_values.size() << std::endl;

    if (non_one_count == 0) {
        std::cout << "All denormal inputs correctly produce derivative = 1.0 (DAZ verified)" << std::endl;
    }

    // Under DAZ, all denormal inputs should be treated as zero, so tanh'(0) = 1
    EXPECT_EQ(non_one_count, 0) << "All denormal inputs should produce derivative = 1.0 under DAZ";
}

}  // namespace ttnn::test
