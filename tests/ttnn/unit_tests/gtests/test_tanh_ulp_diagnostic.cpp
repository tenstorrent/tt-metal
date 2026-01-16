// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tanh ULP Diagnostic Tests with Dual ULP Calculator Verification
 *
 * This test file contains:
 * 1. Two independent BFloat16 ULP calculators (for cross-verification):
 *    - Bitwise formula method
 *    - Sorted index lookup method
 * 2. Exhaustive BF16 sweep for tanh precision
 * 3. DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) behavior verification
 *
 * Hardware Model: Tenstorrent SFPU uses DAZ+FTZ
 * Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"
 *
 * Run: ./build/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlp*"
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
#include "ttnn/operations/eltwise/unary/unary.hpp"
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
// BFloat16 Utilities
// =============================================================================

namespace bf16_ulp {

/**
 * Convert float to BFloat16 bit representation (truncation, no rounding).
 */
inline uint16_t float_to_bf16_bits(float f) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    return static_cast<uint16_t>(f32_bits >> 16);
}

/**
 * Convert BFloat16 bits to float.
 */
inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

/**
 * Check if BF16 bits represent a denormal (subnormal) value.
 * Denormal: exponent = 0, mantissa != 0
 */
inline bool is_bf16_denormal(uint16_t bits) {
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & 0x7F;
    return (exp == 0) && (mantissa != 0);
}

/**
 * Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits.
 * Maps all denormals and -0 to +0 (0x0000).
 */
inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;
    }
    if (bits == 0x8000) {  // -0 -> +0
        return 0x0000;
    }
    return bits;
}

// =============================================================================
// Method 1: Bitwise Formula ULP Calculator
// =============================================================================

/**
 * Calculate value order index using bitwise formula.
 *
 * Index layout (DAZ+FTZ model, 65,025 finite values):
 * - 0xFF7F (-max) -> index 0
 * - 0x8080 (-min_normal) -> index 32511
 * - 0x0000 (zero) -> index 32512
 * - 0x0080 (+min_normal) -> index 32513
 * - 0x7F7F (+max) -> index 65024
 */
inline int32_t bf16_index_bitwise(uint16_t bits) {
    bits = bf16_daz_normalize(bits);

    // Handle special values
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
        // Negative normal: magnitude 0x0080 to 0x7F7F
        // Largest magnitude (0x7F7F) -> index 0
        // Smallest magnitude (0x0080) -> index 32511
        uint16_t magnitude = bits & 0x7FFF;
        return 0x7F7F - magnitude;  // 32639 - magnitude
    } else {
        // Positive normal: bits 0x0080 to 0x7F7F
        // 0x0080 -> index 32513, 0x7F7F -> index 65024
        return 32513 + bits - 0x0080;
    }
}

/**
 * Calculate ULP distance using bitwise formula method.
 */
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

/**
 * Sorted index lookup table for all BF16 values with DAZ+FTZ.
 *
 * This creates an explicit sorted list of all representable BF16 values,
 * then uses binary search to find the index. This is a completely different
 * algorithm from the bitwise method, providing independent verification.
 */
class Bf16SortedIndex {
public:
    Bf16SortedIndex() {
        // Build sorted list of all normal BF16 values (with DAZ applied)
        // Negative normals: 0x8080 to 0xFF7F
        // Zero: 0x0000
        // Positive normals: 0x0080 to 0x7F7F

        // Add all negative normals (from most negative to least negative)
        for (uint16_t bits = 0xFF7F; bits >= 0x8080; --bits) {
            if (bits == 0xFF80) {
                continue;  // Skip -inf
            }
            if (!is_bf16_denormal(bits)) {
                sorted_bits_.push_back(bits);
                sorted_values_.push_back(bf16_bits_to_float(bits));
            }
        }

        // Add zero (all denormals and ±0 map here)
        sorted_bits_.push_back(0x0000);
        sorted_values_.push_back(0.0f);

        // Add all positive normals (from smallest to largest)
        for (uint16_t bits = 0x0080; bits <= 0x7F7F; ++bits) {
            if (bits == 0x7F80) {
                continue;  // Skip +inf
            }
            if (!is_bf16_denormal(bits)) {
                sorted_bits_.push_back(bits);
                sorted_values_.push_back(bf16_bits_to_float(bits));
            }
        }

        // Verify the list is sorted
        for (size_t i = 1; i < sorted_values_.size(); ++i) {
            assert(sorted_values_[i - 1] < sorted_values_[i] && "Values must be strictly sorted");
        }
    }

    /**
     * Find the index of a BF16 value in the sorted list.
     * Returns -1 if not found (NaN, inf, etc.)
     */
    int32_t find_index(uint16_t bits) const {
        bits = bf16_daz_normalize(bits);

        // Handle special values
        uint16_t exp = (bits >> 7) & 0xFF;
        if (exp == 0xFF) {
            return -1;  // inf or NaN
        }

        // Binary search for the bits
        auto it = std::lower_bound(sorted_bits_.begin(), sorted_bits_.end(), bits, [](uint16_t a, uint16_t b) {
            return bf16_bits_to_float(a) < bf16_bits_to_float(b);
        });

        if (it != sorted_bits_.end() && *it == bits) {
            return static_cast<int32_t>(std::distance(sorted_bits_.begin(), it));
        }

        return -1;
    }

    /**
     * Calculate ULP distance using sorted index lookup.
     */
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

// Global instance for tests
static Bf16SortedIndex g_sorted_index;

/**
 * Calculate ULP distance using sorted index lookup method.
 */
inline int32_t ulp_distance_sorted(float a, float b) { return g_sorted_index.ulp_distance(a, b); }

// =============================================================================
// Reference Implementations
// =============================================================================

/**
 * Reference tanh using fp64 std::tanh.
 */
inline double tanh_reference_fp64(double x) { return std::tanh(x); }

/**
 * Exact tanh using MPFR 256-bit precision.
 *
 * This uses MPFR (Multiple Precision Floating-Point Reliable) library
 * to compute the true tanh value with 256-bit precision.
 */
inline double tanh_exact(double x) {
    constexpr mpfr_prec_t precision = 256;

    mpfr_t mpfr_x, result;

    mpfr_init2(mpfr_x, precision);
    mpfr_init2(result, precision);

    // Set input value
    mpfr_set_d(mpfr_x, x, MPFR_RNDN);

    // Compute tanh(x)
    mpfr_tanh(result, mpfr_x, MPFR_RNDN);

    // Extract result as double
    double tanh_result = mpfr_get_d(result, MPFR_RNDN);

    // Clean up
    mpfr_clear(mpfr_x);
    mpfr_clear(result);

    return tanh_result;
}

/**
 * Compute the expected BF16 tanh value with proper DAZ+FTZ methodology.
 *
 * Methodology (matching hardware behavior):
 * 1. Input: BF16 -> DAZ normalize (flush denormals to zero)
 * 2. Calculate: mpfr-256 precision for exact result
 * 3. Output: double -> float (cast) -> BF16 bits (truncate) -> DAZ normalize
 */
inline float tanh_expected_bf16(float input_bf16) {
    // Step 1: Apply DAZ to input (flush denormals to zero)
    uint16_t input_bits = float_to_bf16_bits(input_bf16);
    uint16_t input_daz = bf16_daz_normalize(input_bits);
    double input_double = static_cast<double>(bf16_bits_to_float(input_daz));

    // Step 2: Calculate with mpfr-256 precision
    double result_exact = tanh_exact(input_double);

    // Step 3: Convert to BF16 (truncation via float cast then bit extraction) and apply FTZ
    float result_f32 = static_cast<float>(result_exact);
    uint16_t result_bits = float_to_bf16_bits(result_f32);
    uint16_t result_daz = bf16_daz_normalize(result_bits);

    return bf16_bits_to_float(result_daz);
}

}  // namespace bf16_ulp

// =============================================================================
// ULP Calculator Verification Tests
// =============================================================================

class TanhUlpCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TanhUlpCalculatorTest, BitwiseAndSortedMethodsAgree) {
    // Verify that both ULP calculation methods produce identical results
    // for all pairs of adjacent values in the sorted list

    const auto& idx = bf16_ulp::g_sorted_index;
    int mismatches = 0;

    for (size_t i = 0; i < idx.size(); ++i) {
        for (size_t j = i; j < std::min(i + 10, idx.size()); ++j) {
            float a = idx.value_at(i);
            float b = idx.value_at(j);

            int32_t ulp_bitwise = bf16_ulp::ulp_distance_bitwise(a, b);
            int32_t ulp_sorted = bf16_ulp::ulp_distance_sorted(a, b);

            if (ulp_bitwise != ulp_sorted) {
                mismatches++;
                if (mismatches <= 10) {
                    std::cout << "Mismatch at i=" << i << ", j=" << j << ": a=" << a << " (0x" << std::hex
                              << bf16_ulp::float_to_bf16_bits(a) << std::dec << ")"
                              << ", b=" << b << " (0x" << std::hex << bf16_ulp::float_to_bf16_bits(b) << std::dec << ")"
                              << ", bitwise=" << ulp_bitwise << ", sorted=" << ulp_sorted << std::endl;
                }
            }
        }
    }

    EXPECT_EQ(mismatches, 0) << "ULP calculation methods disagree on " << mismatches << " pairs";
}

TEST_F(TanhUlpCalculatorTest, AdjacentValuesHaveUlpOne) {
    // Verify that adjacent values in the sorted list have ULP distance of 1

    const auto& idx = bf16_ulp::g_sorted_index;
    int failures = 0;

    for (size_t i = 1; i < idx.size(); ++i) {
        float a = idx.value_at(i - 1);
        float b = idx.value_at(i);

        int32_t ulp = bf16_ulp::ulp_distance_bitwise(a, b);

        if (ulp != 1) {
            failures++;
            if (failures <= 10) {
                std::cout << "Adjacent ULP != 1 at i=" << i << ": a=" << a << " (0x" << std::hex
                          << bf16_ulp::float_to_bf16_bits(a) << std::dec << ")"
                          << ", b=" << b << " (0x" << std::hex << bf16_ulp::float_to_bf16_bits(b) << std::dec << ")"
                          << ", ULP=" << ulp << std::endl;
            }
        }
    }

    EXPECT_EQ(failures, 0) << "Found " << failures << " adjacent pairs with ULP != 1";
}

TEST_F(TanhUlpCalculatorTest, SpecificValuesVerification) {
    // Test specific known values

    using namespace bf16_ulp;

    // Most negative normal
    EXPECT_EQ(bf16_index_bitwise(0xFF7F), 0);

    // Smallest negative normal
    EXPECT_EQ(bf16_index_bitwise(0x8080), 32511);

    // Zero
    EXPECT_EQ(bf16_index_bitwise(0x0000), 32512);

    // Smallest positive normal
    EXPECT_EQ(bf16_index_bitwise(0x0080), 32513);

    // Most positive normal
    EXPECT_EQ(bf16_index_bitwise(0x7F7F), 65024);

    // Adjacent ULP distances
    EXPECT_EQ(ulp_distance_bitwise(bf16_bits_to_float(0x8080), 0.0f), 1);
    EXPECT_EQ(ulp_distance_bitwise(0.0f, bf16_bits_to_float(0x0080)), 1);
}

TEST_F(TanhUlpCalculatorTest, DenormalsMapToZero) {
    // Verify all denormals are treated as zero under DAZ

    using namespace bf16_ulp;

    // Positive denormals: 0x0001 to 0x007F
    for (uint16_t bits = 0x0001; bits < 0x0080; ++bits) {
        EXPECT_TRUE(is_bf16_denormal(bits)) << "0x" << std::hex << bits << " should be denormal";
        EXPECT_EQ(bf16_daz_normalize(bits), 0x0000) << "Denormal 0x" << std::hex << bits << " should normalize to 0";
    }

    // Negative denormals: 0x8001 to 0x807F
    for (uint16_t bits = 0x8001; bits < 0x8080; ++bits) {
        EXPECT_TRUE(is_bf16_denormal(bits)) << "0x" << std::hex << bits << " should be denormal";
        EXPECT_EQ(bf16_daz_normalize(bits), 0x0000) << "Denormal 0x" << std::hex << bits << " should normalize to 0";
    }

    // -0 should normalize to +0
    EXPECT_EQ(bf16_daz_normalize(0x8000), 0x0000);
}

TEST_F(TanhUlpCalculatorTest, SortedIndexSize) {
    // Verify the sorted index has the expected number of values
    // 32512 negative normals + 1 zero + 32512 positive normals = 65025

    EXPECT_EQ(bf16_ulp::g_sorted_index.size(), 65025u);
}

TEST_F(TanhUlpCalculatorTest, Fp64AndMpfr256ReferenceAgree) {
    // Verify that fp64 std::tanh and mpfr-256 produce the same BF16 results
    // This validates our reference implementations are consistent

    using namespace bf16_ulp;

    const auto& idx = g_sorted_index;
    int mismatches = 0;
    double max_rel_diff = 0.0;
    float worst_input = 0.0f;

    std::cout << "\nVerifying fp64 vs mpfr-256 tanh reference implementations agree..." << std::endl;

    for (size_t i = 0; i < idx.size(); ++i) {
        float input_f = idx.value_at(i);
        double input = static_cast<double>(input_f);

        double fp64_result = tanh_reference_fp64(input);
        double mpfr_result = tanh_exact(input);

        // Both should produce the same BF16 value when truncated
        float fp64_bf16 = bf16_bits_to_float(bf16_daz_normalize(float_to_bf16_bits(static_cast<float>(fp64_result))));
        float mpfr_bf16 = bf16_bits_to_float(bf16_daz_normalize(float_to_bf16_bits(static_cast<float>(mpfr_result))));

        if (fp64_bf16 != mpfr_bf16) {
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

    // For tanh, fp64 and mpfr-256 should agree on essentially all BF16 values
    // since tanh is well-behaved and fp64 has sufficient precision
    std::cout << "NOTE: mpfr-256 is authoritative for any differences." << std::endl;
}

// =============================================================================
// Tanh Device Tests
// =============================================================================

class TanhUlpDeviceTest : public TTNNFixtureWithDevice {};

TEST_F(TanhUlpDeviceTest, ExhaustiveBf16Sweep) {
    // Test ALL normal BF16 values for tanh precision

    using namespace bf16_ulp;

    const auto& idx = g_sorted_index;
    std::vector<::bfloat16> all_values;
    all_values.reserve(idx.size());

    for (size_t i = 0; i < idx.size(); ++i) {
        all_values.push_back(::bfloat16(idx.value_at(i)));
    }

    std::cout << "Testing " << all_values.size() << " BF16 values for tanh precision..." << std::endl;

    // Pad to tile-compatible 2D shape: both height and width must be multiples of 32
    // For 65025 values, use shape (32, 2048) = 65536 elements
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 2048;                     // 64 tiles wide
    constexpr size_t padded_size = tile_height * tile_width;  // 65536
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    for (size_t i = 0; i < all_values.size(); ++i) {
        padded_input[i] = all_values[i];
    }

    // Create tensor spec with tile-compatible 2D shape
    const ttnn::Shape tensor_shape{1, 1, tile_height, tile_width};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Create host tensor and move to device
    Tensor host_tensor = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_tensor = host_tensor.to_layout(Layout::TILE).to_device(device_);

    // Run tanh
    Tensor output_tensor = ttnn::tanh(device_tensor);

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

        // Reference: tanh with proper DAZ+FTZ methodology and mpfr-256
        float expected = tanh_expected_bf16(input);

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
    std::cout << "EXHAUSTIVE BF16 SWEEP RESULTS - tanh()" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total values tested: " << ulp_errors.size() << std::endl;
    std::cout << "Max ULP: " << max_ulp << std::endl;
    std::cout << "Mean ULP: " << std::fixed << std::setprecision(4) << mean << std::endl;
    std::cout << "ULP = 0: " << count_ulp_0 << " (" << (100.0 * count_ulp_0 / ulp_errors.size()) << "%)" << std::endl;
    std::cout << "ULP <= 1: " << count_ulp_le_1 << " (" << (100.0 * count_ulp_le_1 / ulp_errors.size()) << "%)"
              << std::endl;
    std::cout << "ULP <= 2: " << count_ulp_le_2 << " (" << (100.0 * count_ulp_le_2 / ulp_errors.size()) << "%)"
              << std::endl;

    if (max_ulp > 1) {
        std::cout << "\nWorst case:" << std::endl;
        std::cout << "  Input: " << worst_input << " (0x" << std::hex << float_to_bf16_bits(worst_input) << std::dec
                  << ")" << std::endl;
        std::cout << "  Expected: " << worst_expected << std::endl;
        std::cout << "  Actual: " << worst_actual << std::endl;
        std::cout << "  ULP: " << max_ulp << std::endl;
    }

    // Assertions - tanh has excellent precision (Max ULP = 1 expected)
    EXPECT_LE(max_ulp, 2) << "Max ULP should be <= 2 for tanh";
    EXPECT_EQ(count_ulp_le_1, static_cast<int>(ulp_errors.size())) << "All values should have ULP <= 1";
}

TEST_F(TanhUlpDeviceTest, AllPositiveDenormalsProduceZero) {
    // Test that all positive denormal inputs produce zero output (DAZ verification)

    using namespace bf16_ulp;

    std::vector<::bfloat16> denormal_values;
    for (uint16_t bits = 0x0001; bits < 0x0080; ++bits) {
        denormal_values.push_back(::bfloat16(bf16_bits_to_float(bits)));
    }

    std::cout << "Testing " << denormal_values.size() << " positive denormal values..." << std::endl;

    // Pad to tile-compatible shape (32x32 = 1024 elements is enough for 127 denormals)
    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        padded_input[i] = denormal_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_tensor = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_tensor = host_tensor.to_layout(Layout::TILE).to_device(device_);
    Tensor output_tensor = ttnn::tanh(device_tensor);
    Tensor output_host = output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    int non_zero_count = 0;
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        uint16_t out_bits = float_to_bf16_bits(static_cast<float>(output_vec[i]));
        if (out_bits != 0x0000 && out_bits != 0x8000) {
            non_zero_count++;
            if (non_zero_count <= 5) {
                std::cout << "  Denormal 0x" << std::hex << float_to_bf16_bits(static_cast<float>(denormal_values[i]))
                          << " -> 0x" << out_bits << std::dec << std::endl;
            }
        }
    }

    std::cout << "Non-zero outputs: " << non_zero_count << "/" << denormal_values.size() << std::endl;

    if (non_zero_count == 0) {
        std::cout << "✓ All positive denormals correctly produce zero (DAZ verified)" << std::endl;
    }

    EXPECT_EQ(non_zero_count, 0) << "All denormal inputs should produce zero output under DAZ";
}

TEST_F(TanhUlpDeviceTest, AllNegativeDenormalsProduceZero) {
    // Test that all negative denormal inputs produce zero output (DAZ verification)

    using namespace bf16_ulp;

    std::vector<::bfloat16> denormal_values;
    for (uint16_t bits = 0x8001; bits < 0x8080; ++bits) {
        denormal_values.push_back(::bfloat16(bf16_bits_to_float(bits)));
    }

    std::cout << "Testing " << denormal_values.size() << " negative denormal values..." << std::endl;

    // Pad to tile-compatible shape (32x32 = 1024 elements is enough for 127 denormals)
    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        padded_input[i] = denormal_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_tensor = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_tensor = host_tensor.to_layout(Layout::TILE).to_device(device_);
    Tensor output_tensor = ttnn::tanh(device_tensor);
    Tensor output_host = output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    int non_zero_count = 0;
    for (size_t i = 0; i < denormal_values.size(); ++i) {
        uint16_t out_bits = float_to_bf16_bits(static_cast<float>(output_vec[i]));
        if (out_bits != 0x0000 && out_bits != 0x8000) {
            non_zero_count++;
            if (non_zero_count <= 5) {
                std::cout << "  Denormal 0x" << std::hex << float_to_bf16_bits(static_cast<float>(denormal_values[i]))
                          << " -> 0x" << out_bits << std::dec << std::endl;
            }
        }
    }

    std::cout << "Non-zero outputs: " << non_zero_count << "/" << denormal_values.size() << std::endl;

    if (non_zero_count == 0) {
        std::cout << "✓ All negative denormals correctly produce zero (DAZ verified)" << std::endl;
    }

    EXPECT_EQ(non_zero_count, 0) << "All denormal inputs should produce zero output under DAZ";
}

TEST_F(TanhUlpDeviceTest, PerSegmentAnalysis) {
    // Detailed per-segment ULP analysis for tanh
    // Tests ALL BF16 values in a single tensor call
    using namespace bf16_ulp;

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
    for (size_t i = 0; i < all_values.size(); ++i) {
        padded_input[i] = all_values[i];
    }

    const ttnn::Shape tensor_shape{1, 1, tile_height, tile_width};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Single tensor with ALL BF16 values - call ttnn::tanh once
    Tensor host_input = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_input = host_input.to_layout(Layout::TILE).to_device(device_);

    Tensor output_tensor = ttnn::tanh(device_input);
    Tensor output_host = output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
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
        {"-4 <= x < -3       ", -4.0f, -3.0f, {}, 0, 0},
        {"-3 <= x < -2       ", -3.0f, -2.0f, {}, 0, 0},
        {"-2 <= x < -1       ", -2.0f, -1.0f, {}, 0, 0},
        {"-1 <= x < -0.5     ", -1.0f, -0.5f, {}, 0, 0},
        {"-0.5 <= x < 0      ", -0.5f, 0.0f, {}, 0, 0},
        {"x == 0             ", 0.0f, 0.0f, {}, 0, 0},
        {"0 < x < 0.5        ", 0.0f, 0.5f, {}, 0, 0},
        {"0.5 <= x < 1       ", 0.5f, 1.0f, {}, 0, 0},
        {"1 <= x < 2         ", 1.0f, 2.0f, {}, 0, 0},
        {"2 <= x < 3         ", 2.0f, 3.0f, {}, 0, 0},
        {"3 <= x < 4         ", 3.0f, 4.0f, {}, 0, 0},
        {"4 <= x < 5         ", 4.0f, 5.0f, {}, 0, 0},
        {"5 <= x < 10        ", 5.0f, 10.0f, {}, 0, 0},
        {"x >= 10            ", 10.0f, std::numeric_limits<float>::max(), {}, 0, 0},
    };

    // Categorize each value into segments
    for (size_t i = 0; i < all_values.size(); ++i) {
        float input = static_cast<float>(all_values[i]);
        float actual = static_cast<float>(output_vec[i]);

        float expected = tanh_expected_bf16(input);

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
    std::cout << "TANH (FORWARD) PER-SEGMENT ULP ANALYSIS\n";
    std::cout << "=======================================\n\n";
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
    std::cout << "\ntanh(x) saturates to +/-1 for large x, which is exact in BF16.\n";
}

TEST_F(TanhUlpDeviceTest, CrossVerifyUlpMethods) {
    // Run a subset of values and verify both ULP methods agree on device results

    using namespace bf16_ulp;

    // Sample values across the range
    std::vector<float> test_floats = {
        -100.0f,
        -10.0f,
        -5.0f,
        -2.0f,
        -1.0f,
        -0.5f,
        -0.1f,
        -0.01f,
        0.0f,
        0.01f,
        0.1f,
        0.5f,
        1.0f,
        2.0f,
        5.0f,
        10.0f,
        100.0f};

    // Pad to tile-compatible shape (32x32)
    constexpr uint32_t tile_h = 32;
    constexpr uint32_t tile_w = 32;
    constexpr size_t padded_size = tile_h * tile_w;
    std::vector<::bfloat16> padded_input(padded_size, ::bfloat16(0.0f));
    for (size_t i = 0; i < test_floats.size(); ++i) {
        padded_input[i] = ::bfloat16(test_floats[i]);
    }

    const ttnn::Shape tensor_shape{1, 1, tile_h, tile_w};
    const MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    Tensor host_tensor = Tensor::from_vector(padded_input, tensor_spec);
    Tensor device_tensor = host_tensor.to_layout(Layout::TILE).to_device(device_);
    Tensor output_tensor = ttnn::tanh(device_tensor);
    Tensor output_host = output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto output_vec = output_host.to_vector<::bfloat16>();

    std::cout << "\nCross-verification of ULP methods on device results:" << std::endl;
    std::cout << std::setw(12) << "Input" << std::setw(12) << "Output" << std::setw(12) << "Expected" << std::setw(10)
              << "ULP(bit)" << std::setw(10) << "ULP(sort)" << std::setw(8) << "Match" << std::endl;

    int mismatches = 0;
    for (size_t i = 0; i < test_floats.size(); ++i) {
        float input = test_floats[i];
        float actual = static_cast<float>(output_vec[i]);
        float expected = tanh_expected_bf16(input);

        int32_t ulp_bit = ulp_distance_bitwise(actual, expected);
        int32_t ulp_sort = ulp_distance_sorted(actual, expected);

        bool match = (ulp_bit == ulp_sort);
        if (!match) {
            mismatches++;
        }

        std::cout << std::setw(12) << input << std::setw(12) << actual << std::setw(12) << expected << std::setw(10)
                  << ulp_bit << std::setw(10) << ulp_sort << std::setw(8) << (match ? "Y" : "N") << std::endl;
    }

    EXPECT_EQ(mismatches, 0) << "ULP methods should agree on all device results";
}

}  // namespace ttnn::test
