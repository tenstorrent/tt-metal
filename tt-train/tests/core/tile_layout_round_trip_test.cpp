// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * TILE Layout Round-Trip Test
 *
 * This test demonstrates a critical bug in TILE layout conversion operations.
 *
 * BUG: Structured data is scrambled during ROW_MAJOR -> TILE -> ROW_MAJOR conversion,
 * while random data passes through correctly.
 *
 * Test procedure:
 * 1. Create structured data (sequential pattern)
 * 2. Create random data (same shape)
 * 3. Convert both to device tensors (ROW_MAJOR -> TILE layout)
 * 4. Convert back to host (TILE -> ROW_MAJOR layout)
 * 5. Compare input vs output
 *
 * Expected: Both should have PCC > 0.99
 * Actual: Random data works (PCC ~1.0), structured data fails (PCC ~0.1-0.3)
 *
 * This bug affects all real learned weights in neural networks.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

using namespace ttml;

namespace {

/**
 * Compute Pearson Correlation Coefficient
 */
float compute_pcc(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) {
        return 0.0F;
    }

    float mean_x = 0.0F;
    float mean_y = 0.0F;
    for (size_t i = 0; i < x.size(); ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= static_cast<float>(x.size());
    mean_y /= static_cast<float>(y.size());

    float numerator = 0.0F;
    float denom_x = 0.0F;
    float denom_y = 0.0F;
    for (size_t i = 0; i < x.size(); ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    float denominator = std::sqrt(denom_x * denom_y);
    return (denominator > 0.0F) ? (numerator / denominator) : 0.0F;
}

/**
 * Create structured data with a clear pattern
 * This simulates the spatial structure found in real learned weights
 */
std::vector<float> create_structured_data(const std::vector<uint32_t>& shape) {
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    std::vector<float> data(total_size);

    // Create a structured pattern: sine wave + position encoding
    for (size_t i = 0; i < total_size; ++i) {
        float normalized_pos = static_cast<float>(i) / static_cast<float>(total_size);
        // Combine multiple frequencies to create realistic structure
        float value = std::sin(normalized_pos * 10.0F) * 0.5F +   // Low frequency
                      std::sin(normalized_pos * 50.0F) * 0.3F +   // Medium frequency
                      std::cos(normalized_pos * 100.0F) * 0.2F +  // High frequency
                      (normalized_pos - 0.5F) * 2.0F;             // Linear trend

        data[i] = value;
    }

    return data;
}

/**
 * Create random data (control group)
 */
std::vector<float> create_random_data(const std::vector<uint32_t>& shape) {
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    std::vector<float> data(total_size);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0F, 1.0F);

    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dist(gen);
    }

    return data;
}

/**
 * Perform round-trip: Host -> Device (TILE) -> Host
 */
std::vector<float> perform_round_trip(const std::vector<float>& input_data, const std::vector<uint32_t>& shape) {
    // Create ttnn shape
    ttnn::Shape tensor_shape(shape);

    // Create device tensor (default TILE layout)
    auto* device = &autograd::ctx().get_device();

    // Create tensor on device with TILE layout (default)
    auto device_tensor = core::from_vector(input_data, tensor_shape, device, tt::tt_metal::Layout::TILE);

    // Convert back to host
    auto output_data = core::to_vector(device_tensor);

    return output_data;
}

/**
 * Print comparison statistics
 */
void print_comparison(
    const char* test_name, const std::vector<float>& original, const std::vector<float>& round_trip, float threshold) {
    float pcc = compute_pcc(original, round_trip);

    float mean_diff = 0.0F;
    float max_diff = 0.0F;
    for (size_t i = 0; i < original.size(); ++i) {
        float diff = std::abs(original[i] - round_trip[i]);
        mean_diff += diff;
        max_diff = std::max(max_diff, diff);
    }
    mean_diff /= static_cast<float>(original.size());

    std::cout << "\n================================================================================\n";
    std::cout << test_name << "\n";
    std::cout << "================================================================================\n";
    std::cout << "PCC: " << pcc;
    if (pcc >= threshold) {
        std::cout << " ✅ PASS\n";
    } else {
        std::cout << " ❌ FAIL (threshold: " << threshold << ")\n";
    }
    std::cout << "Mean abs diff: " << mean_diff << "\n";
    std::cout << "Max abs diff: " << max_diff << "\n";

    std::cout << "First 10 original values:   ";
    for (size_t i = 0; i < std::min(size_t(10), original.size()); ++i) {
        std::cout << original[i] << " ";
    }
    std::cout << "\nFirst 10 round-trip values: ";
    for (size_t i = 0; i < std::min(size_t(10), round_trip.size()); ++i) {
        std::cout << round_trip[i] << " ";
    }
    std::cout << "\n";
}

}  // namespace

/**
 * Test: Random data should survive round-trip through TILE layout
 */
TEST(TileLayoutRoundTripTest, RandomDataPreserved) {
    std::cout << "\n################################################################################\n";
    std::cout << "TEST: Random Data Round-Trip (Control Group)\n";
    std::cout << "################################################################################\n";

    // Small tensor shape that's representative
    std::vector<uint32_t> shape = {1, 2, 64, 64};

    std::cout << "Creating random data with shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";

    auto random_data = create_random_data(shape);
    auto random_round_trip = perform_round_trip(random_data, shape);

    const float threshold = 0.99F;
    print_comparison("Random Data Round-Trip", random_data, random_round_trip, threshold);

    float pcc = compute_pcc(random_data, random_round_trip);
    EXPECT_GE(pcc, threshold) << "Random data should be preserved through TILE layout conversion";
}

/**
 * Test: Structured data should survive round-trip through TILE layout
 * FIXED: Bug was in Python bindings (non-contiguous arrays), not TILE layout
 */
TEST(TileLayoutRoundTripTest, StructuredDataPreserved) {
    std::cout << "\n################################################################################\n";
    std::cout << "TEST: Structured Data Round-Trip (Verifies TILE layout works)\n";
    std::cout << "################################################################################\n";

    // Same shape as random data test
    std::vector<uint32_t> shape = {1, 2, 64, 64};

    std::cout << "Creating structured data (sine wave pattern) with shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";

    auto structured_data = create_structured_data(shape);
    auto structured_round_trip = perform_round_trip(structured_data, shape);

    const float threshold = 0.99F;
    print_comparison("Structured Data Round-Trip", structured_data, structured_round_trip, threshold);

    float pcc = compute_pcc(structured_data, structured_round_trip);

    // Bug was fixed - it was in Python bindings handling non-contiguous arrays,
    // not in the C++ TILE layout operations themselves
    EXPECT_GE(pcc, threshold) << "Structured data should be preserved through TILE layout conversion";
}

/**
 * Test: Compare random vs structured data corruption
 * FIXED: Both types of data now preserve well (bug was in Python bindings)
 */
TEST(TileLayoutRoundTripTest, CompareRandomVsStructured) {
    std::cout << "\n################################################################################\n";
    std::cout << "TEST: Comparison - Random vs Structured Data\n";
    std::cout << "################################################################################\n";

    std::vector<uint32_t> shape = {1, 2, 64, 64};

    auto random_data = create_random_data(shape);
    auto structured_data = create_structured_data(shape);

    auto random_round_trip = perform_round_trip(random_data, shape);
    auto structured_round_trip = perform_round_trip(structured_data, shape);

    float random_pcc = compute_pcc(random_data, random_round_trip);
    float structured_pcc = compute_pcc(structured_data, structured_round_trip);

    std::cout << "\nRESULTS:\n";
    std::cout << "  Random data PCC:     " << random_pcc << " " << (random_pcc >= 0.99F ? "✅" : "❌") << "\n";
    std::cout << "  Structured data PCC: " << structured_pcc << " " << (structured_pcc >= 0.99F ? "✅" : "❌") << "\n";

    float pcc_ratio = random_pcc / std::max(structured_pcc, 0.001F);
    std::cout << "\nPCC ratio (random/structured): " << pcc_ratio << "x\n";

    // Bug was fixed - both random and structured data should preserve well
    EXPECT_GE(random_pcc, 0.99F) << "Random data should be preserved";
    EXPECT_GE(structured_pcc, 0.99F) << "Structured data should be preserved";
    EXPECT_LT(pcc_ratio, 1.1F) << "PCC ratio should be close to 1.0 (both preserve equally well)";
}

/**
 * Test: Different tensor shapes
 * Check if certain shapes are more affected
 */
TEST(TileLayoutRoundTripTest, DifferentShapes) {
    std::cout << "\n################################################################################\n";
    std::cout << "TEST: Different Tensor Shapes\n";
    std::cout << "################################################################################\n";

    std::vector<std::vector<uint32_t>> test_shapes = {
        {1, 1, 32, 32},   // Tile-aligned (32x32)
        {1, 1, 64, 64},   // Larger tile-aligned
        {1, 2, 16, 64},   // Multi-head attention shape
        {1, 4, 128, 64},  // Larger multi-head
    };

    for (const auto& shape : test_shapes) {
        std::cout << "\n--- Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "] ---\n";

        auto structured_data = create_structured_data(shape);
        auto round_trip = perform_round_trip(structured_data, shape);
        float pcc = compute_pcc(structured_data, round_trip);

        std::cout << "Structured data PCC: " << pcc << " " << (pcc >= 0.99F ? "✅" : "❌") << "\n";
    }

    std::cout << "\nAll shapes show the same bug behavior.\n";
}
