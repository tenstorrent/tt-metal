// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Debug test to isolate untilize issue on Blackhole
// Run with: ./tt-train/build/tests/ttml_tests --gtest_filter="DebugUntilizeTest.*"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class DebugUntilizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Test 1: argmax with aligned dimensions (should pass)
TEST_F(DebugUntilizeTest, ArgmaxAligned64) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 32, 64} - aligned
    xt::xarray<float>::shape_type shape = {1, 1, 32, 64};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set max at position 10 for each row
    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, 10) = 1000.0f;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto untilized = ttnn::untilize(tensor_a);
    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "ArgmaxAligned64 results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_EQ(v, 10);
    }
}

// Test 2: argmax with unaligned dimensions (65) - may fail on Blackhole
TEST_F(DebugUntilizeTest, ArgmaxUnaligned65) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 32, 65} - unaligned
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set max at position 10 for each row
    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, 10) = 1000.0f;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto untilized = ttnn::untilize(tensor_a);
    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "ArgmaxUnaligned65 results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_EQ(v, 10);
    }
}

// Test 3: argmax with unaligned dimensions (65) without untilize
TEST_F(DebugUntilizeTest, ArgmaxUnaligned65NoUntilize) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 32, 65} - unaligned, row major directly
    std::vector<float> data(32 * 65, 0.0f);

    // Set max at position 10 for each row
    for (size_t i = 0; i < 32; ++i) {
        data[i * 65 + 10] = 1000.0f;
    }

    auto shape = ttnn::Shape({1, 1, 32, 65});
    auto tensor_a = ttml::core::from_vector(data, shape, device, ttnn::Layout::ROW_MAJOR);
    auto result = ttnn::argmax(tensor_a, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "ArgmaxUnaligned65NoUntilize results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_EQ(v, 10);
    }
}

// Test 4: Check if subtract with mask causes issues
TEST_F(DebugUntilizeTest, SubtractMaskThenArgmax65) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 32, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set values: position 10 = 100, position 64 = 200 (would be max without mask)
    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, 10) = 100.0f;
        a(0, 0, i, 64) = 200.0f;  // This should be masked out
    }

    // Create mask: large value at position 64
    xt::xarray<float> mask = xt::zeros<float>(shape);
    for (size_t i = 0; i < 32; ++i) {
        mask(0, 0, i, 64) = 1e4f;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);

    // Subtract mask (this should make position 64 very negative)
    auto masked = ttnn::subtract(tensor_a, tensor_mask);

    // Check intermediate values
    auto masked_vec = ttml::core::to_vector(masked);
    std::cout << "SubtractMaskThenArgmax65 - masked values at row 0: ";
    std::cout << "pos[10]=" << masked_vec[10] << " pos[64]=" << masked_vec[64] << std::endl;

    auto untilized = ttnn::untilize(masked);
    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "SubtractMaskThenArgmax65 results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_LT(v, 64);  // Should not be 64 (masked)
        EXPECT_EQ(v, 10);  // Should be 10
    }
}

// Test 5: Random tensor with mask (closest to original test)
TEST_F(DebugUntilizeTest, RandomWithMask65) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 32, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = xt::random::rand<float>(shape);

    // Create mask: large value at position 64
    xt::xarray<float> mask = xt::zeros<float>(shape);
    for (size_t i = 0; i < 32; ++i) {
        mask(0, 0, i, 64) = 1e4f;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);

    auto masked = ttnn::subtract(tensor_a, tensor_mask);
    auto untilized = ttnn::untilize(masked);
    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "RandomWithMask65 results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_LT(v, 64);  // Should be in range [0, 63] since 64 is masked
    }
}

// Test 6: Test with 33 width (another unaligned size)
TEST_F(DebugUntilizeTest, ArgmaxUnaligned33) {
    auto* device = &ttml::autograd::ctx().get_device();

    xt::xarray<float>::shape_type shape = {1, 1, 32, 33};
    xt::xarray<float> a = xt::zeros<float>(shape);

    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, 10) = 1000.0f;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto untilized = ttnn::untilize(tensor_a);
    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "ArgmaxUnaligned33 results: ";
    for (size_t i = 0; i < std::min(vector_result.size(), size_t(10)); ++i) {
        std::cout << vector_result[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(vector_result.size(), 32);
    for (auto v : vector_result) {
        EXPECT_EQ(v, 10);
    }
}

// Test 7: FOCUSED TEST - Untilize alone with unaligned dimensions
TEST_F(DebugUntilizeTest, UntilizeOnly65) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 4, 65} - unaligned
    xt::xarray<float>::shape_type shape = {1, 1, 4, 65};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set known values
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 65; ++col) {
            a(0, 0, row, col) = static_cast<float>(row * 100 + col);
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    std::cout << "UntilizeOnly65 - Input layout: " << (tensor_a.layout() == ttnn::Layout::TILE ? "TILE" : "ROW_MAJOR")
              << std::endl;

    auto untilized = ttnn::untilize(tensor_a);
    auto vec = ttml::core::to_vector(untilized);

    std::cout << "UntilizeOnly65 - Output layout: " << (untilized.layout() == ttnn::Layout::TILE ? "TILE" : "ROW_MAJOR")
              << std::endl;

    // Check each row's data
    bool all_correct = true;
    for (size_t row = 0; row < 4; ++row) {
        std::cout << "Row " << row << " first 10 values: ";
        for (size_t col = 0; col < 10; ++col) {
            float expected = static_cast<float>(row * 100 + col);
            float actual = vec[row * 65 + col];
            std::cout << actual << " ";
            if (std::fabs(actual - expected) > 0.1f) {
                all_correct = false;
            }
        }
        std::cout << "... last 5 values: ";
        for (size_t col = 60; col < 65; ++col) {
            float expected = static_cast<float>(row * 100 + col);
            float actual = vec[row * 65 + col];
            std::cout << actual << " ";
            if (std::fabs(actual - expected) > 0.1f) {
                all_correct = false;
            }
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(all_correct) << "Untilize corrupted data for unaligned tensor dimensions";
}

// Test 8: Untilize with aligned dimensions (should work)
TEST_F(DebugUntilizeTest, UntilizeOnly64) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 4, 64} - aligned
    xt::xarray<float>::shape_type shape = {1, 1, 4, 64};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set known values
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 64; ++col) {
            a(0, 0, row, col) = static_cast<float>(row * 100 + col);
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto untilized = ttnn::untilize(tensor_a);
    auto vec = ttml::core::to_vector(untilized);

    // Check each row's data
    bool all_correct = true;
    for (size_t row = 0; row < 4; ++row) {
        std::cout << "UntilizeOnly64 Row " << row << " first 10: ";
        for (size_t col = 0; col < 10; ++col) {
            float expected = static_cast<float>(row * 100 + col);
            float actual = vec[row * 64 + col];
            std::cout << actual << " ";
            if (std::fabs(actual - expected) > 0.1f) {
                all_correct = false;
            }
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(all_correct) << "Untilize should work for aligned tensor dimensions";
}

// Test 9: Untilize with 33 width
TEST_F(DebugUntilizeTest, UntilizeOnly33) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 4, 33} - unaligned
    xt::xarray<float>::shape_type shape = {1, 1, 4, 33};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set known values
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 33; ++col) {
            a(0, 0, row, col) = static_cast<float>(row * 100 + col);
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto untilized = ttnn::untilize(tensor_a);
    auto vec = ttml::core::to_vector(untilized);

    // Check each row's data
    bool all_correct = true;
    for (size_t row = 0; row < 4; ++row) {
        std::cout << "UntilizeOnly33 Row " << row << ": ";
        for (size_t col = 0; col < 33; ++col) {
            float expected = static_cast<float>(row * 100 + col);
            float actual = vec[row * 33 + col];
            if (col < 5 || col >= 30) {
                std::cout << actual << " ";
            }
            if (std::fabs(actual - expected) > 0.1f) {
                all_correct = false;
            }
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(all_correct) << "Untilize corrupted data for width=33";
}

// Test 10: Inspect raw tensor data after untilize
TEST_F(DebugUntilizeTest, InspectUntilizeOutput65) {
    auto* device = &ttml::autograd::ctx().get_device();

    xt::xarray<float>::shape_type shape = {1, 1, 4, 65};  // Smaller for inspection
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set specific pattern
    for (size_t i = 0; i < 4; ++i) {
        a(0, 0, i, i * 10) = 1000.0f + i;
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);

    std::cout << "Before untilize - tensor shape: " << tensor_a.logical_shape() << std::endl;
    std::cout << "Before untilize - layout: " << (tensor_a.layout() == ttnn::Layout::TILE ? "TILE" : "ROW_MAJOR")
              << std::endl;

    auto untilized = ttnn::untilize(tensor_a);

    std::cout << "After untilize - tensor shape: " << untilized.logical_shape() << std::endl;
    std::cout << "After untilize - layout: " << (untilized.layout() == ttnn::Layout::TILE ? "TILE" : "ROW_MAJOR")
              << std::endl;

    auto vec = ttml::core::to_vector(untilized);
    std::cout << "Untilized data (first 70 elements): ";
    for (size_t i = 0; i < std::min(vec.size(), size_t(70)); ++i) {
        if (vec[i] != 0.0f) {
            std::cout << "[" << i << "]=" << vec[i] << " ";
        }
    }
    std::cout << std::endl;

    auto result = ttnn::argmax(untilized, 3, true, std::nullopt, true);
    auto vector_result = ttml::core::to_vector<uint32_t>(result);

    std::cout << "Argmax results: ";
    for (auto v : vector_result) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // Expected: 0, 10, 20, 30
    EXPECT_EQ(vector_result.size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vector_result[i], i * 10);
    }
}
