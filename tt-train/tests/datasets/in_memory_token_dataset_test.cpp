// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "datasets/in_memory_token_dataset.hpp"

#include <gtest/gtest.h>

using namespace ttml::datasets;

// Test fixture for InMemoryTokenDataset
class InMemoryTokenDatasetTest : public ::testing::Test {
protected:
    // Example tokens for testing
    std::vector<uint32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Sequence length
    uint32_t seq_length = 3;

    // Create an instance of InMemoryTokenDataset
    InMemoryTokenDataset dataset = InMemoryTokenDataset(tokens, seq_length);
};

// Test get_size_impl function
TEST_F(InMemoryTokenDatasetTest, GetSize) {
    // Expected number of samples
    size_t expected_size = tokens.size() - seq_length;

    ASSERT_EQ(dataset.get_size(), expected_size);
}

// Test get_item_impl function for the first sample
TEST_F(InMemoryTokenDatasetTest, GetItemFirstSample) {
    size_t index = 0;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<uint32_t> expected_input = {1, 2, 3};
    std::vector<uint32_t> expected_target = {2, 3, 4};

    ASSERT_EQ(std::vector<uint32_t>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<uint32_t>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test get_item_impl function for the second sample
TEST_F(InMemoryTokenDatasetTest, GetItemSecondSample) {
    size_t index = 1;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<uint32_t> expected_input = {2, 3, 4};
    std::vector<uint32_t> expected_target = {3, 4, 5};

    ASSERT_EQ(std::vector<uint32_t>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<uint32_t>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test get_item_impl function for the last sample
TEST_F(InMemoryTokenDatasetTest, GetItemLastSample) {
    size_t index = dataset.get_size() - 1;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<uint32_t> expected_input = {7, 8, 9};
    std::vector<uint32_t> expected_target = {8, 9, 10};

    ASSERT_EQ(std::vector<uint32_t>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<uint32_t>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test out of range error for get_item_impl function
TEST_F(InMemoryTokenDatasetTest, GetItemOutOfRange) {
    size_t index = dataset.get_size();  // Index out of range
    auto test_throw_lambda = [&]() { auto _ = dataset.get_item(index); };
    EXPECT_THROW(test_throw_lambda(), std::out_of_range);
}
