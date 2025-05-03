// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "datasets/dataset_subset.hpp"
#include "datasets/in_memory_dataset.hpp"
#include "datasets/utils.hpp"

using namespace ttml::datasets;

class RandomSplitTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
        targets = {0, 1, 0, 1};
        dataset = std::make_unique<InMemoryDataset<std::vector<float>, int>>(data, targets);
    }

    void TearDown() override {
        dataset = nullptr;
    }

    std::vector<std::vector<float>> data;
    std::vector<int> targets;
    std::unique_ptr<InMemoryDataset<std::vector<float>, int>> dataset;
};

TEST_F(RandomSplitTest, TestCorrectSplitting) {
    std::array<size_t, 2> split_indices = {2, 2};
    auto subsets = random_split(*dataset, split_indices);

    ASSERT_EQ(subsets.size(), 2);
    EXPECT_EQ(subsets[0].get_size(), 2);
    EXPECT_EQ(subsets[1].get_size(), 2);

    // Check that the subsets contain correct number of samples
    for (const auto& subset : subsets) {
        for (size_t i = 0; i < subset.get_size(); ++i) {
            auto sample = subset.get_item(i);
            ASSERT_TRUE(std::find(data.begin(), data.end(), sample.first) != data.end());
            ASSERT_TRUE(std::find(targets.begin(), targets.end(), sample.second) != targets.end());
        }
    }
}

TEST_F(RandomSplitTest, TestShuffling) {
    ttml::autograd::AutoContext::get_instance().set_seed(322);
    std::array<size_t, 4> batch_indices = {0, 1, 2, 3};
    auto original_data = dataset->get_batch(batch_indices);
    std::array<size_t, 2> split_indices = {2, 2};
    auto subsets = random_split(*dataset, split_indices, true);

    // We expect that at least one of the first elements in the subsets is different from the original order
    bool shuffled =
        (subsets[0].get_item(0).first != original_data[0].first ||
         subsets[1].get_item(0).first != original_data[2].first);
    EXPECT_TRUE(shuffled);
}

TEST_F(RandomSplitTest, TestSingleSubset) {
    std::array<size_t, 1> split_indices = {4};
    auto subsets = random_split(*dataset, split_indices, false);

    ASSERT_EQ(subsets.size(), 1);
    EXPECT_EQ(subsets[0].get_size(), 4);

    for (size_t i = 0; i < subsets[0].get_size(); ++i) {
        auto sample = subsets[0].get_item(i);
        EXPECT_EQ(sample.first, data[i]);
        EXPECT_EQ(sample.second, targets[i]);
    }
}

TEST_F(RandomSplitTest, TestInvalidSplitting) {
    std::array<size_t, 2> invalid_split0 = {3, 2};
    std::array<size_t, 2> invalid_split1 = {1, 2};
    EXPECT_THROW(random_split(*dataset, invalid_split0), std::invalid_argument);
    EXPECT_THROW(random_split(*dataset, invalid_split1), std::invalid_argument);
}
