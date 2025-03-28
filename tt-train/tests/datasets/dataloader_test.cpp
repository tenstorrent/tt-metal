// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "datasets/dataloader.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "datasets/in_memory_dataset.hpp"

using InMemoryDatasetFloatVecInt = ttml::datasets::InMemoryDataset<std::vector<float>, int>;
class DataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};

        targets = {1, 2, 3, 4};

        dataset = std::make_unique<InMemoryDatasetFloatVecInt>(data, targets);
    }

    void TearDown() override {
        dataset = nullptr;
    }

    std::vector<std::vector<float>> data;
    std::vector<int> targets;
    std::unique_ptr<InMemoryDatasetFloatVecInt> dataset;
};

// Test that the DataLoader correctly loads batches of data
TEST_F(DataLoaderTest, TestBatchLoading) {
    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt> dataloader(*dataset, 2, false);

    auto it = dataloader.begin();
    auto batch = *it;

    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(batch[0].first, data[0]);
    EXPECT_EQ(batch[1].first, data[1]);
    EXPECT_EQ(batch[0].second, targets[0]);
    EXPECT_EQ(batch[1].second, targets[1]);

    ++it;
    batch = *it;

    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(batch[0].first, data[2]);
    EXPECT_EQ(batch[1].first, data[3]);
    EXPECT_EQ(batch[0].second, targets[2]);
    EXPECT_EQ(batch[1].second, targets[3]);
}

// Test that the DataLoader correctly handles dataset sizes not divisible by batch size
TEST_F(DataLoaderTest, TestLastBatchHandling) {
    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt> dataloader(*dataset, 3, false);

    auto it = dataloader.begin();
    ++it;  // Move to the last batch

    auto batch = *it;

    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0].first, data[3]);
    EXPECT_EQ(batch[0].second, targets[3]);
}

// Test that shuffling works correctly
TEST_F(DataLoaderTest, TestShuffling) {
    ttml::autograd::ctx().set_seed(1337U);
    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt> dataloader(*dataset, 2, true);

    auto first_batch_before_shuffle = *dataloader.begin();
    auto it = dataloader.begin();
    auto batch_after_shuffle = *it;

    // Since shuffling is random, there's no guarantee that the batches will be different
    // so we can't do a direct comparison here. However, you can check if they differ:
    bool different = !(first_batch_before_shuffle == batch_after_shuffle);
    EXPECT_TRUE(different);  // This might not always hold, depending on the shuffle results
}

// Test that the DataLoader correctly iterates over the entire dataset
TEST_F(DataLoaderTest, TestIterationOverDataset) {
    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt> dataloader(*dataset, 2);

    size_t count = 0;
    for (const auto& batch : dataloader) {
        count += batch.size();
    }

    EXPECT_EQ(count, data.size());
}

// Test that the DataLoader works with a single-element batch
TEST_F(DataLoaderTest, TestSingleElementBatch) {
    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt> dataloader(*dataset, 1);

    auto it = dataloader.begin();
    auto batch = *it;

    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0].first, data[0]);
    EXPECT_EQ(batch[0].second, targets[0]);
}

// Test that the DataLoader correctly applies the collate_fn
TEST_F(DataLoaderTest, TestCollateFn) {
    // Custom collate function that sums all elements in the vectors and returns the sum as a new batch
    auto custom_collate_fn = [](const std::vector<std::pair<std::vector<float>, int>>& batch) {
        std::vector<std::pair<std::vector<float>, int>> collated_batch;
        for (const auto& sample : batch) {
            std::vector<float> summed_data(sample.first.size(), 0.0F);
            for (size_t i = 0; i < sample.first.size(); ++i) {
                summed_data[i] += sample.first[i];
            }
            collated_batch.emplace_back(summed_data, sample.second);
        }
        return collated_batch;
    };

    ttml::datasets::DataLoader<InMemoryDatasetFloatVecInt, decltype(custom_collate_fn)> dataloader(
        *dataset, 2, false, custom_collate_fn);

    auto it = dataloader.begin();
    auto batch = *it;

    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(batch[0].first[0], data[0][0]);  // Ensure the collate function was applied
    EXPECT_EQ(batch[0].first[1], data[0][1]);
    EXPECT_EQ(batch[0].first[2], data[0][2]);
    EXPECT_EQ(batch[0].second, targets[0]);

    EXPECT_EQ(batch[1].first[0], data[1][0]);
    EXPECT_EQ(batch[1].first[1], data[1][1]);
    EXPECT_EQ(batch[1].first[2], data[1][2]);
    EXPECT_EQ(batch[1].second, targets[1]);
}
