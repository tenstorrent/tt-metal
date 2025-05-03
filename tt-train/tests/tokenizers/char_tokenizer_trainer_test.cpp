// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tokenizers/char_tokenizer_trainer.hpp"

#include <gtest/gtest.h>

using namespace ttml::tokenizers;

// Test fixture for CharTokenizerTrainer
class CharTokenizerTrainerTest : public ::testing::Test {
protected:
    // Example CharTokenizerTrainer instance
    CharTokenizerTrainer trainer;
};

// Test that the trainer creates a tokenizer with the correct vocabulary
TEST_F(CharTokenizerTrainerTest, TrainVocabulary) {
    std::string text = "hello world";
    std::unique_ptr<CharTokenizer> tokenizer_ptr = trainer.train(text);

    CharTokenizer::Vocabulary expected_vocabulary = {
        {" ", 1}, {"d", 2}, {"e", 3}, {"h", 4}, {"l", 5}, {"o", 6}, {"r", 7}, {"w", 8}};

    // Verify that the generated vocabulary matches the expected one
    const auto special_tokens_count = 3UL;
    ASSERT_EQ(tokenizer_ptr->get_vocabulary().size(), expected_vocabulary.size() + special_tokens_count);

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer_ptr->get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer_ptr->get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer handles duplicate characters correctly
TEST_F(CharTokenizerTrainerTest, TrainWithDuplicateCharacters) {
    std::string text = "aaaabbbb";
    std::unique_ptr<CharTokenizer> tokenizer_ptr = trainer.train(text);

    CharTokenizer::Vocabulary expected_vocabulary = {{"a", 1}, {"b", 2}};

    // Verify that the generated vocabulary has no duplicates
    const auto special_tokens_count = 3UL;
    ASSERT_EQ(tokenizer_ptr->get_vocabulary().size(), expected_vocabulary.size() + special_tokens_count);

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer_ptr->get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer_ptr->get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer starts indexing from the specified starting index
TEST_F(CharTokenizerTrainerTest, TrainWithNoPaddingToken) {
    std::string text = "abc";
    std::unique_ptr<CharTokenizer> tokenizer_ptr = trainer.train(text, /* add_padding_token */ false);

    CharTokenizer::Vocabulary expected_vocabulary = {{"a", 0}, {"b", 1}, {"c", 2}};

    // Verify that the generated vocabulary starts at the correct index
    const auto special_tokens_count = 2UL;
    ASSERT_EQ(tokenizer_ptr->get_vocabulary().size(), expected_vocabulary.size() + special_tokens_count);

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer_ptr->get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer_ptr->get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer handles an empty string correctly
TEST_F(CharTokenizerTrainerTest, TrainWithEmptyString) {
    std::string text;
    std::unique_ptr<CharTokenizer> tokenizer_ptr = trainer.train(text, /* add_padding_token */ false);

    // Verify that the generated vocabulary is empty
    const auto special_tokens_count = 2UL;
    ASSERT_EQ(tokenizer_ptr->get_vocabulary().size(), special_tokens_count);
}
