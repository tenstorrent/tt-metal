// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tokenizers/char_tokenizer.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace ttml::tokenizers;

// Test fixture for CharTokenizer
class CharTokenizerTest : public ::testing::Test {
protected:
    CharTokenizer::Vocabulary vocabulary = {
        {"h", 1}, {"e", 2}, {"l", 3}, {"o", 4}, {" ", 5}, {"w", 6}, {"r", 7}, {"d", 8}};

    CharTokenizer tokenizer = CharTokenizer(vocabulary);
};

// Test encoding functionality
TEST_F(CharTokenizerTest, Encode) {
    std::string text = "hello world";
    std::vector<uint32_t> expected_tokens = {1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8};

    std::vector<uint32_t> encoded = tokenizer.encode(text);

    ASSERT_EQ(encoded, expected_tokens);
}

// Test encoding with a character not in vocabulary
TEST_F(CharTokenizerTest, EncodeUnknownCharacter) {
    std::string text = "hello world!";
    EXPECT_THROW({ auto _ = tokenizer.encode(text); }, std::runtime_error);
}

// Test decoding functionality
TEST_F(CharTokenizerTest, Decode) {
    std::vector<uint32_t> tokens = {1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8};
    std::string expected_text = "hello world";

    std::string decoded = tokenizer.decode(tokens);

    ASSERT_EQ(decoded, expected_text);
}

// Test decoding with a token ID not in vocabulary
TEST_F(CharTokenizerTest, DecodeUnknownToken) {
    std::vector<uint32_t> tokens = {1, 2, 3, 3, 4, 33};  // Token 33 is not in the vocabulary

    EXPECT_THROW({ auto _ = tokenizer.decode(tokens); }, std::runtime_error);
}

// Test encoding and decoding consistency
TEST_F(CharTokenizerTest, EncodeDecodeConsistency) {
    std::string text = "hello world";
    std::vector<uint32_t> encoded = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(encoded);

    ASSERT_EQ(decoded, text);
}
