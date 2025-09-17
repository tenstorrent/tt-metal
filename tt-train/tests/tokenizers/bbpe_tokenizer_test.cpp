// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tokenizers/bbpe_tokenizer.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

using namespace ttml::tokenizers;

namespace {
std::string get_test_data_dir() {
    const char* env_var = std::getenv("TEST_DATA_DIR");
    return (env_var) ? std::string(env_var) : std::string(TEST_DATA_DIR);
}
}  // namespace

class BBPETokenizerTest : public ::testing::Test {
protected:
    // Note: This assumes you have a BBPE tokenizer JSON file for testing
    // You may need to create or obtain a proper BBPE tokenizer configuration
    BBPETokenizer tokenizer = BBPETokenizer(get_test_data_dir() + "/tokenizer.json");
};

TEST_F(BBPETokenizerTest, EncodeAndDecode) {
    const std::string prompt = "What is the capital of Canada?";
    auto ids = tokenizer.encode(prompt);
    auto decoded_prompt = tokenizer.decode(ids);
    EXPECT_EQ(decoded_prompt, prompt);
}

TEST_F(BBPETokenizerTest, EncodeSpecialCharacters) {
    // BBPE should handle special characters and unicode well
    const std::string prompt = "Hello!🌍";
    auto ids = tokenizer.encode(prompt);
    auto decoded_prompt = tokenizer.decode(ids);
    EXPECT_EQ(decoded_prompt, prompt);
}

TEST_F(BBPETokenizerTest, EncodeEmptyString) {
    const std::string prompt = "";
    auto ids = tokenizer.encode(prompt);
    auto decoded_prompt = tokenizer.decode(ids);
    EXPECT_EQ(decoded_prompt, prompt);
}

TEST_F(BBPETokenizerTest, VocabSize) {
    auto vocab_size = tokenizer.get_vocab_size();
    // BBPE tokenizers typically have a vocab size that includes byte-level tokens
    // The exact size depends on the specific tokenizer configuration
    EXPECT_GT(vocab_size, 0);
}

TEST_F(BBPETokenizerTest, EncodeByteLevelText) {
    // Test with text that contains various byte patterns
    const std::string prompt = "This text contains\nnewlines\tand\ttabs.";
    auto ids = tokenizer.encode(prompt);
    auto decoded_prompt = tokenizer.decode(ids);
    EXPECT_EQ(decoded_prompt, prompt);
}