// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <tokenizers_cpp.h>

#include <fstream>
#include <iostream>
#include <string>

using tokenizers::Tokenizer;

namespace {

std::string get_test_data_dir() {
    const char* env_var = std::getenv("TEST_DATA_DIR");
    return (env_var) ? std::string(env_var) : std::string(TEST_DATA_DIR);
}

std::string load_bytes_from_file(const std::string& path) {
    std::ifstream file_stream(path, std::ios::in | std::ios::binary);
    EXPECT_TRUE(file_stream.is_open());
    std::string data;
    file_stream.seekg(0, std::ios::end);
    auto size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);
    data.resize(size);
    file_stream.read(data.data(), size);
    return data;
}

void test_tokenizer(std::unique_ptr<Tokenizer> tok, bool check_id_back = true) {
    // Check #1. Encode and Decode
    std::string prompt = "What is the  capital of Canada?";
    std::vector<int> ids = tok->Encode(prompt);
    std::string decoded_prompt = tok->Decode(ids);
    EXPECT_EQ(decoded_prompt, prompt);

    // Check #2. IdToToken and TokenToId
    std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
    for (auto id : ids_to_test) {
        auto token = tok->IdToToken(id);
        auto id_new = tok->TokenToId(token);
        if (check_id_back) {
            EXPECT_EQ(id, id_new);
        }
    }

    // Check #3. GetVocabSize
    auto vocab_size = tok->GetVocabSize();

    EXPECT_EQ(vocab_size, 50277);
}

}  // namespace

TEST(HuggingFaceTokenizer, ExampleUsage) {
    auto blob = load_bytes_from_file(get_test_data_dir() + "/tokenizer.json");
    auto tok = Tokenizer::FromBlobJSON(blob);
    test_tokenizer(std::move(tok), true);
}

TEST(HuggingFaceTokenizer, TinyLlama) {
    auto blob = load_bytes_from_file(get_test_data_dir() + "/tinyllama-tokenizer.json");
    auto tok = Tokenizer::FromBlobJSON(blob);
    std::string prompt = "What is the capital of Canada?";
    std::vector<int> ids = tok->Encode(prompt);
    std::vector<int> expected_ids = {1724, 338, 278, 7483, 310, 7400, 29973};
    fmt::println("expected_ids: {}", tok->Decode(expected_ids));
    EXPECT_EQ(ids, expected_ids);
    EXPECT_EQ(tok->Decode(ids), prompt);
    EXPECT_EQ(tok->GetVocabSize(), 32000);
}
