// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "datasets/in_memory_token_dataset.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer_trainer.hpp"
#include "tokenizers/tokenizer_base.hpp"

namespace ttml::datasets {

template <>
std::tuple<InMemoryTokenDataset, std::unique_ptr<tokenizers::TokenizerBase>>
create_in_memory_token_dataset<tokenizers::CharTokenizer>(
    const std::string &text, uint32_t seq_length, [[maybe_unused]] const std::string &json_file_path) {
    std::unique_ptr<tokenizers::TokenizerBase> tokenizer = tokenizers::CharTokenizerTrainer::train(text);

    std::vector<uint32_t> tokenized_text = tokenizer->encode(text);

    return {InMemoryTokenDataset(tokenized_text, seq_length), std::move(tokenizer)};
}

template <>
std::tuple<InMemoryTokenDataset, std::unique_ptr<tokenizers::TokenizerBase>>
create_in_memory_token_dataset<tokenizers::BPETokenizer>(
    const std::string &text, uint32_t seq_length, const std::string &json_file_path) {
    std::unique_ptr<tokenizers::TokenizerBase> tokenizer = std::make_unique<tokenizers::BPETokenizer>(json_file_path);

    const std::vector<uint32_t> tokenized_text = tokenizer->encode(text);

    return {InMemoryTokenDataset(tokenized_text, seq_length), std::move(tokenizer)};
}

template <>
std::tuple<InMemoryTokenDataset, std::unique_ptr<tokenizers::TokenizerBase>>
create_in_memory_token_dataset<tokenizers::BPETokenizer>(
    const std::vector<uint32_t> &tokens, uint32_t seq_length, const std::string &json_file_path) {
    std::unique_ptr<tokenizers::TokenizerBase> tokenizer = std::make_unique<tokenizers::BPETokenizer>(json_file_path);

    return {InMemoryTokenDataset(tokens, seq_length), std::move(tokenizer)};
}

std::vector<uint32_t> load_tokens_from_space_separated_file(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::vector<uint32_t> tokens;
    uint32_t token;
    size_t line_number = 1;

    while (file >> token) {
        tokens.push_back(token);
    }

    if (file.bad()) {
        throw std::runtime_error("I/O error while reading file: " + file_path);
    } else if (!file.eof()) {
        throw std::runtime_error("Non-integer data encountered in file: " + file_path);
    }

    file.close();
    return tokens;
}

}  // namespace ttml::datasets
