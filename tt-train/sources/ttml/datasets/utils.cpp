// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "datasets/in_memory_token_dataset.hpp"
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


std::tuple<std::vector<uint32_t>, uint32_t> load_tokens_from_space_separated_file(const std::string &file_path) {

    auto yaml_data = YAML::LoadFile(file_path);

    std::vector<uint32_t> tokens;
    std::stringstream token_stream(yaml_data["tokens"].as<std::string>());
    uint32_t token;

    while (token_stream >> token) {
        tokens.push_back(token);
    }

    return std::make_tuple(tokens, yaml_data["tokenizer_vocab_size"].as<uint32_t>());
}

}  // namespace ttml::datasets
