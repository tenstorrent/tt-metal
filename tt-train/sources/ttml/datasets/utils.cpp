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


InMemoryTokenDataset create_token_dataset_from_yaml(const YAML::Node& yaml_data) {


    std::vector<uint32_t> tokens = yaml_data["tokens"].as<std::vector<uint32_t>>();
        uint32_t seq_length = yaml_data["sequence_length"].as<uint32_t>();

    return InMemoryTokenDataset(tokens, seq_length);
}

}  // namespace ttml::datasets
