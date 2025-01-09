// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "char_tokenizer_trainer.hpp"

#include <algorithm>
#include <set>
#include <string>

namespace ttml::tokenizers {

std::unique_ptr<CharTokenizer> CharTokenizerTrainer::train(const std::string& text, bool add_padding_token) {
    CharTokenizer::Vocabulary vocabulary;

    // using set instead of unordered_set to stabilize order
    std::set<char> unique_chars(text.begin(), text.end());

    if (add_padding_token) {
        vocabulary[PAD_TOKEN] = 0U;
    }

    for (char chr : unique_chars) {
        vocabulary[std::string(1, chr)] = static_cast<uint32_t>(vocabulary.size());
    }

    return std::make_unique<CharTokenizer>(vocabulary);
}

}  // namespace ttml::tokenizers
