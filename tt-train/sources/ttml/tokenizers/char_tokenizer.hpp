// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "tokenizer_base.hpp"

namespace ttml::tokenizers {

constexpr auto PAD_TOKEN = "<PAD>";
constexpr auto END_TOKEN = "<END>";
constexpr auto BEGIN_TOKEN = "<BEG>";

class CharTokenizer : public TokenizerBase {
public:
    using Vocabulary = std::unordered_map<std::string, uint32_t>;
    using IdtoChars = std::unordered_map<uint32_t, std::string>;
    // Constructor that initializes the tokenizer with a vocabulary
    explicit CharTokenizer(Vocabulary vocabulary);

    CharTokenizer(const CharTokenizer&) = default;
    CharTokenizer& operator=(const CharTokenizer&) = default;

    CharTokenizer(CharTokenizer&&) = default;
    CharTokenizer& operator=(CharTokenizer&&) = default;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const override;

    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const override;

    [[nodiscard]] const CharTokenizer::Vocabulary& get_vocabulary() const;

    [[nodiscard]] uint32_t get_vocab_size() const override;

    ~CharTokenizer() override = default;

private:
    Vocabulary m_vocabulary;
    IdtoChars m_id_to_char;

    void build_reverse_mapping();
};

}  // namespace ttml::tokenizers
