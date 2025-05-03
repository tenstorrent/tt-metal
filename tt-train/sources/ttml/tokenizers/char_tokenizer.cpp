// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "char_tokenizer.hpp"

#include <sstream>
#include <stdexcept>

namespace ttml::tokenizers {
CharTokenizer::CharTokenizer(Vocabulary vocabulary) : m_vocabulary(std::move(vocabulary)) {
    auto vocab_size = static_cast<uint32_t>(m_vocabulary.size());
    m_vocabulary[BEGIN_TOKEN] = vocab_size++;
    m_vocabulary[END_TOKEN] = vocab_size++;
    build_reverse_mapping();
}

std::vector<uint32_t> CharTokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> tokens;
    for (char chr : text) {
        auto chr_str = std::string(1, chr);
        auto it = m_vocabulary.find(chr_str);
        if (it != m_vocabulary.end()) {
            tokens.push_back(it->second);
        } else {
            throw std::runtime_error("Character not in vocabulary: " + chr_str);
        }
    }
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::ostringstream oss;
    for (uint32_t token : tokens) {
        auto it = m_id_to_char.find(token);
        if (it != m_id_to_char.end()) {
            oss << it->second;
        } else {
            throw std::runtime_error("Token ID not in reverse vocabulary: " + std::to_string(token));
        }
    }
    return oss.str();
}
const CharTokenizer::Vocabulary& CharTokenizer::get_vocabulary() const {
    return m_vocabulary;
}

void CharTokenizer::build_reverse_mapping() {
    for (const auto& [token, id] : m_vocabulary) {
        m_id_to_char[id] = token;
    }
}

uint32_t CharTokenizer::get_vocab_size() const {
    return static_cast<uint32_t>(m_vocabulary.size());
}

}  // namespace ttml::tokenizers
