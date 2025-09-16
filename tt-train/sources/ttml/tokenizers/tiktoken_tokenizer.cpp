// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tiktoken_tokenizer.hpp"

#include <fmt/format.h>
#include <tokenizers_cpp.h>

#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <regex>
#include <algorithm>

namespace {

std::string load_bytes_from_file(const std::string& path) {
    std::ifstream file_stream(path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open file. Path: {}\n", path));
    }
    std::string data;
    file_stream.seekg(0, std::ios::end);
    auto size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);
    data.resize(size);
    file_stream.read(data.data(), size);
    return data;
}

using HuggingFaceTokenizer = tokenizers::Tokenizer;

}  // namespace

namespace ttml::tokenizers {

class TikTokenTokenizer::TikTokenTokenizerImpl {
public:
    explicit TikTokenTokenizerImpl(const std::string& json_file) {
        auto blob = load_bytes_from_file(json_file);
        m_tokenizer = HuggingFaceTokenizer::FromBlobJSON(blob);
    }
    ~TikTokenTokenizerImpl() = default;
    TikTokenTokenizerImpl(const TikTokenTokenizerImpl&) = delete;
    TikTokenTokenizerImpl& operator=(const TikTokenTokenizerImpl&) = delete;
    TikTokenTokenizerImpl(TikTokenTokenizerImpl&&) = default;
    TikTokenTokenizerImpl& operator=(TikTokenTokenizerImpl&&) = default;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<int32_t> results = m_tokenizer->Encode(text);
        // we currently use uint32_t for tokens, might change in the future
        return {results.begin(), results.end()};
    }

    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const {
        const std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());
        return m_tokenizer->Decode(tokens_i32);
    }

    [[nodiscard]] uint32_t get_vocab_size() const {
        return m_tokenizer->GetVocabSize();
    }

private:
    std::unique_ptr<HuggingFaceTokenizer> m_tokenizer;
};

TikTokenTokenizer::TikTokenTokenizer(const std::string& json_file)
    : m_pimpl(std::make_unique<TikTokenTokenizerImpl>(json_file)) {}

TikTokenTokenizer::~TikTokenTokenizer() = default;

TikTokenTokenizer::TikTokenTokenizer(TikTokenTokenizer&&) noexcept = default;
TikTokenTokenizer& TikTokenTokenizer::operator=(TikTokenTokenizer&&) noexcept = default;

std::vector<uint32_t> TikTokenTokenizer::encode(const std::string& text) const {
    return m_pimpl->encode(text);
}

std::string TikTokenTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    return m_pimpl->decode(tokens);
}

uint32_t TikTokenTokenizer::get_vocab_size() const {
    return m_pimpl->get_vocab_size();
}

}  // namespace ttml::tokenizers