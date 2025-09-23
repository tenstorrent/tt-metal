// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bpe_tokenizer.hpp"

#include <fmt/format.h>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <fstream>
#include <string>

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

class BPETokenizer::BPETokenizerImpl {
public:
    explicit BPETokenizerImpl(const std::string& json_file) {
        auto blob = load_bytes_from_file(json_file);
        m_tokenizer = HuggingFaceTokenizer::FromBlobJSON(blob);
    }
    ~BPETokenizerImpl() = default;
    BPETokenizerImpl(const BPETokenizerImpl&) = delete;
    BPETokenizerImpl& operator=(const BPETokenizerImpl&) = delete;
    BPETokenizerImpl(BPETokenizerImpl&&) = default;
    BPETokenizerImpl& operator=(BPETokenizerImpl&&) = default;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<int32_t> results = m_tokenizer->Encode(text);
        // we currently use uint32_t for tokens, might change in the future
        return {results.begin(), results.end()};
    }

    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const {
        try {
            const std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());
            std::string result = m_tokenizer->Decode(tokens_i32);
            
            // Only remove null bytes, keep the original text as much as possible
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            
            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format("Error decoding tokens: {}", e.what()));
        }
    }

    [[nodiscard]] uint32_t get_vocab_size() const {
        return m_tokenizer->GetVocabSize();
    }

private:
    std::unique_ptr<HuggingFaceTokenizer> m_tokenizer;
};

BPETokenizer::BPETokenizer(const std::string& json_file) {
    m_pimpl = std::make_unique<BPETokenizerImpl>(json_file);
}

BPETokenizer::~BPETokenizer() = default;
BPETokenizer::BPETokenizer(BPETokenizer&&) noexcept = default;
BPETokenizer& BPETokenizer::operator=(BPETokenizer&&) noexcept = default;

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    return m_pimpl->encode(text);
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    return m_pimpl->decode(tokens);
}

uint32_t BPETokenizer::get_vocab_size() const {
    return m_pimpl->get_vocab_size();
}

}  // namespace ttml::tokenizers
