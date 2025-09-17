// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bbpe_tokenizer.hpp"

#include <fmt/format.h>
#include <tokenizers_cpp.h>

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

class BBPETokenizer::BBPETokenizerImpl {
public:
    explicit BBPETokenizerImpl(const std::string& json_file) {
        auto blob = load_bytes_from_file(json_file);
        m_tokenizer = HuggingFaceTokenizer::FromBlobJSON(blob);
        
        // BBPE tokenizers typically use byte-level pre-processing
        // The tokenizer configuration should already include byte-level settings
        // from the JSON file, but we can verify it's properly configured
    }
    ~BBPETokenizerImpl() = default;
    BBPETokenizerImpl(const BBPETokenizerImpl&) = delete;
    BBPETokenizerImpl& operator=(const BBPETokenizerImpl&) = delete;
    BBPETokenizerImpl(BBPETokenizerImpl&&) = default;
    BBPETokenizerImpl& operator=(BBPETokenizerImpl&&) = default;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const {
        // BBPE encoding: convert text to bytes, then apply BPE
        std::vector<int32_t> results = m_tokenizer->Encode(text);
        // Convert to uint32_t for consistency with the interface
        return {results.begin(), results.end()};
    }

    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const {
        // BBPE decoding: convert tokens back to bytes, then to text
        const std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());
        return m_tokenizer->Decode(tokens_i32);
    }

    [[nodiscard]] uint32_t get_vocab_size() const {
        return m_tokenizer->GetVocabSize();
    }

private:
    std::unique_ptr<HuggingFaceTokenizer> m_tokenizer;
};

BBPETokenizer::BBPETokenizer(const std::string& json_file) {
    m_pimpl = std::make_unique<BBPETokenizerImpl>(json_file);
}

BBPETokenizer::~BBPETokenizer() = default;
BBPETokenizer::BBPETokenizer(BBPETokenizer&&) noexcept = default;
BBPETokenizer& BBPETokenizer::operator=(BBPETokenizer&&) noexcept = default;

std::vector<uint32_t> BBPETokenizer::encode(const std::string& text) const {
    return m_pimpl->encode(text);
}

std::string BBPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    return m_pimpl->decode(tokens);
}

uint32_t BBPETokenizer::get_vocab_size() const {
    return m_pimpl->get_vocab_size();
}

}  // namespace ttml::tokenizers