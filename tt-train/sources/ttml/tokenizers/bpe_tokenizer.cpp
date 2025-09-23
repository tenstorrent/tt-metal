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
            
            // Remove null bytes from the result
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            
            // Clean up potential encoding artifacts
            // Remove leading/trailing whitespace that might cause display issues
            auto start = result.find_first_not_of(" \t\n\r\f\v");
            if (start == std::string::npos) {
                return ""; // String is all whitespace
            }
            auto end = result.find_last_not_of(" \t\n\r\f\v");
            result = result.substr(start, end - start + 1);
            
            // Validate UTF-8 and replace invalid sequences
            std::string cleaned_result;
            cleaned_result.reserve(result.size());
            
            for (size_t i = 0; i < result.size(); ) {
                unsigned char c = static_cast<unsigned char>(result[i]);
                
                // ASCII characters (0-127)
                if (c < 0x80) {
                    // Skip control characters except newline, tab, and carriage return
                    if (c >= 0x20 || c == '\n' || c == '\t' || c == '\r') {
                        cleaned_result += c;
                    }
                    i++;
                }
                // Multi-byte UTF-8 sequences
                else if (c < 0xC0) {
                    // Invalid start byte, skip
                    i++;
                }
                else if (c < 0xE0) {
                    // 2-byte sequence
                    if (i + 1 < result.size() && 
                        (static_cast<unsigned char>(result[i + 1]) & 0xC0) == 0x80) {
                        cleaned_result += result.substr(i, 2);
                        i += 2;
                    } else {
                        i++; // Invalid sequence, skip
                    }
                }
                else if (c < 0xF0) {
                    // 3-byte sequence
                    if (i + 2 < result.size() && 
                        (static_cast<unsigned char>(result[i + 1]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(result[i + 2]) & 0xC0) == 0x80) {
                        cleaned_result += result.substr(i, 3);
                        i += 3;
                    } else {
                        i++; // Invalid sequence, skip
                    }
                }
                else if (c < 0xF8) {
                    // 4-byte sequence
                    if (i + 3 < result.size() && 
                        (static_cast<unsigned char>(result[i + 1]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(result[i + 2]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(result[i + 3]) & 0xC0) == 0x80) {
                        cleaned_result += result.substr(i, 4);
                        i += 4;
                    } else {
                        i++; // Invalid sequence, skip
                    }
                }
                else {
                    // Invalid UTF-8 start byte
                    i++;
                }
            }
            
            return cleaned_result;
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
