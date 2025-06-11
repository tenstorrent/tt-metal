// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ttml::tokenizers {

class TokenizerBase {
public:
    TokenizerBase() = default;
    TokenizerBase(const TokenizerBase&) = default;
    TokenizerBase& operator=(const TokenizerBase&) = default;
    TokenizerBase(TokenizerBase&&) = default;
    TokenizerBase& operator=(TokenizerBase&&) = default;

    // Virtual destructor for proper cleanup in derived classes
    virtual ~TokenizerBase() = default;

    // Pure virtual function to encode a string into a vector of token IDs
    [[nodiscard]] virtual std::vector<uint32_t> encode(const std::string& text) const = 0;

    // Pure virtual function to decode a vector of token IDs back into a string
    [[nodiscard]] virtual std::string decode(const std::vector<uint32_t>& tokens) const = 0;

    // Pure virtual function to get the vocabulary size
    [[nodiscard]] virtual uint32_t get_vocab_size() const = 0;
};

}  // namespace ttml::tokenizers
