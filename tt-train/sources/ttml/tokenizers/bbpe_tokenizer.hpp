// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tokenizer_base.hpp"

namespace ttml::tokenizers {

class BBPETokenizer : public TokenizerBase {
public:
    explicit BBPETokenizer(const std::string& json_file);
    ~BBPETokenizer() override;
    BBPETokenizer(const BBPETokenizer&) = delete;
    BBPETokenizer& operator=(const BBPETokenizer&) = delete;
    BBPETokenizer(BBPETokenizer&&) noexcept;
    BBPETokenizer& operator=(BBPETokenizer&&) noexcept;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const override;
    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const override;
    [[nodiscard]] uint32_t get_vocab_size() const override;

private:
    class BBPETokenizerImpl;
    std::unique_ptr<BBPETokenizerImpl> m_pimpl;
};

}  // namespace ttml::tokenizers