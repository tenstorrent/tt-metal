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

class TikTokenTokenizer : public TokenizerBase {
public:
    explicit TikTokenTokenizer(const std::string& json_file);
    ~TikTokenTokenizer() override;
    TikTokenTokenizer(const TikTokenTokenizer&) = delete;
    TikTokenTokenizer& operator=(const TikTokenTokenizer&) = delete;
    TikTokenTokenizer(TikTokenTokenizer&&) noexcept;
    TikTokenTokenizer& operator=(TikTokenTokenizer&&) noexcept;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const override;
    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const override;
    [[nodiscard]] uint32_t get_vocab_size() const override;

private:
    class TikTokenTokenizerImpl;
    std::unique_ptr<TikTokenTokenizerImpl> m_pimpl;
};

}  // namespace ttml::tokenizers