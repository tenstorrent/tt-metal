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

class BPETokenizer : public TokenizerBase {
public:
    explicit BPETokenizer(const std::string& json_file);
    ~BPETokenizer() override;
    BPETokenizer(const BPETokenizer&) = delete;
    BPETokenizer& operator=(const BPETokenizer&) = delete;
    BPETokenizer(BPETokenizer&&) noexcept;
    BPETokenizer& operator=(BPETokenizer&&) noexcept;

    [[nodiscard]] std::vector<uint32_t> encode(const std::string& text) const override;
    [[nodiscard]] std::string decode(const std::vector<uint32_t>& tokens) const override;
    [[nodiscard]] uint32_t get_vocab_size() const override;

private:
    class BPETokenizerImpl;
    std::unique_ptr<BPETokenizerImpl> m_pimpl;
};

}  // namespace ttml::tokenizers
