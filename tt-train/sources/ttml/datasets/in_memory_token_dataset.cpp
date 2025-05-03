// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "in_memory_token_dataset.hpp"

#include <cstddef>

namespace ttml::datasets {

InMemoryTokenDataset::InMemoryTokenDataset(const std::vector<uint32_t>& tokens, uint32_t seq_length) :
    m_tokens(tokens), m_seq_length(seq_length) {
}

[[nodiscard]] size_t InMemoryTokenDataset::get_size_impl() const {
    if (m_tokens.size() <= m_seq_length) {
        return 0UL;
    }
    return m_tokens.size() - m_seq_length;
}

[[nodiscard]] InMemoryTokenDataset::Sample InMemoryTokenDataset::get_item_impl(size_t index) const {
    size_t dataset_size = get_size_impl();
    if (index >= dataset_size) {
        throw std::out_of_range("Index out of range");
    }

    const auto* data_ptr = std::next(m_tokens.data(), static_cast<ptrdiff_t>(index));
    std::span<const uint32_t> input_span(data_ptr, m_seq_length);
    std::span<const uint32_t> target_span(std::next(data_ptr), m_seq_length);

    return {input_span, target_span};
}

}  // namespace ttml::datasets
