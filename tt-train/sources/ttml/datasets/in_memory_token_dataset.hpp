// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "dataset_base.hpp"

namespace ttml::datasets {
class InMemoryTokenDataset
    : public DatasetBase<InMemoryTokenDataset, std::span<const uint32_t>, std::span<const uint32_t>> {
public:
    using Parent = DatasetBase<InMemoryTokenDataset, std::span<const uint32_t>, std::span<const uint32_t>>;
    using Sample = typename Parent::Sample;
    friend Parent;

    InMemoryTokenDataset(const std::vector<uint32_t>& tokens, uint32_t seq_length);

    InMemoryTokenDataset(const InMemoryTokenDataset&) = default;
    InMemoryTokenDataset(InMemoryTokenDataset&&) = default;
    InMemoryTokenDataset& operator=(const InMemoryTokenDataset&) = default;
    InMemoryTokenDataset& operator=(InMemoryTokenDataset&&) = default;
    ~InMemoryTokenDataset() = default;

private:
    [[nodiscard]] size_t get_size_impl() const;

    [[nodiscard]] Sample get_item_impl(size_t index) const;

    std::vector<uint32_t> m_tokens;
    uint32_t m_seq_length = 0;
};
}  // namespace ttml::datasets
