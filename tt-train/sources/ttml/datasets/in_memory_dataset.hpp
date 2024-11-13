// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataset_base.hpp"

namespace ttml::datasets {
template <class DataType, class TargetType>
class InMemoryDataset : public DatasetBase<InMemoryDataset<DataType, TargetType>, DataType, TargetType> {
public:
    using Parent = DatasetBase<InMemoryDataset<DataType, TargetType>, DataType, TargetType>;
    using Sample = typename Parent::Sample;
    friend Parent;

    InMemoryDataset(const std::vector<DataType>& data, const std::vector<TargetType>& targets) :
        m_data(data), m_targets(targets) {
    }

    InMemoryDataset(const InMemoryDataset&) = default;
    InMemoryDataset(InMemoryDataset&&) = default;
    InMemoryDataset& operator=(const InMemoryDataset&) = default;
    InMemoryDataset& operator=(InMemoryDataset&&) = default;
    ~InMemoryDataset() = default;

private:
    [[nodiscard]] size_t get_size_impl() const {
        return m_data.size();
    }

    [[nodiscard]] Sample get_item_impl(size_t index) const {
        return {m_data[index], m_targets[index]};
    }
    std::vector<DataType> m_data;
    std::vector<TargetType> m_targets;
};
}  // namespace ttml::datasets
