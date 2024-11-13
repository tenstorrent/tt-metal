// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <span>
#include <vector>

namespace ttml::datasets {
template <class Derived, class DataType, class TargetType>
class DatasetBase {
public:
    using DataTypeT = DataType;
    using TargetTypeT = TargetType;
    using Sample = std::pair<DataType, TargetType>;
    // using Samples = std::pair<std::vector<DataType>, std::vector<TargetType>>; TODO: consider return Samples instead
    // of vector<Sample>
    DatasetBase() = default;
    DatasetBase(const DatasetBase&) = default;
    DatasetBase(DatasetBase&&) = default;
    DatasetBase& operator=(const DatasetBase&) = default;
    DatasetBase& operator=(DatasetBase&&) = default;
    ~DatasetBase() = default;

    [[nodiscard]] size_t get_size() const {
        return static_cast<const Derived*>(this)->get_size_impl();
    }

    [[nodiscard]] Sample get_item(size_t index) const {
        return static_cast<const Derived*>(this)->get_item_impl(index);
    }

    [[nodiscard]] std::vector<Sample> get_batch(std::span<size_t> indices) const {
        std::vector<Sample> batch;
        auto size = get_size();
        for (size_t index : indices) {
            assert(index < size);
            batch.push_back(get_item(index));
        }
        return batch;
    }
};
}  // namespace ttml::datasets
