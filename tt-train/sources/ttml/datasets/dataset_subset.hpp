// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/not_null.hpp"
#include "dataset_base.hpp"

namespace ttml::datasets {

template <typename DatasetType>
class DatasetSubset : public DatasetBase<
                          DatasetSubset<DatasetType>,
                          typename DatasetType::DataTypeT,
                          typename DatasetType::TargetTypeT> {
public:
    DatasetSubset(const DatasetType& dataset, const std::vector<size_t>& indices) :
        m_dataset(&dataset), m_indices(indices) {
    }

    [[nodiscard]] size_t get_size_impl() const {
        return m_indices.size();
    }

    [[nodiscard]] DatasetType::Sample get_item_impl(size_t index) const {
        if (index >= m_indices.size()) {
            throw std::out_of_range("Index out of range.");
        }
        return m_dataset->get_item(m_indices[index]);
    }

private:
    core::not_null<const DatasetType*> m_dataset;
    std::vector<size_t> m_indices;
};

}  // namespace ttml::datasets
