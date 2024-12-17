// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/not_null.hpp"
namespace ttml::datasets {

template <typename SampleType>
std::vector<SampleType> default_collate_fn(std::vector<SampleType>&& samples) {
    return std::forward<std::vector<SampleType>>(samples);
}

template <
    typename DatasetType,
    typename CollateFn =
        std::function<std::vector<typename DatasetType::Sample>(std::vector<typename DatasetType::Sample>&&)>,
    typename BatchType = std::vector<typename DatasetType::Sample>>
class DataLoader {
public:
    using Sample = typename DatasetType::Sample;

    DataLoader(
        DatasetType& dataset,
        size_t batch_size,
        bool shuffle = false,
        CollateFn collate_fn = default_collate_fn<Sample>) :
        m_dataset(&dataset),
        m_batch_size(batch_size),
        m_shuffle(shuffle),
        m_indices(dataset.get_size()),
        m_collate_fn(collate_fn) {
        std::iota(m_indices.begin(), m_indices.end(), 0);
    }

    void shuffle_indices() {
        if (!m_shuffle) {
            return;
        }
        std::mt19937& gen = autograd::AutoContext::get_instance().get_generator();
        std::shuffle(m_indices.begin(), m_indices.end(), gen);
    }

    class Iterator {
    public:
        Iterator(DataLoader& data_loader, size_t start_index) :
            m_data_loader(&data_loader), m_current_index(start_index) {
        }

        Iterator& operator++() {
            m_current_index += m_data_loader->m_batch_size;
            m_current_index = std::min(m_current_index, m_data_loader->m_indices.size());
            return *this;
        }

        BatchType operator*() const {
            return m_data_loader->fetch_batch(m_current_index);
        }

        bool operator!=(const Iterator& other) const {
            return m_current_index != other.m_current_index;
        }

    private:
        core::not_null<DataLoader*> m_data_loader;
        size_t m_current_index = 0;
    };

    Iterator begin() {
        shuffle_indices();
        return Iterator(*this, 0);
    }

    Iterator end() {
        return Iterator(*this, m_indices.size());
    }

private:
    core::not_null<DatasetType*> m_dataset;
    size_t m_batch_size = 0;
    bool m_shuffle = false;
    std::vector<size_t> m_indices;
    CollateFn m_collate_fn;

    BatchType fetch_batch(size_t start_index) const {
        size_t end_index = std::min(start_index + m_batch_size, m_indices.size());
        std::vector<Sample> batch;
        batch.reserve(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            batch.push_back(m_dataset->get_item(m_indices[i]));
        }

        return m_collate_fn(std::move(batch));
    }
};
}  // namespace ttml::datasets
