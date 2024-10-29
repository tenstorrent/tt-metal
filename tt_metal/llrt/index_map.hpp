// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>

namespace tt {

/**
 * IndexMap is a container, which mimics std::map with an assumption that all keys are indexes.
 * Provides vector-like performance, consumes the size proportional to the maximum used index.
 * Implemented as an auto-resizing vector of optionals.
 */
template <typename T>
class IndexMap {
public:
    class Iterator {
    public:
        Iterator(const std::vector<std::optional<T>>& data, size_t idx): data_(data), idx_(idx) {
            for (;idx_ < data_.size() && !data_[idx_].has_value(); idx_++);
        }
        Iterator(const Iterator&) = default;
        Iterator& operator=(const Iterator&) = default;
        bool operator==(const Iterator& other) const {
            return idx_ == other.idx_;
        }
        bool operator!=(const Iterator& other) const {
            return idx_ != other.idx_;
        }

        Iterator& operator++() {
            idx_++;
            for (;idx_ < data_.size() && !data_[idx_].has_value(); idx_++);
            return *this;
        }

        std::pair<size_t, const T&> operator*() const {
            return {idx_, data_[idx_].value()};
        }
    private:
        const std::vector<std::optional<T>>& data_;
        size_t idx_ = 0;
    };

    IndexMap() = default;

    void clear() {
        data_.clear();
        size_ = 0;
    }

    T& at(size_t idx) {
        return data_.at(idx).value();
    }
    const T& at(size_t idx) const {
        return data_.at(idx).value();
    }
    bool contains(size_t idx) const {
        if (idx >= data_.size()) {
            return false;
        }
        return data_[idx].has_value();
    }
    T& operator[](size_t idx) {
        if (idx >= data_.size()) {
            data_.resize(idx + 1);
        }
        if (!data_[idx].has_value()) {
            data_[idx] = T();
            size_++;
        }
        return *data_[idx];
    }
    bool insert(size_t idx, T value) {
        if(idx >= data_.size()) {
            data_.resize(idx + 1);
        }
        if (!data_[idx].has_value()) {
            data_[idx] = std::move(value);
            size_++;
            return true;
        }
        return false;
    }

    Iterator find(size_t idx) const {
        if (idx >= data_.size()) {
            return end();
        }
        if (!data_[idx].has_value()) {
            return end();
        }
        return Iterator(data_, idx);
    }

    size_t size() const {
        return size_;
    }
    bool empty() const {
        return size_ == 0;
    }

    Iterator begin() const {
        return Iterator(data_, 0);
    }
    Iterator end() const {
        return Iterator(data_, data_.size());
    }

private:
    std::vector<std::optional<T>> data_;
    size_t size_ = 0;
};

}
