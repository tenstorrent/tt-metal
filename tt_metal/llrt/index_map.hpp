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
    class Iterator final {
    public:
        Iterator(const std::vector<std::optional<T>>* data, size_t idx) noexcept: data_(data), idx_(idx) {
            for (;idx_ < data_->size() && !(*data_)[idx_].has_value(); idx_++) {}
        }
        ~Iterator() = default;
        Iterator(const Iterator&) noexcept = default;
        Iterator& operator=(const Iterator&) noexcept = default;
        Iterator(Iterator&&) noexcept = default;
        Iterator& operator=(Iterator&&) noexcept = default;
        [[nodiscard]] bool operator==(const Iterator& other) const noexcept {
            return idx_ == other.idx_;
        }
        [[nodiscard]] bool operator!=(const Iterator& other) const noexcept {
            return idx_ != other.idx_;
        }

        Iterator& operator++() noexcept {
            idx_++;
            for (;idx_ < data_->size() && !(*data_)[idx_].has_value(); idx_++) {}
            return *this;
        }

        [[nodiscard]] std::pair<size_t, const T&> operator*() const noexcept {
            return {idx_, (*data_)[idx_].value()};
        }
    private:
        const std::vector<std::optional<T>>* data_;
        size_t idx_ = 0;
    };

    IndexMap() = default;

    void clear() noexcept {
        data_.clear();
        size_ = 0;
    }

    [[nodiscard]] T& at(size_t idx) {
        return data_.at(idx).value();
    }
    [[nodiscard]] const T& at(size_t idx) const {
        return data_.at(idx).value();
    }
    [[nodiscard]] bool contains(size_t idx) const noexcept {
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

    [[nodiscard]] Iterator find(size_t idx) const noexcept {
        if (idx >= data_.size()) {
            return end();
        }
        if (!data_[idx].has_value()) {
            return end();
        }
        return Iterator(&data_, idx);
    }

    [[nodiscard]] size_t size() const noexcept {
        return size_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] Iterator begin() const noexcept {
        return Iterator(&data_, 0);
    }
    [[nodiscard]] Iterator end() const noexcept {
        return Iterator(&data_, data_.size());
    }

private:
    std::vector<std::optional<T>> data_;
    size_t size_ = 0;
};

}
