// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace tt {

namespace tt_metal {

namespace borrowed_buffer {

template<typename T>
struct Buffer {
    explicit Buffer() = default;
    explicit Buffer(T* data_ptr, std::size_t size) :
        data_ptr_(data_ptr),
        size_(size) {}

    const std::size_t size() const { return this->size_; }

    inline T& operator[](std::size_t index) noexcept { return this->data_ptr_[index]; }
    inline const T& operator[](std::size_t index) const noexcept { return this->data_ptr_[index]; }

    inline T* begin() noexcept { return this->data_ptr_; }
    inline T* end() noexcept { return this->data_ptr_ + this->size(); }

    inline const T* begin() const noexcept { return this->data_ptr_; }
    inline const T* end() const noexcept { return this->data_ptr_ + this->size(); }

    inline void* data() noexcept { return static_cast<void*>(this->data_ptr_); }
    inline const void* data() const noexcept { return static_cast<void*>(this->data_ptr_); }

   private:
    T* data_ptr_;
    std::size_t size_;
};


template<typename T>
bool operator==(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    if (buffer_a.size() != buffer_b.size()) {
        return false;
    }
    for (auto index = 0; index < buffer_a.size(); index++) {
        if (buffer_a[index] != buffer_b[index]) {
            return false;
        }
    }
    return true;
}


template<typename T>
bool operator!=(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    return not (buffer_a == buffer_b);
}

}  // namespace borrowed_buffer

}  // namespace tt_metal

}  // namespace tt
