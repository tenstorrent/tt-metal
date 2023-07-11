#pragma once

#include <vector>

template<typename T>
class span_t {
   T* pointer_;
   std::size_t size_;

public:
    span_t(T* pointer, std::size_t size) noexcept
        : pointer_{pointer}, size_{size}
    {}
    span_t(std::vector<T>& vector) noexcept
        : pointer_{vector.data()}, size_{vector.size()}
    {}

    T& operator[](std::size_t index) noexcept {
        return this->pointer_[index];
    }

    const T& operator[](std::size_t index) const noexcept {
        return this->pointer_[index];
    }

    std::size_t size() const noexcept {
        return this->size_;
    }

    T* begin() noexcept {
        return this->pointer_;
    }

    T* end() noexcept {
        return this->pointer_ + this->size_;
    }

    const T* begin() const noexcept {
        return this->pointer_;
    }

    const T* end() const noexcept {
        return this->pointer_ + this->size_;
    }
};


template<typename T>
bool operator==(const span_t<T>& span_a, const span_t<T>& span_b) noexcept {
    if (span_a.size() != span_b.size()) {
        return false;
    }
    for (auto index = 0; index < span_a.size(); index++) {
        if (span_a[index] != span_b[index]) {
            return false;
        }
    }
    return true;
}


template<typename T>
bool operator!=(const span_t<T>& span_a, const span_t<T>& span_b) noexcept {
    return not (span_a == span_b);
}
