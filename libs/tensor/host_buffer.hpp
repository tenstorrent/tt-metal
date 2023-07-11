#pragma once

#include <memory>
#include <vector>

namespace tt {

namespace tt_metal {

namespace host_buffer {

template<typename T>
struct HostBufferForDataType {

    explicit HostBufferForDataType(std::shared_ptr<std::vector<T>>&& shared_vector) :
        shared_vector_(shared_vector),
        pointer_for_faster_access_(shared_vector->data()),
        size_(shared_vector->size()) {}

    const std::size_t size() const { return this->size_; }

    inline T& operator[](std::size_t index) noexcept { return this->pointer_for_faster_access_[index]; }
    inline const T& operator[](std::size_t index) const noexcept { return this->pointer_for_faster_access_[index]; }

    inline T* begin() noexcept { return this->pointer_for_faster_access_; }
    inline T* end() noexcept { return this->pointer_for_faster_access_ + this->size(); }

    inline const T* begin() const noexcept { return this->pointer_for_faster_access_; }
    inline const T* end() const noexcept { return this->pointer_for_faster_access_ + this->size(); }

    inline bool is_allocated() const{ return bool(this->shared_vector_); }
    inline const std::vector<T>& get() const { return *this->shared_vector_; }
    inline void reset() { this->shared_vector_.reset(); }

  private:
    std::shared_ptr<std::vector<T>> shared_vector_;
    T* pointer_for_faster_access_;
    std::size_t size_;
};


template<typename T>
bool operator==(const HostBufferForDataType<T>& host_buffer_a, const HostBufferForDataType<T>& host_buffer_b) noexcept {
    if (host_buffer_a.size() != host_buffer_b.size()) {
        return false;
    }
    for (auto index = 0; index < host_buffer_a.size(); index++) {
        if (host_buffer_a[index] != host_buffer_b[index]) {
            return false;
        }
    }
    return true;
}


template<typename T>
bool operator!=(const HostBufferForDataType<T>& host_buffer_a, const HostBufferForDataType<T>& host_buffer_b) noexcept {
    return not (host_buffer_a == host_buffer_b);
}

}  // namespace host_buffer

}  // namespace tt_metal

}  // namespace tt
