// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/memory_pin.hpp>

#include <functional>
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tt::tt_metal {

// HostBuffer is wrapper around data, which can either be owned or borrowed from external sources (Python objects,
// mmap-ed regions, etc).
class HostBuffer {
public:
    HostBuffer();
    ~HostBuffer() = default;

    // Constructors for `HostBuffer` based on the owned data.
    template <typename T>
    explicit HostBuffer(std::shared_ptr<std::vector<T>> data);

    template <typename T>
    explicit HostBuffer(std::vector<T>&& data);

    template <typename T>
    explicit HostBuffer(const std::vector<T>& data);

    // Constructor for `HostBuffer` based on the borrowed data.
    template <typename T>
    HostBuffer(tt::stl::Span<T> borrowed_data, MemoryPin pin);

    HostBuffer(const HostBuffer& other);
    HostBuffer& operator=(const HostBuffer& other);
    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;
    void swap(HostBuffer& other) noexcept;

    tt::stl::Span<std::byte> view_bytes() & noexcept;
    tt::stl::Span<const std::byte> view_bytes() const& noexcept;
    tt::stl::Span<std::byte> view_bytes() && noexcept = delete;
    tt::stl::Span<const std::byte> view_bytes() const&& noexcept = delete;

    template <typename T>
    tt::stl::Span<T> view_as() &;

    template <typename T>
    tt::stl::Span<const T> view_as() const&;

    template <typename T>
    tt::stl::Span<T> view_as() && = delete;

    template <typename T>
    tt::stl::Span<const T> view_as() const&& = delete;

    // Returns true if the data buffer is allocated.
    bool is_allocated() const;

    // Returns true if the data buffer is borrowed.
    bool is_borrowed() const;

    // Makes a deep copy of the data buffer.
    HostBuffer deep_copy() const;

    // Returns a pin for the data buffer.
    // The data won't be freed until the pin is destroyed.
    MemoryPin pin() const;

    // Deallocates the data buffer.
    void deallocate();

private:
    MemoryPin pin_;
    tt::stl::Span<std::byte> view_;
    const std::type_info* type_info_ = nullptr;
    bool is_borrowed_ = false;
};

template <typename T>
HostBuffer::HostBuffer(std::shared_ptr<std::vector<T>> data) {
    const size_t size_bytes = data->size() * sizeof(T);
    view_ = tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(data->data()), size_bytes);
    pin_ = MemoryPin(data);
    type_info_ = &typeid(T);
    is_borrowed_ = false;
}

template <typename T>
HostBuffer::HostBuffer(std::vector<T>&& data) :
    HostBuffer(std::shared_ptr<std::vector<T>>(std::make_shared<std::vector<T>>(std::move(data)))) {}

template <typename T>
HostBuffer::HostBuffer(const std::vector<T>& data) :
    HostBuffer(std::shared_ptr<std::vector<T>>(std::make_shared<std::vector<T>>(data))) {}

template <typename T>
HostBuffer::HostBuffer(tt::stl::Span<T> borrowed_data, MemoryPin pin) :
    view_(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(borrowed_data.data()), borrowed_data.size() * sizeof(T))),
    pin_(std::move(pin)),
    type_info_(&typeid(T)),
    is_borrowed_(true) {}

template <typename T>
tt::stl::Span<T> HostBuffer::view_as() & {
    TT_FATAL(*type_info_ == typeid(T), "Requested type T does not match the underlying buffer type.");
    return tt::stl::Span<T>(reinterpret_cast<T*>(view_.data()), view_.size() / sizeof(T));
}

template <typename T>
tt::stl::Span<const T> HostBuffer::view_as() const& {
    TT_FATAL(*type_info_ == typeid(T), "Requested type T does not match the underlying buffer type.");
    return tt::stl::Span<const T>(reinterpret_cast<const T*>(view_.data()), view_.size() / sizeof(T));
}

// Compares data buffers by their data.
bool operator==(const HostBuffer& a, const HostBuffer& b) noexcept;
bool operator!=(const HostBuffer& buffer_a, const HostBuffer& buffer_b) noexcept;

void swap(HostBuffer& lhs, HostBuffer& rhs) noexcept;

}  // namespace tt::tt_metal
