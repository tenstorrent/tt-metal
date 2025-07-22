// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/overloaded.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace tt::tt_metal {

HostBuffer::HostBuffer() : pin_(), view_(tt::stl::Span<std::byte>()), type_info_(nullptr) {}

HostBuffer::HostBuffer(const HostBuffer& other) = default;

HostBuffer& HostBuffer::operator=(const HostBuffer& other) {
    HostBuffer temp(other);
    swap(temp);
    return *this;
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept : HostBuffer() { swap(other); }

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    swap(other);
    return *this;
}

void HostBuffer::swap(HostBuffer& other) noexcept {
    using std::swap;
    swap(pin_, other.pin_);
    swap(view_, other.view_);
    swap(type_info_, other.type_info_);
}

tt::stl::Span<std::byte> HostBuffer::view_bytes() & noexcept { return view_; }

tt::stl::Span<const std::byte> HostBuffer::view_bytes() const& noexcept { return view_; }

bool operator==(const HostBuffer& a, const HostBuffer& b) noexcept {
    auto a_view = a.view_bytes();
    auto b_view = b.view_bytes();
    if (a_view.size() != b_view.size()) {
        return false;
    }
    for (auto i = 0; i < a_view.size(); i++) {
        if (a_view[i] != b_view[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const HostBuffer& a, const HostBuffer& b) noexcept { return !(a == b); }

void swap(HostBuffer& lhs, HostBuffer& rhs) noexcept { lhs.swap(rhs); }

}  // namespace tt::tt_metal
