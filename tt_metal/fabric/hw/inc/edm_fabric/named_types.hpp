// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"

namespace tt::tt_fabric {

template <typename T, typename Parameter>
class NamedType {
public:
    FORCE_INLINE NamedType() = default;
    FORCE_INLINE explicit NamedType(const T& value) : value_(value) {}
    FORCE_INLINE explicit NamedType(T&& value) : value_(std::move(value)) {}
    FORCE_INLINE NamedType<T, Parameter>& operator=(const NamedType<T, Parameter>& rhs) = default;
    FORCE_INLINE T& get() { return value_; }
    FORCE_INLINE const T& get() const { return value_; }
    FORCE_INLINE operator T() const { return value_; }
    FORCE_INLINE operator T&() { return value_; }

private:
    T value_;
};

}  // namespace tt::tt_fabric
