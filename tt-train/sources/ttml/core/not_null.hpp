// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdexcept>
#include <utility>

namespace ttml::core {

/*
Simplified gsl::not_null to comply with clang-tidy checks.
*/
template <typename T>
class not_null {
private:
    T m_ptr;

public:
    // Constructor
    explicit not_null(T ptr) : m_ptr(std::move(ptr)) {
        if (m_ptr == nullptr) {
            throw std::invalid_argument("Pointer must not be null");
        }
    }

    not_null() = delete;

    template <typename U>
    not_null(U) = delete;

    explicit operator T() const noexcept {
        return m_ptr;
    }

    // Dereference operators
    auto operator*() const noexcept -> decltype(*m_ptr) {
        return *m_ptr;
    }

    auto operator->() const noexcept -> T {
        return m_ptr;
    }

    // Get the underlying pointer
    T get() const noexcept {
        return m_ptr;
    }

    // Assignment operator
    not_null& operator=(T ptr) {
        if (ptr == nullptr) {
            throw std::invalid_argument("Pointer must not be null");
        }
        m_ptr = std::move(ptr);
        return *this;
    }
};

}  // namespace ttml::core
