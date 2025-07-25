// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

template <typename E>
class Flags {
public:
    static_assert(std::is_enum_v<E>, "Flags<E> requires E to be an enum.");

    using Underlying = std::underlying_type_t<E>;

    constexpr Flags() noexcept : bits_(0) {}
    constexpr Flags(E value) noexcept : bits_{static_cast<Underlying>(value)} {}
    constexpr Flags(E a, E b) noexcept : bits_(static_cast<Underlying>(a) | static_cast<Underlying>(b)) {}
    constexpr explicit Flags(Underlying bits) noexcept : bits_(bits) {}

    // Bitwise OR a single enum
    constexpr Flags operator|(E other) const noexcept { return Flags(bits_ | static_cast<Underlying>(other)); }
    // Bitwise OR another Flags
    constexpr Flags operator|(Flags other) const noexcept { return Flags(bits_ | other.bits_); }

    constexpr Flags operator&(E other) const noexcept { return Flags(bits_ & static_cast<Underlying>(other)); }
    constexpr bool test(E single) const noexcept {
        return (bits_ & static_cast<Underlying>(single)) == static_cast<Underlying>(single);
    }
    void set(E single, bool value = true) noexcept {
        if (value) {
            bits_ |= static_cast<Underlying>(single);
        } else {
            bits_ &= ~static_cast<Underlying>(single);
        }
    }

    constexpr Underlying raw() const noexcept { return bits_; }

private:
    Underlying bits_ = 0;
};
