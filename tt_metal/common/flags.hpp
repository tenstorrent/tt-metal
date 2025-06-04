#pragma once

#include <type_traits>

template <typename E>
struct Flags {
    static_assert(std::is_enum_v<E>, "Flags<E> requires E to be an enum.");

    using Underlying = std::underlying_type_t<E>;

    constexpr Flags() noexcept : bits_(0) {}
    constexpr Flags(E single) noexcept : bits_(static_cast<Underlying>(single)) {}
    constexpr Flags(E a, E b) noexcept : bits_(static_cast<Underlying>(a) | static_cast<Underlying>(b)) {}
    constexpr explicit Flags(Underlying bits) noexcept : bits_(bits) {}

    // Bitwise OR a single enum
    constexpr Flags operator|(E rhs) const noexcept { return Flags(bits_ | static_cast<Underlying>(rhs)); }
    // Bitwise OR another Flags
    constexpr Flags operator|(Flags rhs) const noexcept { return Flags(bits_ | rhs.bits_); }

    // ... similarly, operator&, operator^, operator~ if you like ...
    constexpr Flags operator&(E rhs) const noexcept { return Flags(bits_ & static_cast<Underlying>(rhs)); }
    constexpr bool test(E single) const noexcept {
        return (bits_ & static_cast<Underlying>(single)) == static_cast<Underlying>(single);
    }

    constexpr Underlying raw() const noexcept { return bits_; }

private:
    Underlying bits_;
};
