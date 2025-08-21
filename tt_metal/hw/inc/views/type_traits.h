#pragma once

namespace views {

template <auto Value>
struct ct_t {
    __attribute__((always_inline)) constexpr operator auto() const noexcept { return Value; }
};

template <auto Value>
inline constexpr ct_t<Value> ct{};

}  // namespace views
