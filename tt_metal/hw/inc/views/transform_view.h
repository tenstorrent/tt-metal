#pragma once

#include "type_traits.h"

#include <cstddef>
#include <cstdint>

namespace views {

template <typename F>
struct pipeable : F {
    using F::operator();

    template <typename G>
    constexpr auto operator|(pipeable<G> other) const noexcept {
        F f = *this;
        G g = other;
        return views::pipeable{
            [f, g](auto... indices) {
                f(indices...);
                g(indices...);
            },
        };
    }
};

template <typename F>
pipeable(F) -> pipeable<F>;

template <size_t... Is, typename Compute>
constexpr auto transform(Compute compute) {
    return views::pipeable{
        [compute](auto... args) {
            constexpr uint32_t array[]{args...};
            compute(ct<array[Is]>...);
        },
    };
}

}  // namespace views
