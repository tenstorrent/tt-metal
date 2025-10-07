#pragma once

#include "type_traits.h"

#include <cstddef>
#include <cstdint>

namespace views {

template <size_t... Is, typename... Compute>
constexpr auto transform(Compute... compute) {
    return [=](auto... args) {
        constexpr uint32_t array[]{args...};
        (..., compute(ct<array[Is]>...));
    };
}

}  // namespace views
