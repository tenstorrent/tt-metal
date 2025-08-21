#pragma once

#include "type_traits.h"

#include "compile_time_args.h"
#include "compute_kernel_api/cb_api.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace views {

template <size_t Accessors, size_t Offset = 0, uint32_t DefaultNumTilesPerCycle = 1, uint32_t... Indices>
class ComputeView {
    using Next = ComputeView<Accessors - 1, Offset + 1, DefaultNumTilesPerCycle, Indices...>;
    static constexpr uint32_t cb_index = get_compile_time_arg_val(Offset);

public:
    // compute members

    template <uint32_t... Args, typename Init>
    __attribute__((always_inline)) static auto init_tiles(Init init) noexcept {
        if constexpr (Accessors > 1) {
            Next::template init_tiles<Args..., cb_index>(init);
        } else {
            init(views::ct<DefaultNumTilesPerCycle>, views::ct<Args>..., views::ct<cb_index>, views::ct<Indices>...);
        }
    }

    template <uint32_t... Args, typename Compute>
    __attribute__((always_inline)) static auto compute_tile(Compute compute) noexcept {
        // recursive partial application of cb_index corresponding to input
        if constexpr (Accessors > 1) {
            Next::template compute_tile<Args..., cb_index>(compute);
        } else {
            compute(views::ct<Args>..., views::ct<cb_index>, views::ct<Indices>...);
        }
    }

    template <uint32_t NumTilesPerCycle = DefaultNumTilesPerCycle, typename Compute>
    static auto compute_tiles(uint32_t tiles, Compute compute) noexcept {
        for (; tiles >= NumTilesPerCycle; tiles -= NumTilesPerCycle) {
            ComputeView::compute_tile<NumTilesPerCycle>(compute);
        }

        // only instantiate recursion when stride could allow remaining tiles > 0
        if constexpr (NumTilesPerCycle > 1) {
            if (tiles > 0) {
                ComputeView::compute_tiles<1>(tiles, compute);
            }
        }
    }
};

// convenience alias to calculate initial offset for ct args
template <size_t Accessors, uint32_t... CtArgs>
using MakeComputeView = ComputeView<Accessors, sizeof...(CtArgs), CtArgs...>;

// empty base case
template <size_t Offset, uint32_t DefaultNumTilesPerCycle, uint32_t... Indices>
class ComputeView<0, Offset, DefaultNumTilesPerCycle, Indices...> {};

}  // namespace views
