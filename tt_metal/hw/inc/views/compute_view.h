#pragma once

#include "type_traits.h"

#include "compile_time_args.h"
#include "compute_kernel_api/cb_api.h"

#include <cstddef>
#include <cstdint>

namespace views {

template <uint32_t DefaultNumTilesPerCycle = 1, uint32_t... CBIndices>
struct ComputeView {
    template <typename Init>
    __attribute__((always_inline)) static void init_tiles(Init init) noexcept {
        return init(views::ct<DefaultNumTilesPerCycle>, views::ct<CBIndices>...);
    }

    template <uint32_t NumTilesPerCycle = DefaultNumTilesPerCycle, typename... Compute>
    static void compute_tiles(uint32_t tiles, Compute... compute) noexcept {
        for (; tiles >= NumTilesPerCycle; tiles -= NumTilesPerCycle) {
            (..., compute(views::ct<NumTilesPerCycle>, views::ct<CBIndices>...));
        }

        // only instantiate recursion when stride could allow remaining tiles > 0
        if constexpr (NumTilesPerCycle > 1) {
            if (tiles > 0) {
                return ComputeView::compute_tiles<1>(tiles, compute...);
            }
        }
    }
};

template <class CtArgIndices>
struct ComputeViewType;

template <size_t... CBIndices>
struct ComputeViewType<std::index_sequence<CBIndices...>> {
    using type = ComputeView<get_compile_time_arg_val(0), CBIndices...>;
};

// convenience alias to fetch ct args
template <size_t CBIndices>
using MakeComputeView = typename ComputeViewType<std::make_index_sequence<CBIndices>>::type;

// empty base case
template <size_t Offset, uint32_t DefaultNumTilesPerCycle, uint32_t... Indices>
class ComputeView<0, Offset, DefaultNumTilesPerCycle, Indices...> {};

}  // namespace views
