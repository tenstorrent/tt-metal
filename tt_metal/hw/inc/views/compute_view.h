#pragma once

#include "type_traits.h"

#include "compile_time_args.h"
#include "compute_kernel_api/cb_api.h"

#include <cstddef>
#include <cstdint>

namespace views {

template <size_t Accessors, size_t Offset = 0, uint32_t DefaultNumTilesPerCycle = 1, uint32_t... Indices>
class ComputeView {
    static constexpr uint32_t cb_index = get_compile_time_arg_val(Offset);

    ComputeView<Accessors - 1, Offset + 1, DefaultNumTilesPerCycle, Indices...> tail;

public:
    // compute members

    template <uint32_t... Args, typename Init>
    __attribute__((always_inline)) auto init_tiles(Init init) noexcept {
        if constexpr (Accessors > 1) {
            tail.template init_tiles<Args..., cb_index>(init);
        } else {
            init(ct<DefaultNumTilesPerCycle>, ct<Args>..., ct<cb_index>, ct<Indices>...);
        }
    }

    template <uint32_t... Args, typename Compute>
    __attribute__((always_inline)) auto compute_tile(Compute compute) noexcept {
        if constexpr (Accessors > 1) {
            tail.template compute_tile<Args..., cb_index>(compute);
        } else {
            compute(ct<Args>..., ct<cb_index>, ct<Indices>...);
        }
    }

    template <uint32_t NumTilesPerCycle = DefaultNumTilesPerCycle, typename Compute>
    auto compute_tiles(uint32_t tiles, Compute compute) noexcept {
        for (; tiles >= NumTilesPerCycle; tiles -= NumTilesPerCycle) {
            compute_tile<NumTilesPerCycle>(compute);
        }

        if constexpr (NumTilesPerCycle > 1) {
            if (tiles > 0) {
                compute_tiles<1>(tiles, compute);
            }
        }
    }
};

template <size_t Offset, uint32_t DefaultNumTilesPerCycle, uint32_t... Indices>
class ComputeView<0, Offset, DefaultNumTilesPerCycle, Indices...> {
public:
    __attribute__((always_inline)) ComputeView(int arg_idx = 0) {}
};

}  // namespace views
