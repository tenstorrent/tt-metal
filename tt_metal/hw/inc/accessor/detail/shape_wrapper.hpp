#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nd_sharding {
namespace detail {
template <size_t... Dims>
struct ShapeWrapperStaticDimsStaticRank {
    static constexpr size_t rank = sizeof...(Dims);
    static constexpr bool is_static = true;
    using ShapeBase = std::array<uint32_t, rank>;
    static constexpr ShapeBase shape = {Dims...};

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    // Check that all Dims are > 0
    static_assert(((Dims > 0) && ...), "Shape dims must be greater than 0!");

    // Compute shape properities at compile time
    static constexpr std::pair<size_t, ShapeBase> compute_volume_and_strides(const ShapeBase& shape) {
        ShapeBase strides = {};
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return {strides[0] * shape[0], strides};
    }

    // Compiler should optimize out the second call
    static constexpr auto volume = compute_volume_and_strides(shape).first;
    static constexpr auto strides = compute_volume_and_strides(shape).second;

    constexpr explicit ShapeWrapperStaticDimsStaticRank() = default;
    constexpr explicit ShapeWrapperStaticDimsStaticRank(const ShapeBase&) {}
    constexpr explicit ShapeWrapperStaticDimsStaticRank(ShapeBase&&) {}
};

template <size_t Rank>
struct ShapeWrapperDynamicDimsStaticRank {
    static constexpr size_t rank = Rank;
    static constexpr bool is_static = false;
    using ShapeBase = std::array<uint32_t, rank>;
    ShapeBase shape;    // runtime shape
    ShapeBase strides;  // runtime strides
    size_t volume;      // runtime volume

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    template <class... Ts, std::enable_if_t<sizeof...(Ts) == Rank, int> = 0>
    constexpr explicit ShapeWrapperDynamicDimsStaticRank(Ts... exts) : shape{static_cast<uint32_t>(exts)...} {
        compute_volume_and_strides(shape);
    }

    constexpr explicit ShapeWrapperDynamicDimsStaticRank() = default;

    constexpr explicit ShapeWrapperDynamicDimsStaticRank(const ShapeBase& shape) : shape{shape} {
        compute_volume_and_strides(shape);
    }

    constexpr explicit ShapeWrapperDynamicDimsStaticRank(ShapeBase&& shape) : shape{std::move(shape)} {
        compute_volume_and_strides(shape);
    }

    inline void compute_volume_and_strides(const ShapeBase& shape) {
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        volume = strides[0] * shape[0];
    }
};

}  // namespace detail
}  // namespace nd_sharding
