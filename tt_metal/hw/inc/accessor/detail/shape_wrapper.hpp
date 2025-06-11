#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "const.hpp"
#include "helpers.hpp"

namespace nd_sharding {
namespace detail {
template <uint32_t... Dims>
struct ShapeWrapperStaticDimsStaticRank {
    static constexpr uint32_t rank = sizeof...(Dims);
    using ShapeBase = std::array<uint32_t, rank>;
    static constexpr ShapeBase shape = {Dims...};

    static_assert(rank > 0, "Shape rank must be greater than 0!");
    static_assert(((Dims > 0) && ...), "Shape dims must be greater than 0!");

    constexpr explicit ShapeWrapperStaticDimsStaticRank() = default;
    constexpr explicit ShapeWrapperStaticDimsStaticRank(const ShapeBase&) {}
    constexpr explicit ShapeWrapperStaticDimsStaticRank(ShapeBase&&) {}
};

template <uint32_t Rank>
struct ShapeWrapperDynamicDimsStaticRank {
    static constexpr uint32_t rank = Rank;
    using ShapeBase = std::array<uint32_t, rank>;
    ShapeBase shape;  // runtime shape
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    constexpr explicit ShapeWrapperDynamicDimsStaticRank() = default;
    constexpr explicit ShapeWrapperDynamicDimsStaticRank(const ShapeBase& shape) : shape{shape} {}
    constexpr explicit ShapeWrapperDynamicDimsStaticRank(ShapeBase&& shape) : shape{std::move(shape)} {}
};

struct ShapeWrapperDynamicRank {
    static constexpr uint32_t rank = static_cast<uint32_t>(-1);  // Rank is not known at compile time
    using ShapeBase = Span<uint32_t>;
    ShapeBase shape;  // runtime shape

    constexpr explicit ShapeWrapperDynamicRank() = default;
    constexpr explicit ShapeWrapperDynamicRank(const ShapeBase& shape_in) : shape(shape_in) {}
    explicit ShapeWrapperDynamicRank(ShapeBase&& shape_in) : shape(std::move(shape_in)) {}
};

template <bool RankCTA, bool ShapeStatic, size_t StartIdx, uint32_t Rank>
struct ShapeWrapperTypeSelector;

template <size_t StartIdx, uint32_t Rank>
struct ShapeWrapperTypeSelector<true, true, StartIdx, Rank> {
    // Both rank and dims are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<ShapeWrapperStaticDimsStaticRank, StartIdx, Rank>;
};

template <size_t StartIdx, uint32_t Rank>
struct ShapeWrapperTypeSelector<true, false, StartIdx, Rank> {
    // Rank is known at compile time, but dims are not
    using type = ShapeWrapperDynamicDimsStaticRank<Rank>;
};

template <bool ShapeStatic, size_t StartIdx, uint32_t Rank>
struct ShapeWrapperTypeSelector<false, ShapeStatic, StartIdx, Rank> {
    // Rank is not known at compile time, doesn't matter if dims are known or not, use poorly dynamic wrapper
    using type = ShapeWrapperDynamicRank;
};

}  // namespace detail
}  // namespace nd_sharding
