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
    static constexpr bool is_static = true;
    static constexpr bool has_static_rank = true;
    using ShapeBase = std::array<uint32_t, rank>;
    static constexpr ShapeBase shape = {Dims...};

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    // Check that all Dims are > 0
    static_assert(((Dims > 0) && ...), "Shape dims must be greater than 0!");

    // Compute shape properities at compile time
    static constexpr std::pair<uint32_t, ShapeBase> compute_volume_and_strides(const ShapeBase& shape) {
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

template <uint32_t Rank>
struct ShapeWrapperDynamicDimsStaticRank {
    static constexpr uint32_t rank = Rank;
    static constexpr bool is_static = false;
    static constexpr bool has_static_rank = true;
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

struct ShapeWrapperDynamicRank {
    static constexpr bool is_static = false;
    static constexpr bool has_static_rank = false;
    static constexpr uint32_t rank = static_cast<uint32_t>(-1);  // Rank is not known at compile time
    using ShapeBase = Span<uint32_t>;

    // uint32_t shape_buffer[10];
    // NOTE: No additional buffer fo shape is required, since span in constructed on top of &get_common_arg_addr(BASE)
    uint32_t strides_buffer[MAX_RANK];  // TODO: Can we have rank higher than 10?
    uint32_t rank_rt = 0;
    ShapeBase shape;    // runtime shape
    ShapeBase strides;  // runtime strides
    size_t volume = 0;  // runtime volume

    explicit ShapeWrapperDynamicRank(ShapeBase&& shape_in) :
        rank_rt(shape_in.size()), shape(std::move(shape_in)), strides(strides_buffer, rank_rt) {
        ASSERT(rank_rt <= MAX_RANK);
        compute_volume_and_strides();
    }

    inline void compute_volume_and_strides() {
        uint32_t stride = 1;
        for (int i = rank_rt - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        volume = strides[0] * shape[0];
    }
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
