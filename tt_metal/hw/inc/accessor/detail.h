// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tuple>

namespace detail {

template <size_t... Dims>
struct ShapeWrapper {
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
};

template <size_t Rank>
struct ShapeWrapperDynamic {
    static constexpr size_t rank = Rank;
    static constexpr bool is_static = false;
    using ShapeBase = std::array<uint32_t, rank>;
    ShapeBase shape;    // runtime shape
    ShapeBase strides;  // runtime strides
    size_t volume;      // runtime volume

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    template <class... Ts, std::enable_if_t<sizeof...(Ts) == Rank, int> = 0>
    constexpr explicit ShapeWrapperDynamic(Ts... exts) : shape{static_cast<uint32_t>(exts)...} {
        compute_volume_and_strides(shape);
    }

    constexpr explicit ShapeWrapperDynamic(const ShapeBase& shape) : shape{shape} { compute_volume_and_strides(shape); }

    constexpr explicit ShapeWrapperDynamic(ShapeBase&& shape) : shape{std::move(shape)} {
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

template <size_t... PackedCoords>
struct BankCoordWrapper {
    static constexpr size_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    static constexpr std::array<uint32_t, num_banks> packed_xy_coords = {PackedCoords...};
};

//
template <template <size_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
constexpr auto make_struct_from_sequence_wrapper(std::index_sequence<Is...>)
    -> Wrapper<get_compile_time_arg_val(BASE_IDX + Is)...>;

template <template <size_t...> class Wrapper, size_t base, size_t rank>
using struct_sequence_wrapper_t =
    decltype(make_struct_from_sequence_wrapper<Wrapper, base>(std::make_index_sequence<rank>{}));

template <typename TensorShape, typename ShardShape, typename BankCoords>
struct DistributionSpec {
    static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
    static_assert(
        std::is_same_v<typename TensorShape::ShapeBase, typename ShardShape::ShapeBase>,
        "Tensor and shard shapes bases must be the same");
    static constexpr auto rank = TensorShape::rank;
    using ShapeBase = typename TensorShape::ShapeBase;
    static_assert(rank > 0, "Tensor and shard shape ranks must be greater than 0!");

    static constexpr bool is_static = TensorShape::is_static && ShardShape::is_static;

    template <
        class... Ts1,
        class... Ts2,
        std::enable_if_t<
            (sizeof...(Ts1) == (TensorShape::is_static ? 0 : rank) &&
             sizeof...(Ts2) == (ShardShape::is_static ? 0 : rank)),
            int> = 0>
    constexpr DistributionSpec(
        std::tuple<Ts1...> tensor_shape = {},  // empty tuple == “nothing to pass”
        std::tuple<Ts2...> shard_shape = {}) :
        tensor_shape_{std::make_from_tuple<TensorShape>(tensor_shape)},
        shard_shape_{std::make_from_tuple<ShardShape>(shard_shape)} {
        // Check that the number of shards is greater than or equal to the number of banks
        // Here, shard_grid_strides[0] * shard_grid[0] is the total number of shards
        if constexpr (!is_static) {
            compute_shard_grid_and_strides_rt(tensor_shape_.shape, shard_shape_.shape);
            ASSERT(
                shard_grid_rt[0] * shard_grid_strides_rt[0] >= num_banks,
                "Number of shards must be greater than or equal to number of banks!");
        } else {
            static_assert(
                shard_grid_ct(TensorShape::shape, ShardShape::shape)[0] *
                        shard_grid_strides_ct(TensorShape::shape, ShardShape::shape)[0] >=
                    num_banks,
                "Number of shards must be greater than or equal to number of banks!");
        }
    }

    // Single constructor that handles all array cases
    constexpr DistributionSpec(const ShapeBase& tensor_shape_arr, const ShapeBase& shard_shape_arr = {}) :
        DistributionSpec(tensor_shape_arr, shard_shape_arr, select_constructor_tag()) {}

private:
    struct dynamic_tensor_dynamic_shard_tag {};
    struct dynamic_tensor_static_shard_tag {};
    struct static_tensor_dynamic_shard_tag {};

    // Function to select the appropriate tag based on static/dynamic properties
    static constexpr auto select_constructor_tag() {
        if constexpr (!TensorShape::is_static && !ShardShape::is_static) {
            return dynamic_tensor_dynamic_shard_tag{};
        } else if constexpr (!TensorShape::is_static && ShardShape::is_static) {
            return dynamic_tensor_static_shard_tag{};
        } else if constexpr (TensorShape::is_static && !ShardShape::is_static) {
            return static_tensor_dynamic_shard_tag{};
        } else {
            // This case is handled by the tuple constructor, so it shouldn't reach here
            static_assert(!is_static, "Static tensor and static shard should use the tuple constructor");
        }
    }

    // Specializations for each case
    constexpr DistributionSpec(
        const ShapeBase& tensor_shape_arr, const ShapeBase& shard_shape_arr, dynamic_tensor_dynamic_shard_tag) :
        tensor_shape_(tensor_shape_arr), shard_shape_(shard_shape_arr) {
        compute_shard_grid_and_strides_rt(tensor_shape_.shape, shard_shape_.shape);
    }

    constexpr DistributionSpec(const ShapeBase& tensor_shape_arr, const ShapeBase&, dynamic_tensor_static_shard_tag) :
        tensor_shape_(tensor_shape_arr) {
        compute_shard_grid_and_strides_rt(tensor_shape_.shape, ShardShape::shape);
    }

    constexpr DistributionSpec(const ShapeBase&, const ShapeBase& shard_shape_arr, static_tensor_dynamic_shard_tag) :
        shard_shape_(shard_shape_arr) {
        compute_shard_grid_and_strides_rt(TensorShape::shape, shard_shape_.shape);
    }

public:
    constexpr ShapeBase get_shard_grid() const {
        if constexpr (is_static) {
            return shard_grid_ct(TensorShape::shape, ShardShape::shape);
        } else {
            return shard_grid_rt;
        }
    }

    constexpr ShapeBase get_shard_grid_strides() const {
        if constexpr (is_static) {
            return shard_grid_strides_ct(TensorShape::shape, ShardShape::shape);
        } else {
            return shard_grid_strides_rt;
        }
    }

    constexpr ShapeBase get_tensor_shape() const {
        if constexpr (is_static) {
            return TensorShape::shape;
        } else {
            return tensor_shape_.shape;
        }
    }

    constexpr ShapeBase get_tensor_strides() const {
        if constexpr (is_static) {
            return TensorShape::strides;
        } else {
            return tensor_shape_.strides;
        }
    }

    constexpr size_t get_tensor_volume() const {
        if constexpr (is_static) {
            return TensorShape::volume;
        } else {
            return tensor_shape_.volume;
        }
    }

    constexpr ShapeBase get_shard_shape() const {
        if constexpr (is_static) {
            return ShardShape::shape;
        } else {
            return shard_shape_.shape;
        }
    }

    constexpr ShapeBase get_shard_strides() const {
        if constexpr (is_static) {
            return ShardShape::strides;
        } else {
            return shard_shape_.strides;
        }
    }

    constexpr size_t get_shard_volume() const {
        if constexpr (is_static) {
            return ShardShape::volume;
        } else {
            return shard_shape_.volume;
        }
    }

    // Compute shard grid and shard grid strides at compile time
    static constexpr ShapeBase shard_grid_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        ShapeBase shard_grid = {};
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid;
    }

    static constexpr ShapeBase shard_grid_strides_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        ShapeBase shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
    }

    void compute_shard_grid_and_strides_rt(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid_rt[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
            shard_grid_strides_rt[i] = stride;
            stride *= shard_grid_rt[i];
        }
        // Check that the number of shards is greater than or equal to the number of banks
        ASSERT(
            shard_grid_rt[0] * shard_grid_strides_rt[0] >= num_banks,
            "Number of shards must be greater than or equal to number of banks!");
    }

    std::conditional_t<is_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<is_static, std::monostate, ShapeBase> shard_grid_strides_rt{};

    TensorShape tensor_shape_;  // Tensor shape
    ShardShape shard_shape_;    // Shard shape

    static constexpr auto num_banks = BankCoords::num_banks;
    static constexpr auto packed_xy_coords = BankCoords::packed_xy_coords;
};

template <size_t BASE, size_t RANK, size_t NUM_BANKS>
struct DistributionSpecWrapper {
    using dspec = DistributionSpec<
        struct_sequence_wrapper_t<ShapeWrapper, BASE, RANK>,
        struct_sequence_wrapper_t<ShapeWrapper, BASE + RANK, RANK>,
        struct_sequence_wrapper_t<BankCoordWrapper, BASE + 2 * RANK, NUM_BANKS>>;
};

}  // namespace detail
