// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>
#include "helpers.h"
#include "array_wrapper.h"
#include "args_location.h"
#include <cstring>

// Forward declared from dataflow_api.h
static uint32_t get_common_arg_addr(int arg_idx);

namespace tensor_accessor {

/**
 * @brief Holds all the distribution specification information for a tensor: rank, number of banks, tensor shape, shard
 * shape, bank coordinates. Each of these can be static or dynamic.
 *
 * @tparam RankCT         Compile-time rank of the tensor. If 0, the rank is dynamic.
 * @tparam NumBanksCT     Compile-time number of banks. If 0, the number of banks is dynamic.
 * @tparam TensorShapeWrapper_  Wrapper for the tensor shape. Can be detail::ArrayStaticWrapperU32<...> for static
 * shapes or detail::ArrayDynamicWrapper for dynamic shapes.
 * @tparam ShardShapeWrapper_   Wrapper for the shard shape. Can be detail::ArrayStaticWrapperU32<...> for static shapes
 *                              or detail::ArrayDynamicWrapper for dynamic shapes.
 * @tparam BankCoordsWrapper_   Wrapper for the bank coordinates. Can be detail::ArrayStaticWrapperU16<...> for static
 * shapes or detail::ArrayDynamicWrapper for dynamic shapes.
 */
template <
    uint32_t RankCT = 0,
    uint32_t NumBanksCT = 0,
    typename TensorShapeWrapper = ArrayDynamicWrapper,
    typename ShardShapeWrapper = ArrayDynamicWrapper,
    typename BankCoordsWrapper = ArrayDynamicWrapper>
struct DistributionSpec {
    static constexpr bool has_static_rank = RankCT != 0;
    static constexpr bool has_static_num_banks = NumBanksCT != 0;
    static constexpr bool tensor_shape_static = has_static_rank && TensorShapeWrapper::is_static;
    static constexpr bool shard_shape_static = has_static_rank && ShardShapeWrapper::is_static;
    static constexpr bool bank_coords_static = has_static_num_banks && BankCoordsWrapper::is_static;
    static constexpr bool shapes_static = has_static_rank && tensor_shape_static && shard_shape_static;
    static constexpr bool is_static = shapes_static && bank_coords_static;

    static constexpr auto rank_ct = RankCT;
    static constexpr uint32_t num_banks_ct = NumBanksCT;

    using ShapeDynamic = detail::Span<uint32_t>;
    using BankCoordsDynamic = detail::Span<uint16_t>;
    using ShapeStatic = std::array<uint32_t, rank_ct>;
    using BankCoordsStatic = std::array<uint16_t, num_banks_ct>;

    using Shape = std::conditional_t<has_static_rank, ShapeStatic, ShapeDynamic>;
    using BankCoords = std::conditional_t<has_static_num_banks, BankCoordsStatic, BankCoordsDynamic>;

    // This constructor is only used for completely static DistributionSpec
    template <typename T = void, typename = std::enable_if_t<is_static, T>>
    constexpr DistributionSpec() {}

    template <typename TensorShape = Shape, typename ShardShape = Shape, typename BankCoords = BankCoords>
    constexpr DistributionSpec(
        TensorShape&& tensor_shape_arr, ShardShape&& shard_shape_arr = {}, BankCoords&& bank_coords_arr = {}) :
        tensor_shape_rt(std::forward<TensorShape>(tensor_shape_arr)),
        shard_shape_rt(std::forward<ShardShape>(shard_shape_arr)),
        bank_coords_rt(std::forward<BankCoords>(bank_coords_arr)) {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_rt.size();
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_rt.size();
        }
        if constexpr (!has_static_rank) {
            shard_grid_rt = Shape(shard_grid_rt_buf.value, rank());
            shard_grid_strides_rt = Shape(shard_grid_strides_rt_buf.value, rank());

            tensor_strides_rt = Shape(tensor_strides_rt_buf.value, rank());
            shard_strides_rt = Shape(shard_strides_rt_buf.value, rank());
        }
        if constexpr (!tensor_shape_static) {
            // If tensor shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(tensor_shape(), tensor_strides_rt, tensor_volume_rt);
        }
        if constexpr (!shard_shape_static) {
            // If shard shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(shard_shape(), shard_strides_rt, shard_volume_rt);
        }
        if constexpr (!shapes_static) {
            compute_shard_grid_and_strides_rt(tensor_shape(), shard_shape());
        }
    }

// Helper macro to avoid code duplication in getters
#define getter_helper(is_static, val_ct, val_rt) \
    if constexpr (is_static) {                   \
        return val_ct;                           \
    } else {                                     \
        return val_rt;                           \
    }

    // === Shape and Dimension Queries ===
    FORCE_INLINE constexpr uint32_t rank() const {getter_helper(has_static_rank, rank_ct, rank_rt)}

    FORCE_INLINE constexpr uint32_t num_banks() const {getter_helper(has_static_num_banks, num_banks_ct, num_banks_rt)}

    FORCE_INLINE constexpr const
        auto& tensor_shape() const {getter_helper(tensor_shape_static, TensorShapeWrapper::elements, tensor_shape_rt)}

    FORCE_INLINE constexpr const
        auto& shard_shape() const {getter_helper(shard_shape_static, ShardShapeWrapper::elements, shard_shape_rt)}

    // === Computed Properties ===
    FORCE_INLINE constexpr const
        auto& tensor_strides() const {getter_helper(tensor_shape_static, tensor_strides_ct, tensor_strides_rt)}

    FORCE_INLINE constexpr const
        auto& shard_strides() const {getter_helper(shard_shape_static, shard_strides_ct, shard_strides_rt)}

    FORCE_INLINE constexpr size_t
        tensor_volume() const {getter_helper(tensor_shape_static, tensor_volume_ct, tensor_volume_rt)}

    FORCE_INLINE constexpr size_t
        shard_volume() const {getter_helper(shard_shape_static, shard_volume_ct, shard_volume_rt)}

    // === Sharding Layout ===
    FORCE_INLINE constexpr const auto& shard_grid() const {getter_helper(shapes_static, shard_grid_ct, shard_grid_rt)}

    FORCE_INLINE constexpr const
        auto& shard_grid_strides() const {getter_helper(shapes_static, shard_grid_strides_ct, shard_grid_strides_rt)}

    FORCE_INLINE constexpr const auto& packed_xy_coords() const {
        getter_helper(bank_coords_static, BankCoordsWrapper::elements, bank_coords_rt)
    }

#undef getter_helper

private:
    static constexpr ShapeStatic precompute_shard_grid_ct(
        const ShapeStatic& tensor_shape, const ShapeStatic& shard_shape) {
        // If shapes are dynamic, we cannot compute shard grid at compile time
        if (!shapes_static) {
            return {};
        }
        ShapeStatic shard_grid = {};
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid;
    }

    static constexpr ShapeStatic precompute_shard_grid_strides_ct(
        const ShapeStatic& tensor_shape, const ShapeStatic& shard_shape) {
        ShapeStatic shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
    }

    static constexpr size_t precompute_volume_ct(const ShapeStatic& shape) {
        size_t volume = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            volume *= shape[i];
        }
        return volume;
    }

    static constexpr ShapeStatic precompute_strides_ct(const ShapeStatic& shape) {
        ShapeStatic strides = {};
        uint32_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    template <typename TensorShape, typename ShardShape>
    void compute_strides_volume_rt(const TensorShape& shape, ShardShape& strides, size_t& volume) const {
        uint32_t stride = 1;
        volume = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
            volume *= shape[i];
        }
    }

    template <typename TensorShape, typename ShardShape>
    void compute_shard_grid_and_strides_rt(const TensorShape& tensor_shape, const ShardShape& shard_shape) {
        uint32_t stride = 1;
        for (int i = rank() - 1; i >= 0; --i) {
            shard_grid_rt[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
            shard_grid_strides_rt[i] = stride;
            stride *= shard_grid_rt[i];
        }
        // Check that the number of shards is greater than or equal to the number of banks
        ASSERT(shard_grid_rt[0] * shard_grid_strides_rt[0] >= num_banks());
    }

    uint32_t rank_rt = 0;
    uint32_t num_banks_rt = 0;

    const Shape tensor_shape_rt = {};
    const Shape shard_shape_rt = {};
    const BankCoords bank_coords_rt = {};

    std::conditional_t<shapes_static, std::monostate, Shape> shard_grid_rt{};
    std::conditional_t<shapes_static, std::monostate, Shape> shard_grid_strides_rt{};

    // Buffers to wrap around span in case of dynamic rank
    [[no_unique_address]] mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> shard_grid_rt_buf;
    [[no_unique_address]] mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]>
        shard_grid_strides_rt_buf;

    static constexpr ShapeStatic shard_grid_ct =
        precompute_shard_grid_ct(TensorShapeWrapper::elements, ShardShapeWrapper::elements);
    static constexpr ShapeStatic shard_grid_strides_ct =
        precompute_shard_grid_strides_ct(TensorShapeWrapper::elements, ShardShapeWrapper::elements);

    // Buffers to wrap around span in case of dynamic rank
    [[no_unique_address]] mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> tensor_strides_rt_buf;
    [[no_unique_address]] mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> shard_strides_rt_buf;
    Shape tensor_strides_rt = {};
    Shape shard_strides_rt = {};
    static constexpr ShapeStatic tensor_strides_ct = precompute_strides_ct(TensorShapeWrapper::elements);
    static constexpr ShapeStatic shard_strides_ct = precompute_strides_ct(ShardShapeWrapper::elements);

    size_t tensor_volume_rt = 0;
    size_t shard_volume_rt = 0;
    static constexpr size_t tensor_volume_ct = precompute_volume_ct(TensorShapeWrapper::elements);
    static constexpr size_t shard_volume_ct = precompute_volume_ct(ShardShapeWrapper::elements);
};

/**
 * @brief Build a DistributionSpec from the provided arguments. This function allows for both static and dynamic rank,
 * number of banks, tensor shape, shard shape, and bank coordinates.
 *
 * @tparam RankCT               Compile-time rank of the tensor. If 0, the rank is dynamic.
 * @tparam NumBanksCT           Compile-time number of banks. If 0, the number of banks is dynamic.
 * @tparam TensorShapeWrapper_  ArrayDynamicWrapper or ArrayStaticWrapperU32 for the tensor shape.
 * @tparam ShardShapeWrapper_   ArrayDynamicWrapper or ArrayStaticWrapperU32 for the shard shape.
 * @tparam BankCoordsWrapper_   ArrayDynamicWrapper or ArrayStaticWrapperU16 for the bank coordinates.
 * @param rank_rt               Runtime rank of the tensor. Used if RankCT is 0.
 * @param num_banks_rt          Runtime number of banks. Used if NumBanksCT is 0.
 * @param tensor_shape_ptr      Pointer to the tensor shape array. Used if TensorShapeWrapper_ is dynamic.
 * @param shard_shape_ptr       Pointer to the shard shape array. Used if ShardShapeWrapper_ is dynamic.
 * @param bank_coords_ptr       Pointer to the bank coordinates array. Used if BankCoordsWrapper_ is dynamic.
 * @return auto                 DistributionSpec instance built from the provided arguments.
 */
template <
    uint32_t RankCT = 0,
    uint32_t NumBanksCT = 0,
    typename TensorShapeWrapper_ = ArrayDynamicWrapper,
    typename ShardShapeWrapper_ = ArrayDynamicWrapper,
    typename BankCoordsWrapper_ = ArrayDynamicWrapper>
auto make_dspec(
    uint32_t rank_rt = 0,
    uint32_t num_banks_rt = 0,
    uint32_t* tensor_shape_ptr = nullptr,
    uint32_t* shard_shape_ptr = nullptr,
    uint16_t* bank_coords_ptr = nullptr) {
    using DSpec = DistributionSpec<RankCT, NumBanksCT, TensorShapeWrapper_, ShardShapeWrapper_, BankCoordsWrapper_>;
    constexpr bool RankStatic = RankCT != 0;
    constexpr bool NumBanksStatic = NumBanksCT != 0;
    constexpr bool TensorShapeDynamic = !TensorShapeWrapper_::is_static;
    constexpr bool ShardShapeDynamic = !ShardShapeWrapper_::is_static;
    constexpr bool BankCoordsDynamic = !BankCoordsWrapper_::is_static;

    uint32_t rank = RankStatic ? RankCT : rank_rt;
    uint32_t num_banks = NumBanksStatic ? NumBanksCT : num_banks_rt;
    // Shape = std::array<uint32_t, RankCT> if static, otherwise Span<uint32_t>
    typename DSpec::Shape tensor_shape_array;
    typename DSpec::Shape shard_shape_array;
    // BankCoords = std::array<uint32_t, NumBanksCT> if static, otherwise Span<uint32_t>
    typename DSpec::BankCoords bank_coord_array;

    auto span_from_pointer = []<typename T>(auto& arr, T* ptr, size_t size) { arr = detail::Span<T>(ptr, size); };

    auto array_from_pointer = []<typename T>(auto& arr, T* ptr, size_t size) {
        std::memcpy(arr.data(), ptr, sizeof(T) * size);
        return arr;
    };

    if constexpr (RankStatic) {
        if constexpr (TensorShapeDynamic) {
            ASSERT(tensor_shape_ptr != nullptr);
            array_from_pointer(tensor_shape_array, tensor_shape_ptr, RankCT);
        }
        if constexpr (ShardShapeDynamic) {
            ASSERT(shard_shape_ptr != nullptr);
            array_from_pointer(shard_shape_array, shard_shape_ptr, RankCT);
        }
    } else {
        if constexpr (TensorShapeDynamic) {
            ASSERT(tensor_shape_ptr != nullptr);
            span_from_pointer(tensor_shape_array, tensor_shape_ptr, rank_rt);
        }
        if constexpr (ShardShapeDynamic) {
            ASSERT(shard_shape_ptr != nullptr);
            span_from_pointer(shard_shape_array, shard_shape_ptr, rank_rt);
        }
    }

    if constexpr (BankCoordsDynamic) {
        ASSERT(bank_coords_ptr != nullptr);
        if constexpr (NumBanksStatic) {
            array_from_pointer(bank_coord_array, bank_coords_ptr, NumBanksCT);
        } else {
            span_from_pointer(bank_coord_array, bank_coords_ptr, num_banks_rt);
        }
    }

    // Verify that shapes are non-zero
    for (size_t i = 0; i < rank; ++i) {
        if constexpr (TensorShapeDynamic) {
            ASSERT(tensor_shape_array[i] > 0);
        }
        if constexpr (ShardShapeDynamic) {
            ASSERT(shard_shape_array[i] > 0);
        }
    }
    return DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
}

/**
 * @brief Helper function to build a DistributionSpec from CTA and CRTA. Parses tensor shape, shard shape,
 * bank coordinates if needed, and passes to DSpec constructor.
 *
 * @tparam ArgsOffsets Structure containing offsets for CTA/CRTA.
 * @return auto DistributionSpec instance built from provided argumnets.
 */
template <typename ArgsOffsets>
auto make_dspec_from_args(const ArgsOffsets& args_offsets) {
    // Dispatch to the appropriate ShapeWrapper and BankCoordsWrapper types based on the "staticness"
    using TensorShapeType = typename ArrayWrapperTypeSelectorU32<
        !ArgsOffsets::tensor_shape_is_crta,
        ArgsOffsets::TensorShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using ShardShapeType = typename ArrayWrapperTypeSelectorU32<
        !ArgsOffsets::shard_shape_is_crta,
        ArgsOffsets::ShardShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using BankCoordsType = typename ArrayWrapperTypeSelectorPackedU16<
        !ArgsOffsets::bank_coords_is_crta,
        ArgsOffsets::BankCoordsCTAOffset,
        ArgsOffsets::NumBanksCT>::type;

    auto rank = args_offsets.get_rank();
    auto num_banks = args_offsets.get_num_banks();

    ASSERT(rank > 0);
    ASSERT(num_banks > 0);

    static_assert(
        !ArgsOffsets::rank_is_crta or ArgsOffsets::tensor_shape_is_crta,
        "Tensor shape must be CRTA if rank is not known at compile time!");
    static_assert(
        !ArgsOffsets::rank_is_crta or ArgsOffsets::shard_shape_is_crta,
        "Shard shape must be CRTA if rank is not known at compile time!");
    static_assert(
        !ArgsOffsets::num_banks_is_crta or ArgsOffsets::bank_coords_is_crta,
        "Bank coords must be CRTA if num_banks is not known at compile time!");

    return make_dspec<ArgsOffsets::RankCT, ArgsOffsets::NumBanksCT, TensorShapeType, ShardShapeType, BankCoordsType>(
        rank,
        num_banks,
        ArgsOffsets::tensor_shape_is_crta ? (uint32_t*)get_common_arg_addr(args_offsets.tensor_shape_crta_offset())
                                          : nullptr,
        ArgsOffsets::shard_shape_is_crta ? (uint32_t*)get_common_arg_addr(args_offsets.shard_shape_crta_offset())
                                         : nullptr,
        ArgsOffsets::bank_coords_is_crta ? (uint16_t*)get_common_arg_addr(args_offsets.bank_coords_crta_offset())
                                         : nullptr);
}

}  // namespace tensor_accessor
