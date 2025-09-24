// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>
#include "helpers.h"
#include "array_wrapper.h"
#include "compile_time_args.h"
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
    typename BankCoordsWrapper = ArrayDynamicWrapper,
    bool IsInterleaved = false,
    bool IsDram = false>
struct DistributionSpec {
    static constexpr bool has_static_rank = RankCT != 0;
    static constexpr bool has_static_num_banks = NumBanksCT != 0;
    static constexpr bool tensor_shape_static = has_static_rank && TensorShapeWrapper::is_static;
    static constexpr bool shard_shape_static = has_static_rank && ShardShapeWrapper::is_static;
    static constexpr bool bank_coords_static = has_static_num_banks && BankCoordsWrapper::is_static;
    static constexpr bool shapes_static = has_static_rank && tensor_shape_static && shard_shape_static;
    static constexpr bool is_static = shapes_static && bank_coords_static;
    static constexpr bool is_interleaved = IsInterleaved;
    static constexpr bool is_dram = IsDram;

    static constexpr auto rank_ct = RankCT;
    static constexpr auto num_banks_ct = NumBanksCT;

    using ShapeDynamic = detail::Span<uint32_t>;
    using BankCoordsDynamic = detail::Span<uint16_t>;
    using ShapeStatic = std::array<uint32_t, rank_ct>;
    using BankCoordsStatic = std::array<uint16_t, num_banks_ct>;

    using Shape = std::conditional_t<has_static_rank, ShapeStatic, ShapeDynamic>;
    using BankCoords = std::conditional_t<has_static_num_banks, BankCoordsStatic, BankCoordsDynamic>;

    // This constructor is only used for completely static DistributionSpec
    template <typename T = void, typename = std::enable_if_t<is_static, T>>
    constexpr DistributionSpec() {}

    // Copy constructor
    DistributionSpec(const DistributionSpec& other) :
        rank_rt(other.rank_rt),
        num_banks_rt(other.num_banks_rt),
        tensor_shape_rt(other.tensor_shape_rt),
        shard_shape_rt(other.shard_shape_rt),
        bank_coords_rt(other.bank_coords_rt),
        shard_grid_rt(other.shard_grid_rt),
        shard_grid_strides_rt(other.shard_grid_strides_rt),
        tensor_strides_rt(other.tensor_strides_rt),
        shard_strides_rt(other.shard_strides_rt),
        tensor_volume_rt(other.tensor_volume_rt),
        shard_volume_rt(other.shard_volume_rt) {
        // Copy the buffer arrays and fix self-references
        if constexpr (!has_static_rank) {
            std::memcpy(shard_grid_rt_buf.value, other.shard_grid_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(
                shard_grid_strides_rt_buf.value, other.shard_grid_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(tensor_strides_rt_buf.value, other.tensor_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(shard_strides_rt_buf.value, other.shard_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            update_spans_pointers();
        }
    }

    // Copy assignment operator
    DistributionSpec& operator=(const DistributionSpec& other) {
        if (this != &other) {
            DistributionSpec tmp(other);
            swap(tmp);
        }
        return *this;
    }

    // Move constructor
    DistributionSpec(DistributionSpec&& other) noexcept :
        rank_rt(other.rank_rt),
        num_banks_rt(other.num_banks_rt),
        tensor_shape_rt(std::move(other.tensor_shape_rt)),
        shard_shape_rt(std::move(other.shard_shape_rt)),
        bank_coords_rt(std::move(other.bank_coords_rt)),
        shard_grid_rt(std::move(other.shard_grid_rt)),
        shard_grid_strides_rt(std::move(other.shard_grid_strides_rt)),
        tensor_strides_rt(std::move(other.tensor_strides_rt)),
        shard_strides_rt(std::move(other.shard_strides_rt)),
        tensor_volume_rt(other.tensor_volume_rt),
        shard_volume_rt(other.shard_volume_rt) {
        // Copy the buffer arrays and fix self-references
        if constexpr (!has_static_rank) {
            std::memcpy(shard_grid_rt_buf.value, other.shard_grid_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(
                shard_grid_strides_rt_buf.value, other.shard_grid_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(tensor_strides_rt_buf.value, other.tensor_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            std::memcpy(shard_strides_rt_buf.value, other.shard_strides_rt_buf.value, sizeof(uint32_t) * rank_rt);
            update_spans_pointers();
        }
    }

    // Move assignment operator
    DistributionSpec& operator=(DistributionSpec&& other) noexcept {
        if (this != &other) {
            DistributionSpec tmp(std::move(other));
            swap(tmp);
        }
        return *this;
    }

    template <
        typename TensorShape = Shape,
        typename ShardShape = Shape,
        typename BankCoords = BankCoords,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<TensorShape>, DistributionSpec>>>
    constexpr DistributionSpec(
        TensorShape&& tensor_shape_arr, ShardShape&& shard_shape_arr = {}, BankCoords&& bank_coords_arr = {}) :
        tensor_shape_rt(std::forward<TensorShape>(tensor_shape_arr)),
        shard_shape_rt(std::forward<ShardShape>(shard_shape_arr)),
        bank_coords_rt(std::forward<BankCoords>(bank_coords_arr)) {
        init_runtime_values();
    }

    /**
     * @brief Build a DistributionSpec from the provided arguments. This function allows for both static and dynamic
     * rank, number of banks, tensor shape, shard shape, and bank coordinates.
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
     */
    constexpr DistributionSpec(
        uint32_t rank_rt_param = 0,
        uint32_t num_banks_rt_param = 0,
        uint32_t* tensor_shape_ptr = nullptr,
        uint32_t* shard_shape_ptr = nullptr,
        uint16_t* bank_coords_ptr = nullptr) :
        tensor_shape_rt(init_tensor_shape(tensor_shape_ptr, rank_rt_param)),
        shard_shape_rt(init_shard_shape(shard_shape_ptr, rank_rt_param)),
        bank_coords_rt(init_bank_coords(bank_coords_ptr, num_banks_rt_param)) {
        uint32_t rank = has_static_rank ? RankCT : rank_rt_param;

        if constexpr (!tensor_shape_static) {
            ASSERT(rank == 0 || tensor_shape_ptr != nullptr);
        }
        if constexpr (!shard_shape_static) {
            ASSERT(rank == 0 || shard_shape_ptr != nullptr);
        }

        if constexpr (!bank_coords_static) {
            if constexpr (has_static_num_banks) {
                ASSERT(NumBanksCT == 0 || bank_coords_ptr != nullptr);
            } else {
                ASSERT(num_banks_rt == 0 || bank_coords_ptr != nullptr);
            }
        }

        // Verify that shapes are non-zero
        for (size_t i = 0; i < rank; ++i) {
            if constexpr (!tensor_shape_static) {
                ASSERT(tensor_shape_rt[i] > 0);
            }
            if constexpr (!shard_shape_static) {
                ASSERT(shard_shape_rt[i] > 0);
            }
        }

        init_runtime_values();
    }
    /**
     * @brief Build a DistributionSpec from the provided arguments. This function allows for both static and dynamic
     * rank, number of banks, tensor shape, shard shape, and bank coordinates.
     *
     * @param args                  Arguments to build the DistributionSpec from.
     */
    template <typename Args>
    constexpr DistributionSpec(const Args& args) :
        DistributionSpec(
            args.get_rank(),
            args.get_num_banks(),
            Args::tensor_shape_is_crta ? (uint32_t*)get_common_arg_addr(args.tensor_shape_crta_offset()) : nullptr,
            Args::shard_shape_is_crta ? (uint32_t*)get_common_arg_addr(args.shard_shape_crta_offset()) : nullptr,
            Args::bank_coords_is_crta ? (uint16_t*)get_common_arg_addr(args.bank_coords_crta_offset()) : nullptr) {
        static_assert(
            !Args::rank_is_crta or Args::tensor_shape_is_crta,
            "Tensor shape must be CRTA if rank is not known at compile time!");
        static_assert(
            !Args::rank_is_crta or Args::shard_shape_is_crta,
            "Shard shape must be CRTA if rank is not known at compile time!");
        static_assert(
            !Args::num_banks_is_crta or Args::bank_coords_is_crta,
            "Bank coords must be CRTA if num_banks is not known at compile time!");
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
    // Unified initialization helper for arrays
    // This allows to initialize arrays in initializer list, which allows to keep fields const, which allows compiler to
    // optimize the code better.
    template <
        typename ResultType,
        typename ElementType,
        typename WrapperType,
        bool is_static,
        bool has_static_size,
        uint32_t static_size>
    static constexpr ResultType init_array(ElementType* ptr, uint32_t size_rt_param) {
        if constexpr (is_static) {
            return WrapperType::elements;
        } else if constexpr (has_static_size) {
            ResultType result{};
            if (ptr) {
                std::memcpy(result.data(), ptr, sizeof(ElementType) * static_size);
            }
            return result;
        } else {
            return ptr ? ResultType(ptr, size_rt_param) : ResultType{};
        }
    }

    static constexpr Shape init_tensor_shape(uint32_t* ptr, uint32_t rank_rt_param) {
        return init_array<Shape, uint32_t, TensorShapeWrapper, tensor_shape_static, has_static_rank, RankCT>(
            ptr, rank_rt_param);
    }

    static constexpr Shape init_shard_shape(uint32_t* ptr, uint32_t rank_rt_param) {
        return init_array<Shape, uint32_t, ShardShapeWrapper, shard_shape_static, has_static_rank, RankCT>(
            ptr, rank_rt_param);
    }

    static constexpr BankCoords init_bank_coords(uint16_t* ptr, uint32_t num_banks_rt_param) {
        return init_array<
            BankCoords,
            uint16_t,
            BankCoordsWrapper,
            bank_coords_static,
            has_static_num_banks,
            NumBanksCT>(ptr, num_banks_rt_param);
    }

    void swap(DistributionSpec& other) noexcept {
        std::swap(rank_rt, other.rank_rt);
        std::swap(num_banks_rt, other.num_banks_rt);
        std::swap(tensor_shape_rt, other.tensor_shape_rt);
        std::swap(shard_shape_rt, other.shard_shape_rt);
        std::swap(bank_coords_rt, other.bank_coords_rt);
        std::swap(shard_grid_rt, other.shard_grid_rt);
        std::swap(shard_grid_strides_rt, other.shard_grid_strides_rt);
        std::swap(tensor_strides_rt, other.tensor_strides_rt);
        std::swap(shard_strides_rt, other.shard_strides_rt);
        std::swap(tensor_volume_rt, other.tensor_volume_rt);
        std::swap(shard_volume_rt, other.shard_volume_rt);

        if constexpr (!has_static_rank) {
            std::swap_ranges(shard_grid_rt_buf.value, shard_grid_rt_buf.value + rank_rt, other.shard_grid_rt_buf.value);
            std::swap_ranges(
                shard_grid_strides_rt_buf.value,
                shard_grid_strides_rt_buf.value + rank_rt,
                other.shard_grid_strides_rt_buf.value);
            std::swap_ranges(
                tensor_strides_rt_buf.value, tensor_strides_rt_buf.value + rank_rt, other.tensor_strides_rt_buf.value);
            std::swap_ranges(
                shard_strides_rt_buf.value, shard_strides_rt_buf.value + rank_rt, other.shard_strides_rt_buf.value);
            update_spans_pointers();
            other.update_spans_pointers();
        }
    }

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
    }

    constexpr void update_spans_pointers() {
        if constexpr (!has_static_rank) {
            shard_grid_rt = Shape(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = Shape(shard_grid_strides_rt_buf.value, rank_rt);
            tensor_strides_rt = Shape(tensor_strides_rt_buf.value, rank_rt);
            shard_strides_rt = Shape(shard_strides_rt_buf.value, rank_rt);
        }
    }

    constexpr void init_runtime_values() {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_rt.size();
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_rt.size();
        }
        update_spans_pointers();
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

    uint32_t rank_rt = 0;
    uint32_t num_banks_rt = 0;

    Shape tensor_shape_rt = {};
    Shape shard_shape_rt = {};
    BankCoords bank_coords_rt = {};

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

template <bool IsDram>
auto make_interleaved_dspec() {
    return DistributionSpec<
        /*RankCT=*/0,
        /*NumBanksCT=*/0,
        /*TensorShapeWrapper=*/ArrayStaticWrapper<uint32_t>,
        /*ShardShapeWrapper=*/ArrayStaticWrapper<uint32_t>,
        /*BankCoordsWrapper=*/ArrayStaticWrapper<uint16_t>,
        /*IsInterleaved=*/true,
        IsDram>(
        /* rank_rt */ 0,
        /* num_banks_rt */ 0,
        /* tensor_shape_ptr */ nullptr,
        /* shard_shape_ptr */ nullptr,
        /* bank_coords_ptr */ nullptr);
}

}  // namespace tensor_accessor
