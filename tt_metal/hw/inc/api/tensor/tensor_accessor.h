// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "api/tensor/tensor_accessor_args.h"
#include "internal/tensor/array_wrapper.h"
#include "internal/tensor/dspec.h"
#include "internal/tensor/helpers.h"
#include "api/tensor/shard_pages_address_iterator.h"
#include "api/tensor/pages_address_iterator.h"
#include "api/compile_time_args.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "internal/dataflow/dataflow_api_addrgen.h"
#endif

// NOLINTBEGIN(misc-unused-parameters)

// Forward declared from dataflow_api.h
template <typename T>
T get_arg_val(int arg_idx);

namespace tensor_accessor {
// This helper gets proper additional offset from interleaved_addr_gen::get_bank_offset +
//      Adds proper xy coordinates for NOC address
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc) {
    // TODO: Should interleaved_addr_gen:: functions moved into common helper?
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(bank_id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(bank_id, bank_offset_index);
    uint32_t bank_offset = interleaved_addr_gen::get_bank_offset<true>(bank_index);
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, bank_offset);
    return noc_addr;
}
#endif
}  // namespace tensor_accessor

/**
 * @brief Accessor that encapsulates the logic for accessing tensors pages.
 *
 * The TensorAccessor provides efficient access to pages in a tensor by:
 * 1. Computing which bank contains a given page
 * 2. Calculating the offset within that bank
 * 3. Providing NOC address computation and async operations
 *
 * @tparam DSpec        DistributionSpec type.
 */
template <typename DSpecT>
struct TensorAccessor {
    using DSpec = DSpecT;
    static constexpr bool is_dram = DSpec::is_dram;

private:
    // DSpec can be static or dynamic, so we use a conditional instance
    using StaticDspec = tensor_accessor::detail::ConditionalStaticInstance<DSpec, DSpec::is_static>;
    [[no_unique_address]] tensor_accessor::detail::ConditionalField<!DSpec::is_static, DSpec> dspec_instance;

    [[no_unique_address]] mutable tensor_accessor::detail::
        ConditionalField<!DSpec::has_static_rank, uint32_t[tensor_accessor::MAX_RANK]> _page_coord;

public:
    template <typename DSpec_ = DSpec, std::enable_if_t<std::is_same_v<std::decay_t<DSpec_>, DSpec>, int> = 0>
    constexpr explicit TensorAccessor(
        DSpec_&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        dspec_instance(std::forward<DSpec_>(dspec)), bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <typename DSpec_ = DSpec, std::enable_if_t<DSpec_::is_static, int> = 0>
    TensorAccessor(const size_t bank_base_address_in = 0, uint32_t page_size_in = 0) :
        bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    TensorAccessor(
        const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args,
        const size_t bank_base_address_in,
        const uint32_t page_size_in = 0) :
        dspec_instance(args), bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    constexpr const auto& dspec() const {
        if constexpr (DSpec::is_static) {
            return StaticDspec::instance;
        } else {
            return dspec_instance.value;
        }
    }

    constexpr auto& dspec() {
        if constexpr (DSpec::is_static) {
            return StaticDspec::instance;
        } else {
            return dspec_instance.value;
        }
    }

    // NOC APIs
    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr(get_bank_and_offset(page_id), offset, noc);
    }

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
    FORCE_INLINE std::uint64_t get_noc_addr(
        const ArrType page_coord, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr(get_bank_and_offset(page_coord), offset, noc);
    }

    // Shard NOC APIs
    FORCE_INLINE
    std::uint64_t get_shard_noc_addr(
        const uint32_t shard_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        PageMapping page_mapping{
            .bank_id = shard_id % dspec().num_banks(),
            .bank_page_offset = shard_id / dspec().num_banks() * dspec().shard_volume(),
        };
        return get_noc_addr(page_mapping, offset, noc);
    }

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
    FORCE_INLINE std::uint64_t get_shard_noc_addr(
        const ArrType shard_coord, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t shard_id = 0;
        for (uint32_t i = 0; i < dspec().rank(); ++i) {
            // Check that shard_coord is within bounds
            ASSERT(shard_coord[i] < dspec().shard_shape()[i]);
            shard_id *= dspec().shard_grid_strides()[i];
        }
        return get_shard_noc_addr(shard_id, offset, noc);
    }

    // Helpers
    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };

    PageMapping get_bank_and_offset(uint32_t page_id) const {
        // Check that page_id is within bounds
        ASSERT(page_id < dspec().tensor_volume());
        if (dspec().rank() >= 4) {
            return get_bank_and_offset_from_page_id(page_id);
        }

        // Calculate the page coordinate in the tensor
        typename DSpec::Shape page_coord;
        if constexpr (!DSpec::has_static_rank) {
            // If rank is not known at compile time, we need to use the _page_coord buffer for span
            page_coord = typename DSpec::Shape(_page_coord.value, dspec().rank());
        }
        for (int i = dspec().rank() - 1; i >= 0; --i) {
            page_coord[i] = page_id % dspec().tensor_shape()[i];
            page_id /= dspec().tensor_shape()[i];
        }
        return get_bank_and_offset(page_coord);
    }

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
    PageMapping get_bank_and_offset(const ArrType page_coord) const {
        // Flattened shard id is used to compute the bank id and shard id within a bank
        // - First, get the shard coordinate with page_coord[i] / dspec.shard_shape[i]
        // - Then, multiply by the shard grid strides and accumulate
        // - Repeat for all dims
        // Page offset within shard refers to the offset within the shard the page belongs to
        // - First, get the page coordinate within the shard with page_coord[i] % dspec.shard_shape[i]
        // - Then, multiple by the shard strides and accumulate
        // - Repeat for all dims
        // Final page offset within the bank is simply: bank_shard_id * shard_volume + page_offset_within_shard

        size_t flattened_shard_id = 0;
        size_t page_offset_within_shard = 0;
        for (size_t i = 0; i < dspec().rank(); ++i) {
            // Check that page_coord is within bounds
            ASSERT(page_coord[i] < dspec().tensor_shape()[i]);
            flattened_shard_id += (page_coord[i] / dspec().shard_shape()[i]) * dspec().shard_grid_strides()[i];
            page_offset_within_shard += (page_coord[i] % dspec().shard_shape()[i]) * dspec().shard_strides()[i];
        }

        // NOTE: This assumes shards are round-robin assigned across banks
        uint32_t bank_id = flattened_shard_id % dspec().num_banks();
        uint32_t bank_shard_id = flattened_shard_id / dspec().num_banks();

        uint32_t bank_page_offset = (bank_shard_id * dspec().shard_volume()) + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }

    // Locality APIs
    FORCE_INLINE
    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y, uint8_t noc = noc_index) const {
        return virtual_x == my_x[noc] && virtual_y == my_y[noc];
    }

    FORCE_INLINE
    bool is_local_addr(const uint64_t noc_addr, uint8_t noc = noc_index) const {
        uint32_t x = NOC_UNICAST_ADDR_X(noc_addr);
        uint32_t y = NOC_UNICAST_ADDR_Y(noc_addr);
        return is_local_bank(x, y, noc);
    }

    FORCE_INLINE
    bool is_local_page(const uint32_t page_id, uint8_t noc = noc_index) const {
        auto page_mapping = get_bank_and_offset(page_id);
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[page_mapping.bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[page_mapping.bank_id]);
        return is_local_bank(bank_x, bank_y, noc);
    }

    FORCE_INLINE
    bool is_local_shard(const uint32_t shard_id, uint8_t noc = noc_index) const {
        uint32_t bank_id = shard_id % dspec().num_banks();

        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[bank_id]);
        return is_local_bank(bank_x, bank_y, noc);
    }

    // Returns a proxy for shard pages iterator
    tensor_accessor::ShardPages<TensorAccessor> shard_pages(
        uint32_t shard_id,
        uint32_t start_page_offset = 0,
        uint32_t end_page_offset = 0,
        uint8_t noc = noc_index) const {
        static_assert(DSpec::has_static_rank, "ShardPages is only supported for static rank");
        uint32_t actual_end_page_offset = (end_page_offset == 0) ? dspec().shard_volume() : end_page_offset;
        return tensor_accessor::ShardPages<TensorAccessor>(
            *this, shard_id, start_page_offset, actual_end_page_offset, noc);
    }

    // Returns a proxy for pages iterator (iterates over all pages in the tensor)
    tensor_accessor::Pages<TensorAccessor> pages(
        uint32_t start_page_id = 0, uint32_t end_page_id = 0, uint8_t noc = noc_index) const {
        uint32_t actual_end_page_id = (end_page_id == 0) ? dspec().tensor_volume() : end_page_id;
        return tensor_accessor::Pages<TensorAccessor>(*this, start_page_id, actual_end_page_id, noc);
    }

private:
    // NOC APIs
    FORCE_INLINE
    std::uint64_t get_noc_addr(
        const PageMapping page_mapping, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[page_mapping.bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[page_mapping.bank_id]);
        auto bank_start = DSpec::is_dram ? tensor_accessor::get_dram_bank_base_offset(bank_x, noc)
                                         : NOC_XY_ADDR(DYNAMIC_NOC_X(noc, bank_x), DYNAMIC_NOC_Y(noc, bank_y), 0);
        return bank_start + bank_base_address + (page_mapping.bank_page_offset * page_size) + offset;
    }

    PageMapping get_bank_and_offset_from_page_id(uint32_t page_id) const {
        size_t flattened_shard_id = 0;
        size_t page_offset_within_shard = 0;
        for (int i = dspec().rank() - 1; i >= 0; --i) {
            // Check that page_coord is within bounds
            uint32_t page_coord = page_id % dspec().tensor_shape()[i];
            ASSERT(page_coord < dspec().tensor_shape()[i]);
            page_id /= dspec().tensor_shape()[i];
            flattened_shard_id += (page_coord / dspec().shard_shape()[i]) * dspec().shard_grid_strides()[i];
            page_offset_within_shard += (page_coord % dspec().shard_shape()[i]) * dspec().shard_strides()[i];
        }

        // NOTE: This assumes shards are round-robin assigned across banks
        size_t bank_id = flattened_shard_id % dspec().num_banks();
        size_t bank_shard_id = flattened_shard_id / dspec().num_banks();

        size_t bank_page_offset = (bank_shard_id * dspec().shard_volume()) + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }

    FORCE_INLINE
    uint16_t get_bank_x(uint16_t packed_xy_coord) const { return (packed_xy_coord >> 8) & 0xFF; }

    FORCE_INLINE
    uint16_t get_bank_y(uint16_t packed_xy_coord) const { return packed_xy_coord & 0xFF; }

public:
    const size_t bank_base_address = 0;
    const uint32_t page_size = 0;

    friend class tensor_accessor::ShardPagesAddressIterator<TensorAccessor>;
    friend class tensor_accessor::PagesAddressIteratorSharded<TensorAccessor>;
    friend class tensor_accessor::PagesAddressIteratorInterleaved<TensorAccessor>;
};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
template <
    uint32_t RankCT,
    uint32_t NumBanksCT,
    typename TensorShapeWrapper,
    typename ShardShapeWrapper,
    typename BankCoordsWrapper,
    bool IsDram>
struct TensorAccessor<tensor_accessor::DistributionSpec<
    RankCT,
    NumBanksCT,
    TensorShapeWrapper,
    ShardShapeWrapper,
    BankCoordsWrapper,
    /* IsInterleaved */ true,
    IsDram>> : public InterleavedAddrGen<IsDram> {
    using DSpec = tensor_accessor::DistributionSpec<
        RankCT,
        NumBanksCT,
        TensorShapeWrapper,
        ShardShapeWrapper,
        BankCoordsWrapper,
        /* IsInterleaved */ true,
        IsDram>;

    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    TensorAccessor(
        const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args,
        const size_t bank_base_address_in,
        const uint32_t page_size_in = 0) :
        InterleavedAddrGen<IsDram>({.bank_base_address = bank_base_address_in, .page_size = page_size_in}) {}

    template <typename DSpec_ = DSpec, std::enable_if_t<std::is_same_v<std::decay_t<DSpec_>, DSpec>, int> = 0>
    constexpr explicit TensorAccessor(
        DSpec_&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        InterleavedAddrGen<IsDram>({.bank_base_address = bank_base_address_in, .page_size = page_size_in}) {}

    // Locality APIs
    FORCE_INLINE
    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y, uint8_t noc = noc_index) const {
        static_assert(
            tensor_accessor::detail::always_false_v<TensorAccessor>,
            "TensorAccessor::is_local_bank is not supported by the interleaved tensor accessor");
        return false;
    }

    FORCE_INLINE
    bool is_local_addr(const uint64_t noc_addr, uint8_t noc = noc_index) const {
        static_assert(
            tensor_accessor::detail::always_false_v<TensorAccessor>,
            "TensorAccessor::is_local_addr is not supported by the interleaved tensor accessor");
        return false;
    }

    FORCE_INLINE
    bool is_local_page(const uint32_t page_id, uint8_t noc = noc_index) const {
        static_assert(
            tensor_accessor::detail::always_false_v<TensorAccessor>,
            "TensorAccessor::is_local_page is not supported by the interleaved tensor accessor");
        return false;
    }

    FORCE_INLINE
    bool is_local_shard(const uint32_t shard_id, uint8_t noc = noc_index) const {
        static_assert(
            tensor_accessor::detail::always_false_v<TensorAccessor>,
            "TensorAccessor::is_local_shard is not supported by the interleaved tensor accessor");
        return false;
    }

    // Returns a proxy for shard pages iterator
    tensor_accessor::ShardPages<TensorAccessor> shard_pages(
        uint32_t shard_id,
        uint32_t start_page_offset = 0,
        uint32_t end_page_offset = 0,
        uint8_t noc = noc_index) const {
        static_assert(
            tensor_accessor::detail::always_false_v<TensorAccessor>,
            "TensorAccessor::shard_pages is not supported by the interleaved tensor accessor");
        return {};
    }

    // Returns a proxy for pages iterator (iterates over all pages in the tensor)
    // For interleaved tensors, start_page_id and end_page_id must be provided since the accessor doesn't know tensor
    // volume
    tensor_accessor::Pages<TensorAccessor> pages(
        uint32_t start_page_id, uint32_t end_page_id, uint8_t noc = noc_index) const {
        return tensor_accessor::Pages<TensorAccessor>(*this, start_page_id, end_page_id, noc);
    }
};
#endif

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
TensorAccessor(const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args, size_t, uint32_t)
    -> TensorAccessor<tensor_accessor::DistributionSpec<
        /* RankCT */ TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::RankCT,
        /* NumBanksCT */ TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::NumBanksCT,
        /* TensorShapeWrapper */
        typename tensor_accessor::ArrayWrapperTypeSelectorU32<
            !TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::tensor_shape_is_crta,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::TensorShapeCTAOffset,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::RankCT>::type,
        /* ShardShapeWrapper */
        typename tensor_accessor::ArrayWrapperTypeSelectorU32<
            !TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::shard_shape_is_crta,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::ShardShapeCTAOffset,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::RankCT>::type,
        /* BankCoordsWrapper */
        typename tensor_accessor::ArrayWrapperTypeSelectorPackedU16<
            !TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::bank_coords_is_crta,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::BankCoordsCTAOffset,
            TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::NumBanksCT>::type,
        /* IsInterleaved */ !TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::is_sharded,
        /* IsDram */ TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::is_dram>>;

template <
    uint32_t RankCT,
    uint32_t NumBanksCT,
    typename TensorShapeWrapper,
    typename ShardShapeWrapper,
    typename BankCoordsWrapper,
    bool IsInterleaved,
    bool IsDram>
TensorAccessor(
    tensor_accessor::DistributionSpec<
        RankCT,
        NumBanksCT,
        TensorShapeWrapper,
        ShardShapeWrapper,
        BankCoordsWrapper,
        IsInterleaved,
        IsDram>,
    size_t,
    uint32_t)
    -> TensorAccessor<tensor_accessor::DistributionSpec<
        RankCT,
        NumBanksCT,
        TensorShapeWrapper,
        ShardShapeWrapper,
        BankCoordsWrapper,
        IsInterleaved,
        IsDram>>;

namespace tensor_accessor::detail {
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple(
    const std::tuple<Args...>& args,
    uint32_t address_rt_arg_index_start,
    uint32_t page_size_ct_arg_index_start,
    std::integer_sequence<uint32_t, Indexes...>) {
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args),
        get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes),
        kernel_compile_time_args[page_size_ct_arg_index_start + Indexes])...);
}
}  // namespace tensor_accessor::detail

template <typename... Args>
auto make_tensor_accessor_tuple(
    const std::tuple<Args...>& args, uint32_t address_rt_arg_index_start, uint32_t page_size_ct_arg_index_start) {
    return tensor_accessor::detail::make_tensor_accessor_tuple(
        args,
        address_rt_arg_index_start,
        page_size_ct_arg_index_start,
        std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}

/**
 * @brief AbstractTensorAccessorWrapper provides a unified interface over templated tensor accessors.
 *
 * The wrapper allows to use and iterate over different kinds of tensor accessors in a unified way.
 */
class AbstractTensorAccessorWrapper {
public:
    AbstractTensorAccessorWrapper() = default;

    template <typename Accessor>
    AbstractTensorAccessorWrapper(const Accessor& accessor) :
        accessor_ptr(&accessor),
        get_noc_addr_fn([](const void* accessor, uint32_t page_idx, uint32_t offset, uint8_t noc) {
            return static_cast<const Accessor*>(accessor)->get_noc_addr(page_idx, offset, noc);
        }) {}

    uint64_t get_noc_addr(uint32_t page_idx, uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr_fn(accessor_ptr, page_idx, offset, noc);
    }

private:
    using GetNocAddrFn = uint64_t (*)(const void*, uint32_t, uint32_t, uint8_t);

    const void* accessor_ptr = nullptr;
    GetNocAddrFn get_noc_addr_fn = nullptr;
};

namespace tensor_accessor::detail {
template <typename... Accessors, uint32_t... Indexes>
auto make_abstract_tensor_accessor_wrappers(
    const std::tuple<Accessors...>& accessors, std::integer_sequence<uint32_t, Indexes...>)
    -> std::array<AbstractTensorAccessorWrapper, sizeof...(Accessors)> {
    return {AbstractTensorAccessorWrapper(std::get<Indexes>(accessors))...};
}
}  // namespace tensor_accessor::detail

// Wraps a tuple of templated tensor accessors into an array of AbstractTensorAccessorWrapper,
// allowing for easy iteration and runtime dispatch.
template <typename... Accessors>
auto make_abstract_tensor_accessor_wrappers(const std::tuple<Accessors...>& accessors) {
    return tensor_accessor::detail::make_abstract_tensor_accessor_wrappers(
        accessors, std::make_integer_sequence<uint32_t, sizeof...(Accessors)>());
}

// Adapters for experimental NoC APIs
template <typename Accessor>
struct PageView {
    const Accessor& accessor;
    explicit PageView(const Accessor& acc) : accessor(acc) {}

    uint64_t get_noc_addr(uint32_t page_id, uint32_t offset, uint8_t noc) const {
        return accessor.get_noc_addr(page_id, offset, noc);
    }
};

template <typename Accessor>
struct ShardView {
    const Accessor& accessor;
    explicit ShardView(const Accessor& acc) : accessor(acc) {}

    bool is_local_shard(uint32_t shard_id, uint8_t noc) const { return accessor.is_local_shard(shard_id, noc); }
    uint64_t get_noc_addr(uint32_t shard_id, uint32_t offset, uint8_t noc) const {
        return accessor.get_shard_noc_addr(shard_id, offset, noc);
    }
};
// NOLINTEND(misc-unused-parameters)
