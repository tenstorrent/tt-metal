// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "helpers.hpp"

namespace nd_sharding {
namespace detail {
template <uint32_t... PackedCoords>
struct BankCoordWrapperStaticNBanksStaticCoords {
    static constexpr uint32_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    using PackedCoordsBase = std::array<uint32_t, num_banks>;
    static constexpr PackedCoordsBase packed_xy_coords = {PackedCoords...};
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords() = default;
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(const PackedCoordsBase&) {}
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(PackedCoordsBase&&) {}
};

template <uint32_t NumBanks>
struct BankCoordWrapperDynamicStaticNBanksDynamicCoords {
    static constexpr uint32_t num_banks = NumBanks;
    using PackedCoordsBase = std::array<uint32_t, num_banks>;
    PackedCoordsBase packed_xy_coords;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords() = default;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(const PackedCoordsBase& banks_coords) :
        packed_xy_coords(banks_coords) {}
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(PackedCoordsBase&& banks_coords) :
        packed_xy_coords(std::move(banks_coords)) {}
};

struct BankCoordWrapperDynamicsNBanks {
    using PackedCoordsBase = Span<uint32_t>;
    PackedCoordsBase packed_xy_coords;  // Runtime packed coordinates

    explicit BankCoordWrapperDynamicsNBanks() = default;
    explicit BankCoordWrapperDynamicsNBanks(const PackedCoordsBase& bank_coords) : packed_xy_coords(bank_coords) {}
    explicit BankCoordWrapperDynamicsNBanks(PackedCoordsBase&& bank_coords) :
        packed_xy_coords(std::move(bank_coords)) {}
};

template <bool NumBanksStatic, bool BankCoordsStatic, size_t StartIdx, uint32_t NumBanks>
struct BankCoordsWrapperTypeSelector;

template <size_t StartIdx, uint32_t NumBanks>
struct BankCoordsWrapperTypeSelector<true, true, StartIdx, NumBanks> {
    // Both num_banks and coords are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<BankCoordWrapperStaticNBanksStaticCoords, StartIdx, NumBanks>;
};

template <size_t StartIdx, uint32_t NumBanks>
struct BankCoordsWrapperTypeSelector<true, false, StartIdx, NumBanks> {
    // Num_banks is known at compile time, but coords are not
    using type = BankCoordWrapperDynamicStaticNBanksDynamicCoords<NumBanks>;
};

template <bool BankCoordsStatic, size_t StartIdx, uint32_t NumBanks>
struct BankCoordsWrapperTypeSelector<false, BankCoordsStatic, StartIdx, NumBanks> {
    // Num_banks is not known at compile time, doesn't matter if coords are known or not, use poorly dynamic wrapper
    using type = BankCoordWrapperDynamicsNBanks;
};

}  // namespace detail
}  // namespace nd_sharding
