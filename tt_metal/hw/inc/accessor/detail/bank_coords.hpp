#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "helpers.hpp"

namespace nd_sharding {
namespace detail {
template <size_t... PackedCoords>
struct BankCoordWrapperStaticNBanksStaticCoords {
    static constexpr bool is_static = true;
    static constexpr bool has_static_num_banks = true;
    static constexpr size_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    using PackedCoordsBase = std::array<uint32_t, num_banks>;
    static constexpr PackedCoordsBase packed_xy_coords = {PackedCoords...};
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords() = default;
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(const PackedCoordsBase&) {}
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(PackedCoordsBase&&) {}
};

template <size_t NumBanks>
struct BankCoordWrapperDynamicStaticNBanksDynamicCoords {
    static constexpr bool is_static = false;
    static constexpr bool has_static_num_banks = true;
    static constexpr size_t num_banks = NumBanks;
    using PackedCoordsBase = std::array<uint32_t, num_banks>;
    PackedCoordsBase packed_xy_coords;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords() = default;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(const PackedCoordsBase& banks_coords) :
        packed_xy_coords(banks_coords) {}
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(PackedCoordsBase&& banks_coords) :
        packed_xy_coords(std::move(banks_coords)) {}
};

struct BankCoordWrapperDynamicsNBanks {
    static constexpr bool is_static = false;
    static constexpr bool has_static_num_banks = false;
    using PackedCoordsBase = Span<uint32_t>;

    uint32_t num_banks;                 // Number of banks is dynamic
    PackedCoordsBase packed_xy_coords;  // Runtime packed coordinates

    explicit BankCoordWrapperDynamicsNBanks(PackedCoordsBase&& bank_coords) :
        num_banks(bank_coords.size()), packed_xy_coords(std::move(bank_coords)) {}
};
}  // namespace detail
}  // namespace nd_sharding
