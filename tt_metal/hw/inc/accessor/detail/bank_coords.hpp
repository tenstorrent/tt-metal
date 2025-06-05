#pragma once

#include <array>
#include <cstddef>

namespace nd_sharding {
namespace detail {
template <size_t... PackedCoords>
struct BankCoordWrapperStaticNBanksStaticCoords {
    static constexpr bool is_static = true;
    static constexpr size_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    using PackedCoordsArray = std::array<uint32_t, num_banks>;
    static constexpr PackedCoordsArray packed_xy_coords = {PackedCoords...};
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords() = default;
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(const PackedCoordsArray&) {}
    constexpr explicit BankCoordWrapperStaticNBanksStaticCoords(PackedCoordsArray&&) {}
};

template <size_t NumBanks>
struct BankCoordWrapperDynamicStaticNBanksDynamicCoords {
    static constexpr bool is_static = false;
    static constexpr size_t num_banks = NumBanks;
    using PackedCoordsArray = std::array<uint32_t, num_banks>;
    PackedCoordsArray packed_xy_coords;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords() = default;
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(const PackedCoordsArray& banks_coords) :
        packed_xy_coords(banks_coords) {}
    constexpr explicit BankCoordWrapperDynamicStaticNBanksDynamicCoords(PackedCoordsArray&& banks_coords) :
        packed_xy_coords(std::move(banks_coords)) {}
};
}  // namespace detail
}  // namespace nd_sharding
