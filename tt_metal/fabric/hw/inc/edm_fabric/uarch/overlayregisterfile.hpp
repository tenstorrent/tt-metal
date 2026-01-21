// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef __TT_METAL_FABRIC_UARCH_OVERLAYREGISTERFILE_HPP__
#define __TT_METAL_FABRIC_UARCH_OVERLAYREGISTERFILE_HPP__

#include "../fabric_stream_regs.hpp"

#include <type_traits>
#include <variant>

struct RegisterFile {};

struct OverlayRegisterFile : public RegisterFile {
    static constexpr uint32_t const BEG_ADDR = 0xFFB40000; // base
    static constexpr uint32_t const END_ADDR = 0xFFB5FFFF; // offset
    static constexpr uint32_t const ADDR_RNG = END_ADDR - BEG_ADDR + 1U;

    template<uint32_t ADDR>
    struct OverlayRegister {
        static constexpr uint32_t index = ADDR;
        static constexpr uint32_t ADDRESS = OverlayRegisterFile::BEG_ADDR + ADDR;
    };

    // None of these types are supposed to be instantiated
    // they just serve as type tags for the overlay registers
    //
    using OR0 = OverlayRegister<0U>;
    using OR1 = OverlayRegister<1U>;
    using OR2 = OverlayRegister<2U>;
    using OR3 = OverlayRegister<3U>;
    
    // variant type to hold all overlay register types;
    // used for type checking in template functions
    //
    using overlay_register_t = std::variant<
        std::monostate,
        OR0,
        OR1,
        OR2,
        OR3
    >;

    // uses variant type to check if T is a valid overlay register type
    //
    template<typename T>
    using is_overlay_register_type = std::conditional_t< std::contains<overlay_register_t, T>::value, std::true_type, std::false_type>;

    // uses ADDRESS field to check if T is a valid overlay register address
    //
    template<typename T>
    using IsValidOverlayRegisterAddress = std::conditional_t<T::ADDRESS >= BEG_ADDR && T::ADDRESS <= END_ADDR, std::true_type, std::false_type>;

    // used to define bit widths of overlay register fields
    //
    using NumLowerBits = std::integral_constant<std::uint32_t, 16U>;
    using NumUpperBits = std::integral_constant<std::uint32_t, 8U>;
    using NumReserveBits = std::integral_constant<std::uint32_t, 8U>;

    // serves as convenient storage for bitfield manipulations (masks)
    // used in load/store of combined registers
    //
    static union {
        uint32_t value;
        struct {
            uint16_t lower_addr : NumLowerBits::value;
            uint8_t upper_addr : NumUpperBits::value;
            uint8_t reserved : NumReserveBits::value;
        } fields;
    } storage;

    template<typename T>
    static FORCE_INLINE uint32_t load() {
        static_assert(is_overlay_register_type<T>::value, "T must be an Overlay Register");
        return get_stream_scratch_register_address<T::ADDRESS>();
    }

    template<typename T>
    static FORCE_INLINE void store(uint32_t const value) {
        static_assert(is_overlay_register_type<T>::value, "T must be an Overlay Register");
        storage.value = value;
        storage.fields.reserved = 0U;
        write_stream_scratch_register<T::ADDRESS>(storage.value);
    }

    // eventually support alternating versions of 24 bit overlay registers
    // 16+8 then 8+16; can be split by checking if the LowerBitRegister index is even or odd
    // will need to make more generic by removing "Lower"/"Upper" distinction on templated
    // types
    //
    template<typename LowerBitRegister, typename UpperBitRegister>
    static FORCE_INLINE uint32_t load() {
        static_assert(OverlayRegisterFile::is_overlay_register_type<LowerBitRegister>::value, "LowerBitRegister must be an Overlay Register");
        static_assert(OverlayRegisterFile::is_overlay_register_type<UpperBitRegister>::value, "UpperBitRegister must be an Overlay Register");
        static_assert(LowerBitRegister::index < UpperBitRegister::index, "LowerBitRegister index must be less than UpperBitRegister index");
        static_assert(UpperBitRegister::index - LowerBitRegister::index == 1U, "LowerBitRegister and UpperBitRegister must be consecutive registers");

        uint32_t const lower_bits = OverlayRegisterFile::load<LowerBitRegister>();
        uint32_t const upper_bits = OverlayRegisterFile::load<UpperBitRegister>();

        // overlay registers are 24 bits wide, so we need to combine two registers
        // shift the upper 8 bits into position and combine with lower 24 bits
        //
        return (upper_bits << OverlayRegisterFile::NumLowerBits::value) | lower_bits;
    }

    template<typename LowerBitRegister, typename UpperBitRegister>
    static FORCE_INLINE void store(uint32_t const value) {
        static_assert(OverlayRegisterFile::is_overlay_register_type<LowerBitRegister>::value, "LowerBitRegister must be an Overlay Register");
        static_assert(OverlayRegisterFile::is_overlay_register_type<UpperBitRegister>::value, "UpperBitRegister must be an Overlay Register");
        static_assert(LowerBitRegister::index < UpperBitRegister::index, "LowerBitRegister index must be less than UpperBitRegister index");
        static_assert(UpperBitRegister::index - LowerBitRegister::index == 1U, "LowerBitRegister and UpperBitRegister must be consecutive registers");

        storage.value = value;
        storage.fields.reserved = 0U;

        OverlayRegisterFile::store<LowerBitRegister>(storage.fields.lower_addr);
        OverlayRegisterFile::store<UpperBitRegister>(storage.fields.upper_addr);
    }    
};

#endif // end #define __TT_METAL_FABRIC_UARCH_OVERLAYREGISTERFILE_HPP__