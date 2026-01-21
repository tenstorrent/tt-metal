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

    using OverlayRegisterType = uint8_t[3U]; // 3 bytes per overlay register (24 bits)

    static constexpr uint32_t const BEG_ADDR_VALUE = 0xFFB40000U; // base
    static constexpr uint32_t const END_ADDR_VALUE = 0xFFB5FFFFU; // offset
    static constexpr uint32_t const ADDR_RNG_VALUE = END_ADDR_VALUE - BEG_ADDR_VALUE + 1U;
    
    //static constexpr OverlayRegisterType* const BEG_ADDR = reinterpret_cast<OverlayRegisterType*>(BEG_ADDR_VALUE); // base
    //static constexpr OverlayRegisterType* const END_ADDR = reinterpret_cast<OverlayRegisterType*>(END_ADDR_VALUE); // offset
    
    template<uint32_t Index>
    struct OverlayRegister {
        using size_of_register = std::integral_constant<std::uint32_t, sizeof(OverlayRegisterType)>; // 3 bytes per overlay register (24 bits)

        static constexpr uint32_t INDEX = Index;
        static constexpr uint32_t OFFSET = INDEX * size_of_register::value;        
        static constexpr uint32_t ADDRESS = OverlayRegisterFile::BEG_ADDR_VALUE + OFFSET;

        static FORCE_INLINE uint32_t load() {
            return *(reinterpret_cast<uint32_t*>(get_stream_scratch_register_address<ADDRESS>()));
        }

        static FORCE_INLINE void store(uint32_t const value) {
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

            storage.value = value;
            storage.fields.reserved = 0U;
            write_stream_scratch_register<ADDRESS>(storage.value);
        }
    };

    // None of these types are supposed to be instantiated
    // they just serve as type tags for the overlay registers
    //
    using OR0 = OverlayRegister<0U>;
    using OR1 = OverlayRegister<1U>;
    using OR2 = OverlayRegister<2U>;
    using OR3 = OverlayRegister<3U>;
    using OR4 = OverlayRegister<4U>;
    using OR5 = OverlayRegister<5U>;
    using OR6 = OverlayRegister<6U>;
    using OR7 = OverlayRegister<7U>;
    using OR8 = OverlayRegister<8U>;
    using OR9 = OverlayRegister<9U>;
    using OR10 = OverlayRegister<10U>;
    using OR11 = OverlayRegister<11U>;
    using OR12 = OverlayRegister<12U>;
    using OR13 = OverlayRegister<13U>;
    using OR14 = OverlayRegister<14U>;
    
    // variant type to hold all overlay register types;
    // used for type checking in template functions
    //
    using overlay_register_t = std::variant<
        std::monostate,
        OR0,
        OR1,
        OR2,
        OR3,
        OR4,
        OR5,
        OR6,
        OR7,
        OR8,
        OR9,
        OR10,
        OR11,
        OR12,
        OR13,
        OR14
    >;

    // uses variant type to check if T is a valid overlay register type
    //
    template<typename T>
    struct is_overlay_register_type : std::disjunction<
        std::is_same<T, OR0>,
        std::is_same<T, OR1>,
        std::is_same<T, OR2>,
        std::is_same<T, OR3>,
        std::is_same<T, OR4>,
        std::is_same<T, OR5>,
        std::is_same<T, OR6>,
        std::is_same<T, OR7>,
        std::is_same<T, OR8>,
        std::is_same<T, OR9>,
        std::is_same<T, OR10>,
        std::is_same<T, OR11>,
        std::is_same<T, OR12>,
        std::is_same<T, OR13>,
        std::is_same<T, OR14>
    > {};

    // uses ADDRESS field to check if T is a valid overlay register address
    //
    template<typename T>
    using IsValidOverlayRegisterAddress = std::conditional_t<T::ADDRESS >= BEG_ADDR_VALUE && T::ADDRESS <= END_ADDR_VALUE, std::true_type, std::false_type>;

    // used to define bit widths of overlay register fields
    //
    using NumLowerBits = std::integral_constant<std::uint32_t, 16U>;
    using NumUpperBits = std::integral_constant<std::uint32_t, 8U>;
    using NumReserveBits = std::integral_constant<std::uint32_t, 8U>;

    template<typename T>
    static FORCE_INLINE uint32_t load() {
        static_assert(is_overlay_register_type<T>::value, "T must be an Overlay Register");
        return T::load();
    }

    template<typename T>
    static FORCE_INLINE void store(uint32_t const value) {
        static_assert(is_overlay_register_type<T>::value, "T must be an Overlay Register");
        T::store(value);
    }
};

#endif // end #define __TT_METAL_FABRIC_UARCH_OVERLAYREGISTERFILE_HPP__