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

    template<uint32_t Index>
    struct OverlayRegister {
        static constexpr uint32_t index = Index;
        static constexpr uint32_t ADDRESS = OverlayRegisterFile::BEG_ADDR + Index;

        static volatile uint32_t* get_address_ptr() {
            return reinterpret_cast<volatile uint32_t*>(get_stream_scratch_register_address<ADDRESS>());
        }

        static FORCE_INLINE uint32_t load() {
            return *get_address_ptr();
        }

        static FORCE_INLINE void store(uint32_t const value) {
            // serves as convenient storage for bitfield manipulations (masks)
            // used in load/store of combined registers
            //
            union {
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
    struct is_overlay_register_type : std::disjunction<
        std::is_same<T, OR0>,
        std::is_same<T, OR1>,
        std::is_same<T, OR2>,
        std::is_same<T, OR3>
    > {};

    // uses ADDRESS field to check if T is a valid overlay register address
    //
    template<typename T>
    using IsValidOverlayRegisterAddress = std::conditional_t<T::ADDRESS >= BEG_ADDR && T::ADDRESS <= END_ADDR, std::true_type, std::false_type>;

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