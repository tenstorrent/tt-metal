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
    using size_of_register = std::integral_constant<std::uint32_t, sizeof(OverlayRegisterType)>; // 3 bytes per overlay register (24 bits)

    static constexpr uint32_t const BEG_ADDR_VALUE = 0xFFB40000U; // base
    static constexpr uint32_t const END_ADDR_VALUE = 0xFFB5FFFFU; // offset
    static constexpr uint32_t const ADDR_RNG_VALUE = END_ADDR_VALUE - BEG_ADDR_VALUE + 1U;
    static constexpr uint32_t const NUM_OVERLAY_REGISTERS = 30U;

    using NumLowerBits = std::integral_constant<std::uint32_t, 16U>;
    using NumUpperBits = std::integral_constant<std::uint32_t, 8U>;
    using NumReserveBits = std::integral_constant<std::uint32_t, 8U>;

    static FORCE_INLINE uint32_t load(uint32_t const register_identifier) {
        return read_stream_scratch_register(register_identifier);
    }

    static FORCE_INLINE void store(uint32_t const register_identifier, uint32_t const value) {
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
        write_stream_scratch_register(register_identifier, storage.value);
    }    

    template<uint32_t RegIdx>
    static constexpr FORCE_INLINE uint32_t get_register() {
        static_assert(RegIdx < NUM_OVERLAY_REGISTERS, "Overlay Register Index out of bounds");
        return RegIdx;
    }

};

#endif // end #define __TT_METAL_FABRIC_UARCH_OVERLAYREGISTERFILE_HPP__