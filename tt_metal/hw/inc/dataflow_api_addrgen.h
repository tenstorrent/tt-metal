// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api_common.h"
#include <noc/noc_parameters.h>
#include "dataflow_cmd_bufs.h"
#include "debug/sanitize_noc.h"
#include "debug/waypoint.h"
#include "utils/utils.h"
#include "debug/assert.h"

namespace interleaved_addr_gen {

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_offset_index(uint32_t id) {
    if constexpr (DRAM) {  // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return udivsi3_const_divisor<NUM_DRAM_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_DRAM_BANKS;
#endif
    } else {  // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return udivsi3_const_divisor<NUM_L1_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_L1_BANKS;
#endif
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index) {
    if constexpr (DRAM) {  // DRAM
        return id - bank_offset_index * NUM_DRAM_BANKS;
    } else {  // L1
        return id - bank_offset_index * NUM_L1_BANKS;
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = noc_index) {
    if constexpr (DRAM) {  // DRAM
        return dram_bank_to_noc_xy[noc][bank_index];
    } else {  // L1
        return l1_bank_to_noc_xy[noc][bank_index];
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_offset(uint32_t bank_index) {
    if constexpr (DRAM) {  // DRAM
        return bank_to_dram_offset[bank_index];
    } else {  // L1
        return bank_to_l1_offset[bank_index];
    }
}

template <bool DRAM>
FORCE_INLINE constexpr uint32_t get_allocator_alignment() {
    if constexpr (DRAM) {
        return DRAM_ALIGNMENT;
    } else {
        return L1_ALIGNMENT;
    }
}

template <bool DRAM>
FORCE_INLINE constexpr uint32_t get_log_base2_of_allocator_alignment() {
    if constexpr (DRAM) {
        return LOG_BASE_2_OF_DRAM_ALIGNMENT;
    } else {
        return LOG_BASE_2_OF_L1_ALIGNMENT;
    }
}
}  // namespace interleaved_addr_gen

template <uint32_t tile_hw = 1024>
FORCE_INLINE constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    constexpr uint8_t datum_shift = (tile_hw == 1024)  ? 10
                                    : (tile_hw == 512) ? 9
                                    : (tile_hw == 256) ? 8
                                    : (tile_hw == 128) ? 7
                                    : (tile_hw == 64)  ? 6
                                    : (tile_hw == 32)  ? 5
                                    : (tile_hw == 16)  ? 4
                                                       : 10;

    constexpr uint8_t exp_shift = (tile_hw == 1024)  ? 6
                                  : (tile_hw == 512) ? 5
                                  : (tile_hw == 256) ? 4
                                  : (tile_hw == 128) ? 4
                                  : (tile_hw == 64)  ? 4
                                  : (tile_hw == 32)  ? 4
                                  : (tile_hw == 16)  ? 4
                                                     : 6;
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::UInt8): return (index << datum_shift);
        case ((uint8_t)DataFormat::UInt16):
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << (datum_shift + 1));
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32):
        case ((uint8_t)DataFormat::Float32): return (index << (datum_shift + 2));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << (datum_shift - 2)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << (datum_shift - 1)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << datum_shift) + (index << (exp_shift)));
    };
}

// clang-format off
/**
 * Get an encoding for a noc address which contains Tensix core grid and L1 address.
 *
 * Return value: uint64_t
 *
 * | Argument    | Description                             | Data type | Valid range        | required |
 * |-------------|-----------------------------------------|-----------|--------------------|----------|
 * | noc_x_start | Physical x coordinate of the start core | uint32_t  | WH: 0-9, BH: 0-16  | True     |
 * | noc_y_start | Physical y coordinate of the start core | uint32_t  | WH: 0-11, BH: 0-11 | True     |
 * | noc_x_end   | Physical x coordinate of the end core   | uint32_t  | WH: 0-9, BH: 0-16  | True     |
 * | noc_y_end   | Physical y coordinate of the end core   | uint32_t  | WH: 0-11, BH: 0-11 | True     |
 * | addr        | Address in local L1 memory              | uint32_t  | 0..1MB             | True     |
 * | noc         | Which NOC to use for the transaction    | uint8_t   | 0 or 1             | False    |
 */
// clang-format on
FORCE_INLINE
uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr,
    uint8_t noc = noc_index) {
    return NOC_MULTICAST_ADDR(
        DYNAMIC_NOC_X(noc, noc_x_start),
        DYNAMIC_NOC_Y(noc, noc_y_start),
        DYNAMIC_NOC_X(noc, noc_x_end),
        DYNAMIC_NOC_Y(noc, noc_y_end),
        addr);
}

// clang-format off
/**
 * Get an encoding for a noc address which contains core and L1 address.
 *
 * Return value: uint64_t
 *
 * | Argument | Description                          | Data type | Valid range        | required |
 * |----------|--------------------------------------|-----------|--------------------|----------|
 * | noc_x    | Physical x coordinate of core        | uint32_t  | WH: 0-9, BH: 0-16  | True     |
 * | noc_y    | Physical y coordinate of core        | uint32_t  | WH: 0-11, BH: 0-11 | True     |
 * | addr     | Address in local L1 memory           | uint32_t  | 0..1MB             | True     |
 * | noc      | Which NOC to use for the transaction | uint8_t   | 0 or 1             | False    |
 */
// clang-format on
FORCE_INLINE
uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc = noc_index) {
    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, noc_x), DYNAMIC_NOC_Y(noc, noc_y), addr);
}

/*
    Need an alias to get_noc_addr so that the structs below don't confuse the above get_noc_addr with
    the struct variant
*/
FORCE_INLINE
std::uint64_t get_noc_addr_helper(std::uint32_t noc_xy, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return ((uint64_t)(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;
}

uint64_t get_dram_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(id, bank_offset_index);
    uint32_t addr =
        (bank_offset_index * align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<true>())) +
        bank_base_address + offset + bank_to_dram_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t get_l1_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<false>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<false>(id, bank_offset_index);
    uint32_t addr =
        (bank_offset_index * align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<false>())) +
        bank_base_address + offset + bank_to_dram_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<false>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t get_system_memory_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t base_addr,
    const uint32_t offset = 0,
    uint8_t noc = noc_index) {
    uint64_t pcie_core_noc_encoding =
        uint64_t(NOC_XY_PCIE_ENCODING(DYNAMIC_NOC_X(noc, PCIE_NOC_X), DYNAMIC_NOC_Y(noc, PCIE_NOC_Y)));
    uint32_t addr = base_addr + page_size * id + offset;
    uint64_t noc_addr = pcie_core_noc_encoding | addr;
    return noc_addr;
}

FORCE_INLINE
std::uint64_t get_noc_addr(std::uint32_t addr, uint8_t noc = noc_index) {
    /*
        Get an encoding which contains the address in L1 on the current core that you want to
        read from/write to via the noc
    */
    return NOC_XY_ADDR(my_x[noc], my_y[noc], addr);
}

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    const uint32_t page_size;    // Num bytes in page.
    const uint32_t aligned_page_size =
        align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<DRAM>());

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index * this->aligned_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;  // WARNING: This struct is used for optimized get_noc_addr in which case
                                             // you know that bank_unit_size is a power of 2
    static constexpr uint32_t log_base_2_of_allocator_alignment =
        interleaved_addr_gen::get_log_base2_of_allocator_alignment<DRAM>();
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > log_base_2_of_allocator_alignment
                                                         ? this->log_base_2_of_page_size
                                                         : log_base_2_of_allocator_alignment;

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    // TODO: Remove page_size from argument list. This can be derived from data_format
    uint32_t page_size;      // Num bytes in bank unit.
    DataFormat data_format;  // Data format

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return MUL_WITH_TILE_SIZE<tile_hw>((uint)this->data_format, bank_offset_index) + this->bank_base_address +
               offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

// TODO: need static assert + host assert that page size <= 8192, hard constraint
template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;              // Base address for the whole tensor.
    const uint32_t log_base_2_of_page_size;  // Num bytes in bank unit.
    static constexpr uint32_t log_base_2_of_allocator_alignment =
        interleaved_addr_gen::get_log_base2_of_allocator_alignment<DRAM>();
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > log_base_2_of_allocator_alignment
                                                         ? this->log_base_2_of_page_size
                                                         : log_base_2_of_allocator_alignment;

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }
};

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGenFast<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2 and less than or equal to NOC_MAX_BURST_SIZE.
        For arbitrary bank unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGenFast: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset, noc);
}

// clang-format off
/**
 * Get an encoding for a noc address using DRAM/L1 bank id. Uses addrgen tables to convert bank_ids to physical NOC coordinates
 *
 * Return value: uint64_t
 *
 * | Argument                 | Description                             | Data type | Valid range                                            | required |
 * |--------------------------|-----------------------------------------|-----------|--------------------------------------------------------|----------|
 * | bank_id                  | DRAM/L1 bank id                         | uint32_t  | Refer to relevant yaml in "tt_metal/soc_descriptors"   | True     |
 * | bank_address_offset      | DRAM/L1 bank address offset             | uint32_t  | 0..1MB                                                 | True     |
 * | noc                      | Which NOC to use for the transaction    | uint8_t   | 0 or 1                                                 | False    |
 * | DRAM (template argument) | Signifies if address is from DRAM or L1 | bool      | True or False                                          | True     |
 */
// clang-format on
template <bool DRAM>
FORCE_INLINE uint64_t
get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t bank_address_offset, uint8_t noc = noc_index) {
    uint64_t noc_addr = 0;
    if constexpr (DRAM) {
        noc_addr = dram_bank_to_noc_xy[noc_index][bank_id];
        bank_address_offset += bank_to_dram_offset[bank_id];
    } else {
        noc_addr = l1_bank_to_noc_xy[noc_index][bank_id];
    }
    return (noc_addr << NOC_ADDR_COORD_SHIFT) | (bank_address_offset);
}

template <bool DRAM, uint32_t page_size>
FORCE_INLINE auto get_interleaved_addr_gen(uint32_t base_addr) {
    constexpr bool is_pow_2 = is_power_of_2(page_size);
    if constexpr (is_pow_2) {
        constexpr uint32_t log2_page_size = __builtin_ctz(page_size);
        if constexpr (page_size <= NOC_MAX_BURST_SIZE) {
            return InterleavedPow2AddrGenFast<DRAM>{
                .bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};
        } else {
            return InterleavedPow2AddrGen<DRAM>{
                .bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};
        }
    } else {
        return InterleavedAddrGen<DRAM>{.bank_base_address = base_addr, .page_size = page_size};
    }
}

template <bool DRAM, bool is_size_pow2>
FORCE_INLINE auto get_interleaved_addr_gen(uint32_t base_addr, uint32_t page_size, uint32_t log2_page_size) {
    if constexpr (is_size_pow2) {
        return InterleavedPow2AddrGen<DRAM>{.bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};
    } else {
        return InterleavedAddrGen<DRAM>{.bank_base_address = base_addr, .page_size = page_size};
    }
}

template <bool DRAM, bool is_size_pow2>
FORCE_INLINE auto get_interleaved_addr_gen(uint32_t base_addr, uint32_t size) {
    return get_interleaved_addr_gen<DRAM, is_size_pow2>(base_addr, size, size);
}
