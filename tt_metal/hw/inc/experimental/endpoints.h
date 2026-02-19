// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/noc.h"

namespace experimental {

/**
 * @brief Experimental wrapper around calculating unicast noc address given x, y, and address. This allows direct
 * address to be supplied to NoC apis
 *
 * @note This API is experimental and subject to change.
 */
struct UnicastEndpoint {
    uint64_t get_noc_unicast_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, addr, noc);
    }
};

/**
 * @brief Experimental wrapper around calculating multicast noc address given 2D multicast rectangle and address. This
 * allows direct address to be supplied to NoC apis
 *
 * @note This API is experimental and subject to change.
 */
struct MulticastEndpoint {
    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr, uint8_t noc)
        const {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    }
};

/**
 * @brief Experimental wrapper around calculating noc address targeting a bank managed by the allocator (either DRAM or
 * L1) given bank id and address. This allows direct address to be supplied to NoC apis
 *
 * @note This API is experimental and subject to change.
 */
enum AllocatorBankType { L1, DRAM };

template <AllocatorBankType bank_type>
struct AllocatorBank {
    uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr_from_bank_id < bank_type == AllocatorBankType::DRAM > (bank_id, addr, noc);
    }
};

template <>
struct noc_traits_t<UnicastEndpoint> {
    struct src_args_type {
        uint32_t noc_x{};
        uint32_t noc_y{};
        uint32_t addr{};
    };
    struct dst_args_type {
        uint32_t noc_x{};
        uint32_t noc_y{};
        uint32_t addr{};
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const UnicastEndpoint& src, const Noc& noc, const src_args_type& args) {
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            return args.addr;
        } else {
            uint64_t noc_addr = src.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
            return noc_addr;
        }
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const UnicastEndpoint& dst, const Noc& noc, const dst_args_type& args) {
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            return args.addr;
        } else {
            uint64_t noc_addr = dst.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
            return noc_addr;
        }
    }
};

template <>
struct noc_traits_t<MulticastEndpoint> {
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t addr{};
    };
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const MulticastEndpoint& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(address_type == Noc::AddressType::NOC);
        uint64_t noc_addr = dst.get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, args.addr, noc.get_noc_id());
        return noc_addr;
    }
};

template <AllocatorBankType bank_type>
struct noc_traits_t<AllocatorBank<bank_type>> {
    struct src_args_type {
        uint32_t bank_id{};
        uint32_t addr{};
    };
    struct dst_args_type {
        uint32_t bank_id{};
        uint32_t addr{};
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const AllocatorBank<bank_type>& src, const Noc& noc, const src_args_type& args) {
        static_assert(address_type == Noc::AddressType::NOC);
        uint64_t noc_addr = src.template get_noc_addr_from_bank_id(args.bank_id, args.addr, noc.get_noc_id());
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const AllocatorBank<bank_type>& dst, const Noc& noc, const dst_args_type& args) {
        static_assert(address_type == Noc::AddressType::NOC);
        uint64_t noc_addr = dst.template get_noc_addr_from_bank_id(args.bank_id, args.addr, noc.get_noc_id());
        return noc_addr;
    }
};

}  // namespace experimental
