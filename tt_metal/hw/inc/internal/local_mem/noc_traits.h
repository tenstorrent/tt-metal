// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NoC endpoint traits for the node-local L1 handles: CoreLocalMem, Scratchpad and LocalTensorAccessor. Included by
// their public headers (api/core_local_mem.h, api/scratchpad.h, api/tensor/local_tensor_accessor.h); not for direct
// use.

#if !defined(COMPILE_FOR_TRISC)

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/debug/assert.h"
#include "internal/local_mem/address_access.h"

template <typename T>
class Scratchpad;
template <typename T>
class LocalTensorAccessor;

namespace local_mem::detail {
// Endpoint address of a node-local L1 address: as is for the local side of a transfer, or this node's NoC address of it
// for the NoC side (a loopback transfer).
template <Noc::AddressType address_type>
inline auto endpoint_addr(const Noc& noc, uintptr_t l1_addr) {
    if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
        return l1_addr;
    } else {
        const uint8_t noc_id = noc.get_noc_id();
        return noc_traits_t<UnicastEndpoint>::src_addr<address_type>(
            UnicastEndpoint{},
            noc,
            {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = static_cast<uint32_t>(l1_addr)});
    }
}
}  // namespace local_mem::detail

// A local L1 handle can be either endpoint of a NoC transaction. As the local side (write source / read destination)
// it is the handle's L1 address; as the NoC side (read source / write destination) it names this node's own L1, so the
// transfer is a loopback. `offset_bytes` addresses within the region, in bytes.

template <typename T, typename AddressType>
struct noc_traits_t<CoreLocalMem<T, AddressType>> {
    struct src_args_type {
        AddressType offset_bytes = 0;
    };
    struct dst_args_type {
        AddressType offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const CoreLocalMem<T, AddressType>& src, const Noc& noc, const src_args_type& args) {
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(src) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc& noc, const dst_args_type& args) {
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(dst) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(
        const CoreLocalMem<T, AddressType>& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(false, "CoreLocalMem cannot be used as NoC mcast destination");
    }
};

// Scratchpad: `offset_bytes` is bounds-checked with an assertion.
template <typename T>
struct noc_traits_t<Scratchpad<T>> {
    struct src_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const Scratchpad<T>& src, const Noc& noc, const src_args_type& args) {
        ASSERT(args.offset_bytes < src.size_in_bytes());
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(src.local_mem()) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const Scratchpad<T>& dst, const Noc& noc, const dst_args_type& args) {
        ASSERT(args.offset_bytes < dst.size_in_bytes());
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(dst.local_mem()) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const Scratchpad<T>&, const Noc&, const dst_args_mcast_type&) {
        // A scratchpad is private to one kernel instance per node, so multicasting into it would be
        // writing into other kernels' private memory. Revisit only if a real use case appears.
        static_assert(false, "Scratchpad cannot be used as a NoC multicast destination");
    }
};

// LocalTensorAccessor: the region's extent is not known here, so `offset_bytes` is not bounds-checked.
template <typename T>
struct noc_traits_t<LocalTensorAccessor<T>> {
    struct src_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const LocalTensorAccessor<T>& src, const Noc& noc, const src_args_type& args) {
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(src.local_mem()) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const LocalTensorAccessor<T>& dst, const Noc& noc, const dst_args_type& args) {
        return local_mem::detail::endpoint_addr<address_type>(
            noc, local_mem::detail::AddressAccess::address(dst.local_mem()) + args.offset_bytes);
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const LocalTensorAccessor<T>&, const Noc&, const dst_args_mcast_type&) {
        static_assert(false, "LocalTensorAccessor cannot be used as a NoC multicast destination");
    }
};

template <typename T, typename AddressType>
inline constexpr bool noc_zero_l1_endpoint_v<CoreLocalMem<T, AddressType>> = true;
template <typename T>
inline constexpr bool noc_zero_l1_endpoint_v<Scratchpad<T>> = true;
template <typename T>
inline constexpr bool is_scratchpad_v<Scratchpad<T>> = true;
template <typename T>
inline constexpr bool noc_zero_l1_endpoint_v<LocalTensorAccessor<T>> = true;

#endif  // !defined(COMPILE_FOR_TRISC)
