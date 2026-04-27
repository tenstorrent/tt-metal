// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/noc.h"
#include "api/tensor/tensor_accessor.h"

namespace experimental {

// TODO(#29597): The traits classes for TensorAccessor and related classes could be moved to tensor_accessor.h
// (need to break the include dependency dataflow_api.h -> tensor_accessor.h.).
template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const TensorAccessor<DSpecT>& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const TensorAccessor<DSpecT>& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
};

// Bounds check for a TensorAccessor-addressed transaction.
//
// Interleaved layout: pages with consecutive page_ids live in different banks, so a
//   transaction that steps past page_id's byte boundary lands in physically-adjacent
//   storage that belongs to a DIFFERENT page of the tensor. Bound is one page.
//
// Sharded layout: pages within a shard are physically contiguous on one bank/core, so a
//   transaction is legal iff it stays inside the shard that contains args.page_id. The
//   amount of room left depends on where args.page_id sits inside its shard — starting
//   halfway through a shard gives you half a shard of room, not a whole shard.
//   bank_page_offset encodes the multi-dim shard math already; (bank_page_offset %
//   shard_volume) is the page's index within its shard.
//
// The is_interleaved discriminator is compile-time (DSpecT::is_interleaved), so the
// unused branch is eliminated.
template <typename DSpecT, typename Args>
FORCE_INLINE std::enable_if_t<
    std::is_same_v<Args, typename noc_traits_t<TensorAccessor<DSpecT>>::src_args_type> ||
    std::is_same_v<Args, typename noc_traits_t<TensorAccessor<DSpecT>>::dst_args_type>>
noc_traits_check_bounds(const TensorAccessor<DSpecT>& endpoint, const Args& args, uint32_t size_bytes) {
    if constexpr (DSpecT::is_interleaved) {
        // Interleaved kernel-build specialization inherits from InterleavedAddrGen and
        // does not carry tensor_volume — it is stateless about tensor shape. The runtime
        // NOC address-range sanitizer catches out-of-range page_ids downstream.
        ASSERT(args.offset_bytes + size_bytes <= endpoint.get_aligned_page_size());
    } else {
        // page_id range: surfaced explicitly so the failing line is our trait call,
        // not accessor internals. (get_bank_and_offset below also asserts this.)
        ASSERT(args.page_id < endpoint.dspec().tensor_volume());
        const uint32_t shard_volume = endpoint.dspec().shard_volume();
        const uint32_t page_size_b = endpoint.get_aligned_page_size();
        const uint32_t page_in_shard =
            static_cast<uint32_t>(endpoint.get_bank_and_offset(args.page_id).bank_page_offset % shard_volume);
        const uint32_t bytes_left_in_shard = (shard_volume - page_in_shard) * page_size_b;
        ASSERT(args.offset_bytes + size_bytes <= bytes_left_in_shard);
    }
}

template <typename Accessor>
struct noc_traits_t<PageView<Accessor>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const PageView<Accessor>& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const PageView<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
};

// PageView wraps an accessor; bounds semantics track the underlying layout (see
// TensorAccessor specialization above for the interleaved-vs-sharded rationale).
template <typename Accessor, typename Args>
FORCE_INLINE std::enable_if_t<
    std::is_same_v<Args, typename noc_traits_t<PageView<Accessor>>::src_args_type> ||
    std::is_same_v<Args, typename noc_traits_t<PageView<Accessor>>::dst_args_type>>
noc_traits_check_bounds(const PageView<Accessor>& endpoint, const Args& args, uint32_t size_bytes) {
    if constexpr (Accessor::DSpec::is_interleaved) {
        // Interleaved accessor is stateless about tensor shape; see TensorAccessor
        // specialization above for rationale.
        ASSERT(args.offset_bytes + size_bytes <= endpoint.accessor.get_aligned_page_size());
    } else {
        ASSERT(args.page_id < endpoint.accessor.dspec().tensor_volume());
        const uint32_t shard_volume = endpoint.accessor.dspec().shard_volume();
        const uint32_t page_size_b = endpoint.accessor.get_aligned_page_size();
        const uint32_t page_in_shard =
            static_cast<uint32_t>(endpoint.accessor.get_bank_and_offset(args.page_id).bank_page_offset % shard_volume);
        const uint32_t bytes_left_in_shard = (shard_volume - page_in_shard) * page_size_b;
        ASSERT(args.offset_bytes + size_bytes <= bytes_left_in_shard);
    }
}

template <typename Accessor>
struct noc_traits_t<ShardView<Accessor>> {
    struct src_args_type {
        uint32_t shard_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t shard_id{};
        uint32_t offset_bytes = 0;
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const ShardView<Accessor>& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.shard_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(src.is_local_shard(args.shard_id, noc.get_noc_id()));
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const ShardView<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.shard_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(dst.is_local_shard(args.shard_id, noc.get_noc_id()));
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
};

template <>
struct noc_traits_t<tensor_accessor::Page> {
    struct src_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t offset_bytes = 0;
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const tensor_accessor::Page& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.noc_addr() + args.offset_bytes;
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const tensor_accessor::Page& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.noc_addr() + args.offset_bytes;
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
};

}  // namespace experimental
