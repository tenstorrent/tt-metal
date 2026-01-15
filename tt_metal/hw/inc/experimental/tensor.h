// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
