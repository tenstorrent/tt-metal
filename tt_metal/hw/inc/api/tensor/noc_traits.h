// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/transfer_noc_addr.h"
#include "noc_address_backend.h"

// Op-to-op R/W inference: these endpoints read/write their accessor's tensor (api/dataflow/buf_rw_note.h).
namespace tt_buf_rw {
template <typename Accessor>
struct endpoint<PageView<Accessor>> : endpoint<Accessor> {};
template <typename Accessor>
struct endpoint<ShardView<Accessor>> : endpoint<Accessor> {};
template <typename Accessor>
struct endpoint<tensor_accessor::AccessorPage<Accessor>> : endpoint<Accessor> {};
template <typename Accessor>
struct endpoint<tensor_accessor::ShardPage<Accessor>> : endpoint<Accessor> {};
}  // namespace tt_buf_rw

namespace tensor_accessor::detail {
#if defined(TT_TA_ADDRGEN_PUSH)
// Push (transfer_noc_addr.h): the issue half, for the tensor endpoints' traits below. Noc::async_read / async_write
// (api/dataflow/noc.h) take the remote address with src_addr_or_cmd_buf / dst_addr_or_cmd_buf; when it is
// kAddrInCmdBuf the address generator has already written it into the command buffer, and they issue through these,
// which run the NoC V3 transfer without writing that address.
struct PushIssue {
    static constexpr bool may_push = true;
    static_assert(read_cmd_buf == 1 && write_cmd_buf == 0, "the push sides feed command buffers 1 (reads), 0 (writes)");

    static bool in_cmd_buf(uint64_t noc_addr) { return noc_addr == tt_addrgen::kAddrInCmdBuf; }

    static void read(uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc, uint32_t read_req_vc) {
        WAYPOINT("NAOW");
        ncrisc_noc_fast_read<noc_mode, /*src_in_cmd_buf=*/true>(
            noc, read_cmd_buf, 0, dst_local_l1_addr, size, read_req_vc);
        WAYPOINT("NAOD");
    }

    template <bool posted, bool use_trid = false>
    static void write(uint32_t src_local_l1_addr, uint32_t size, uint8_t noc, uint32_t vc, uint32_t trid = 0) {
        WAYPOINT("NWPW");
        ncrisc_noc_fast_write<noc_mode, use_trid, /*update_counter=*/true, /*dest_in_cmd_buf=*/true>(
            noc,
            write_cmd_buf,
            src_local_l1_addr,
            0,
            size,
            vc,
            false /* mcast */,
            false /* linked */,
            1 /* num_dests */,
            true /* multicast_path_reserve */,
            posted,
            trid);
        WAYPOINT("NWPD");
    }
};
#else
struct PushIssue {};
#endif
}  // namespace tensor_accessor::detail

// TODO(#29597): The traits classes for TensorAccessor and related classes could be moved to tensor_accessor.h
// (need to break the include dependency dataflow_api.h -> tensor_accessor.h.).
template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT>> : tensor_accessor::detail::PushIssue {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_PUSH)
    static uint64_t src_addr_or_cmd_buf(const TensorAccessor<DSpecT>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static uint64_t dst_addr_or_cmd_buf(const TensorAccessor<DSpecT>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const TensorAccessor<DSpecT>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read>(
            src, args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const TensorAccessor<DSpecT>& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write>(
            dst, args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};

template <typename Accessor>
struct noc_traits_t<PageView<Accessor>> : tensor_accessor::detail::PushIssue {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_PUSH)
    static uint64_t src_addr_or_cmd_buf(const PageView<Accessor>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static uint64_t dst_addr_or_cmd_buf(const PageView<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const PageView<Accessor>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read>(
            src.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const PageView<Accessor>& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write>(
            dst.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
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
    static auto src_addr(const ShardView<Accessor>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_shard_noc_addr<tensor_accessor::TransferDir::Read>(
            src.accessor, args.shard_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(src.is_local_shard(args.shard_id, noc.get_noc_id()));
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const ShardView<Accessor>& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_shard_noc_addr<tensor_accessor::TransferDir::Write>(
            dst.accessor, args.shard_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(dst.is_local_shard(args.shard_id, noc.get_noc_id()));
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
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
    static auto src_addr(const tensor_accessor::Page& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = src.noc_addr() + args.offset_bytes;
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const tensor_accessor::Page& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = dst.noc_addr() + args.offset_bytes;
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};

// Pages yielded by the pages() iterators. Same argument types as Page, so `{.offset_bytes = ...}` call sites are
// unchanged; the address comes from the transfer path, which can use the hardware address generator.
template <typename Accessor>
struct noc_traits_t<tensor_accessor::AccessorPage<Accessor>> : tensor_accessor::detail::PushIssue {
    using src_args_type = noc_traits_t<tensor_accessor::Page>::src_args_type;
    using dst_args_type = noc_traits_t<tensor_accessor::Page>::dst_args_type;
#if defined(TT_TA_ADDRGEN_PUSH)
    static uint64_t src_addr_or_cmd_buf(
        const tensor_accessor::AccessorPage<Accessor>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src, args.offset_bytes, noc.get_noc_id());
    }
    static uint64_t dst_addr_or_cmd_buf(
        const tensor_accessor::AccessorPage<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const tensor_accessor::AccessorPage<Accessor>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read>(
            src, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const tensor_accessor::AccessorPage<Accessor>& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write>(
            dst, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};

// Pages yielded by shard_pages(). Same argument types as Page.
template <typename Accessor>
struct noc_traits_t<tensor_accessor::ShardPage<Accessor>> : tensor_accessor::detail::PushIssue {
    using src_args_type = noc_traits_t<tensor_accessor::Page>::src_args_type;
    using dst_args_type = noc_traits_t<tensor_accessor::Page>::dst_args_type;
#if defined(TT_TA_ADDRGEN_PUSH)
    static uint64_t src_addr_or_cmd_buf(
        const tensor_accessor::ShardPage<Accessor>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src, args.offset_bytes, noc.get_noc_id());
    }
    static uint64_t dst_addr_or_cmd_buf(
        const tensor_accessor::ShardPage<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const tensor_accessor::ShardPage<Accessor>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read>(
            src, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const tensor_accessor::ShardPage<Accessor>& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Write>(
            dst, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};

template <>
struct noc_traits_t<AbstractTensorAccessorWrapper> : tensor_accessor::detail::PushIssue {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_PUSH)
    static uint64_t src_addr_or_cmd_buf(
        const AbstractTensorAccessorWrapper& src, const Noc& noc, const src_args_type& args) {
        return src.transfer_noc_addr<tensor_accessor::TransferDir::Read, true>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static uint64_t dst_addr_or_cmd_buf(
        const AbstractTensorAccessorWrapper& dst, const Noc& noc, const dst_args_type& args) {
        return dst.transfer_noc_addr<tensor_accessor::TransferDir::Write, true>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(
        const AbstractTensorAccessorWrapper& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = src.transfer_noc_addr<tensor_accessor::TransferDir::Read>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(
        const AbstractTensorAccessorWrapper& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = dst.transfer_noc_addr<tensor_accessor::TransferDir::Write>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};
