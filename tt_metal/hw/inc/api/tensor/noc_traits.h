// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "internal/tensor/generated_noc_addr.h"
#include "noc_address_backend.h"

// R/W inference: these endpoints read/write their accessor's tensor (api/dataflow/buf_rw_note.h).
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
#if defined(TT_TA_ADDRGEN_ACTIVE)
// Shared base for NoC traits of every TensorAccessor endpoint that uses HW AddrGen. Defined only when AddrGen is
// available.
template <typename Traits>
struct PushIssue {
    static constexpr bool may_push = true;
    static_assert(read_cmd_buf == 1 && write_cmd_buf == 0, "the push sides feed command buffers 1 (reads), 0 (writes)");

    template <typename Src, typename Args, typename Issue>
    static FORCE_INLINE void issue_read(
        const Src& src,
        const Noc& noc,
        const Args& args,
        uint32_t dst_local_l1_addr,
        uint32_t size,
        uint32_t read_req_vc,
        Issue&& issue) {
        const uint64_t src_noc_addr = Traits::src_noc_addr_or_pushed(src, noc, args);
        if (src_noc_addr != tt_addrgen::kAddrPushed) {
            // The address was not pushed into the command buffer. EXPLAIN WHY THIS CAN HAPPEN OR REFER TO COMMENT WHERE
            // ITS EXPLAINED. In this case we can issue the transaction as defined in noc.h. The same applies to
            // issue_write.
            issue(src_noc_addr);
            return;
        }
        const uint8_t noc_id = noc.get_noc_id();
#if (defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE)) || defined(PROFILE_NOC_EVENTS)
        // NoC sanitizer and NoC event profiler need the remote address before the issue, so
        // read it back from the register the address generator pushed it into. issue_write does a similar readback.
        const uint64_t pushed_addr = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
            read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8);
        overlay::rocc_nop();  // reg read must not be in flight with the next value-returning RoCC (AIHWE-6506)
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ, dst_local_l1_addr, pushed_addr, size, -1, false, noc_id);
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, pushed_addr, dst_local_l1_addr, size);
#endif
        WAYPOINT("NAOW");
        // src_addr arg is ignored
        ncrisc_noc_fast_read<noc_mode, /*src_in_cmd_buf=*/true>(
            noc_id, read_cmd_buf, /*src_addr=*/0, dst_local_l1_addr, size, read_req_vc);
        WAYPOINT("NAOD");
    }

    template <bool posted, bool use_trid, typename Dst, typename Args, typename Issue>
    static FORCE_INLINE void issue_write(
        const Dst& dst,
        const Noc& noc,
        const Args& args,
        uint32_t src_local_l1_addr,
        uint32_t size,
        uint32_t vc,
        uint32_t trid,
        Issue&& issue) {
        const uint64_t dst_noc_addr = Traits::dst_noc_addr_or_pushed(dst, noc, args);
        if (dst_noc_addr != tt_addrgen::kAddrPushed) {
            issue(dst_noc_addr);
            return;
        }
        const uint8_t noc_id = noc.get_noc_id();
#if (defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE)) || defined(PROFILE_NOC_EVENTS)
        // See issue_read
        const uint64_t pushed_addr = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
            write_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8);
        overlay::rocc_nop();  // a register read must not be in flight with the next value-returning RoCC (AIHWE-6506)
#if defined(PROFILE_NOC_EVENTS)
        constexpr auto event_type = use_trid ? KernelProfilerNocEventMetadata::NocEventType::WRITE_WITH_TRID
                                             : KernelProfilerNocEventMetadata::NocEventType::WRITE_;
        const int event_vc = use_trid ? -1 : static_cast<int>(vc);
        RECORD_NOC_EVENT_WITH_ADDR(event_type, src_local_l1_addr, pushed_addr, size, event_vc, posted, noc_id);
#endif
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, pushed_addr, src_local_l1_addr, size);
#endif
        WAYPOINT("NWPW");
        // dest_addr arg is ignored
        ncrisc_noc_fast_write<noc_mode, use_trid, /*update_counter=*/true, /*dest_in_cmd_buf=*/true>(
            noc_id,
            write_cmd_buf,
            src_local_l1_addr,
            /*dest_addr=*/0,
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
template <typename Traits>
struct PushIssue {};
#endif
}  // namespace tensor_accessor::detail

// TODO(#29597): The traits classes for TensorAccessor and related classes could be moved to tensor_accessor.h
// (need to break the include dependency dataflow_api.h -> tensor_accessor.h.).
template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT>> : tensor_accessor::detail::PushIssue<noc_traits_t<TensorAccessor<DSpecT>>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_ACTIVE)
    static FORCE_INLINE uint64_t
    src_noc_addr_or_pushed(const TensorAccessor<DSpecT>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static FORCE_INLINE uint64_t
    dst_noc_addr_or_pushed(const TensorAccessor<DSpecT>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const TensorAccessor<DSpecT>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read>(
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
        uint64_t noc_addr = tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Write>(
            dst, args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};

template <typename Accessor>
struct noc_traits_t<PageView<Accessor>> : tensor_accessor::detail::PushIssue<noc_traits_t<PageView<Accessor>>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_ACTIVE)
    static FORCE_INLINE uint64_t
    src_noc_addr_or_pushed(const PageView<Accessor>& src, const Noc& noc, const src_args_type& args) {
        return tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read, true>(
            src.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static FORCE_INLINE uint64_t
    dst_noc_addr_or_pushed(const PageView<Accessor>& dst, const Noc& noc, const dst_args_type& args) {
        return tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Write, true>(
            dst.accessor, args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const PageView<Accessor>& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read>(
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
        uint64_t noc_addr = tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Write>(
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
        uint64_t noc_addr = tensor_accessor::generated_shard_noc_addr<tensor_accessor::TransferDir::Read>(
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
        uint64_t noc_addr = tensor_accessor::generated_shard_noc_addr<tensor_accessor::TransferDir::Write>(
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

namespace tensor_accessor::detail {
// Traits of the pages the iterators yield (AccessorPage from pages(), ShardPage from shard_pages()). Same argument
// types as Page, so `{.offset_bytes = ...}` call sites are unchanged; the address comes from the transfer path, which
// can use the hardware address generator.
template <typename PageT>
struct IteratorPageNocTraits : PushIssue<IteratorPageNocTraits<PageT>> {
    using src_args_type = noc_traits_t<Page>::src_args_type;
    using dst_args_type = noc_traits_t<Page>::dst_args_type;
#if defined(TT_TA_ADDRGEN_ACTIVE)
    static FORCE_INLINE uint64_t src_noc_addr_or_pushed(const PageT& src, const Noc& noc, const src_args_type& args) {
        return ::tensor_accessor::generated_noc_addr<TransferDir::Read, true>(src, args.offset_bytes, noc.get_noc_id());
    }
    static FORCE_INLINE uint64_t dst_noc_addr_or_pushed(const PageT& dst, const Noc& noc, const dst_args_type& args) {
        return ::tensor_accessor::generated_noc_addr<TransferDir::Write, true>(
            dst, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(const PageT& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr =
            ::tensor_accessor::generated_noc_addr<TransferDir::Read>(src, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const PageT& dst, const Noc& noc, const dst_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr =
            ::tensor_accessor::generated_noc_addr<TransferDir::Write>(dst, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};
}  // namespace tensor_accessor::detail

template <typename Accessor>
struct noc_traits_t<tensor_accessor::AccessorPage<Accessor>>
    : tensor_accessor::detail::IteratorPageNocTraits<tensor_accessor::AccessorPage<Accessor>> {};

template <typename Accessor>
struct noc_traits_t<tensor_accessor::ShardPage<Accessor>>
    : tensor_accessor::detail::IteratorPageNocTraits<tensor_accessor::ShardPage<Accessor>> {};

template <>
struct noc_traits_t<AbstractTensorAccessorWrapper>
    : tensor_accessor::detail::PushIssue<noc_traits_t<AbstractTensorAccessorWrapper>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
#if defined(TT_TA_ADDRGEN_ACTIVE)
    static FORCE_INLINE uint64_t
    src_noc_addr_or_pushed(const AbstractTensorAccessorWrapper& src, const Noc& noc, const src_args_type& args) {
        return src.generated_noc_addr<tensor_accessor::TransferDir::Read, true>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
    }
    static FORCE_INLINE uint64_t
    dst_noc_addr_or_pushed(const AbstractTensorAccessorWrapper& dst, const Noc& noc, const dst_args_type& args) {
        return dst.generated_noc_addr<tensor_accessor::TransferDir::Write, true>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
    }
#endif
    template <Noc::AddressType address_type>
    static auto src_addr(
        const AbstractTensorAccessorWrapper& src, const Noc& noc, const src_args_type& args)
        -> std::conditional_t<address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        uint64_t noc_addr = src.generated_noc_addr<tensor_accessor::TransferDir::Read>(
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
        uint64_t noc_addr = dst.generated_noc_addr<tensor_accessor::TransferDir::Write>(
            args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return noc_address_backend::extract_local_address(noc_addr);
        }
        return noc_addr;
    }
};
