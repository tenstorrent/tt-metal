// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Arch-specific interface includes
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h"  // For g_dfb_interface extern declaration

#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#endif
#else  // tt-1xx
#include "internal/circular_buffer_interface.h"
#endif

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/noc.h"
#include "tools/profiler/noc_debugging_profiler.hpp"
#endif

#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "api/lock.h"

// Opaque handle for a DataflowBuffer binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a DFB endpoint to a kernel.
// The user then uses that accessor_name to construct a DataflowBuffer in the kernel code.
//
// Usage example:
//   // (Host code declares "my_dfb_name" as the DFB local accessor name for this kernel.)
//   // In the kernel code:
//   DataflowBuffer my_dfb(dfb::my_dfb_name);
//
// Here my_dfb_name is a constexpr DFBAccessor, auto-included in kernel_bindings_generated.h.
//
struct DFBAccessor {
    explicit constexpr DFBAccessor(uint16_t id) noexcept : id_(id) {}

    // Constexpr behavior for ID extraction:
    //  - On WH/BH, DFBAccessor is always constexpr. It's backed by a compile-time ID.
    //  - On Quasar, we reserve the right to switch to a runtime-backed ID.
    //    (Future-proofing workaround to a future problem with DM assignments.)

    // Implicit conversion to uint32_t:
    // This lets a Metal 2.0 kernel pass a DFBAccessor directly to Gen1 (WH/BH) LLK
    // compute APIs that expect a raw CB id.
    // This conversion is constexpr; it's intended for Gen1 use only.
    constexpr operator uint32_t() const noexcept { return id_; }

    // Explicit ID accessor:
    // This meant only for Quasar LLK APIs, which accept DFBAccessor directly.
    // Intentionally NOT constexpr; since body could become an RTA retrieval.
    // (If constexpr is needed by LLK, we could fix that for compute kernels.)
    uint16_t resolve_id() const noexcept { return id_; }

private:
    uint16_t id_;
    // uint32_t rta_idx; // future-proofing option
};

class DataflowBuffer {
public:
#ifdef ARCH_QUASAR
    using DFBInterface = LocalDFBInterface;
#else
    using DFBInterface = LocalCBInterface;
#endif

    // Preferred constructor for Metal 2.0 / ProgramSpec kernels.
    // Pass the named binding constant from kernel_bindings_generated.h:
    //   DataflowBuffer dfb(my_dfb_name);
    DataflowBuffer(DFBAccessor accessor) : DataflowBuffer(accessor.resolve_id()) {}

    // Low-level constructor: prefer DFBAccessor overload above for new kernel code.
    DataflowBuffer(uint16_t logical_dfb_id);

    uint16_t get_id() const { return logical_dfb_id_; }

    uint32_t get_entry_size() const;
    uint32_t get_stride_size() const;

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) { reserve_back_impl(num_entries); }
    void push_back(uint16_t num_entries) { push_back_impl(num_entries); }
    void wait_front(uint16_t num_entries) { wait_front_impl(num_entries); }
    void pop_front(uint16_t num_entries) { pop_front_impl(num_entries); }
    // Explicit sync APIs end

#ifndef ARCH_QUASAR
#ifndef COMPILE_FOR_TRISC
    bool pages_reservable_at_back(int32_t num_pages) const;
    bool pages_available_at_front(int32_t num_pages) const;
#ifdef DATA_FORMATS_DEFINED
    uint32_t get_tile_size() const;
    uint32_t get_tile_hw() const;
    DataFormat get_dataformat() const;
#endif
#else  // trisc
    uint32_t get_tile_address(uint32_t tile_index);
    uint32_t read_tile_value(uint32_t tile_index, uint32_t element_offset);
#endif
#endif

    void finish() { finish_impl(); }

#ifndef COMPILE_FOR_TRISC
    // This should not be used on WH/BH if the read into/write out of the DFB uses transaction ids because the transaction ids are not tracked.
    // Instead, use noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid})
    void write_barrier(const Noc &noc) const { write_barrier_impl(noc); }
#endif

    uint32_t get_write_ptr() const { return get_write_ptr_impl(); }
    uint32_t get_read_ptr()  const { return get_read_ptr_impl(); }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void reserve_back_impl(uint16_t num_entries);
    void push_back_impl(uint16_t num_entries);
    void wait_front_impl(uint16_t num_entries);
    void pop_front_impl(uint16_t num_entries);
    void finish_impl();
    uint32_t get_write_ptr_impl() const;
    uint32_t get_read_ptr_impl()  const;
#ifndef COMPILE_FOR_TRISC
    void write_barrier_impl(const Noc &noc) const;
#endif

#ifdef ARCH_QUASAR
    template <bool is_producer>
    void handle_final_credits(uint16_t transactions_issued, uint8_t txn_id_index);

#ifndef COMPILE_FOR_TRISC
    friend class Noc;  // grants Noc::async_read/write access to prepare_*/commit_* implicit-sync helpers

    uint32_t prepare_implicit_read();
    void commit_implicit_read();

    uint32_t prepare_implicit_write();
    void commit_implicit_write();
#endif // !COMPILE_FOR_TRISC
#endif // ARCH_QUASAR

    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    DFBInterface& local_dfb_interface_;
    uint16_t logical_dfb_id_;

#ifdef ARCH_QUASAR
    // Metadata for implicit sync
    uint16_t ptxn_id_loop_cnt_ = 0;
    uint8_t ptxn_id_index_ = 0;
    uint16_t ptiles_read_ = 0;  // not the same as tile counter: HW has no way to track pending posts

    uint16_t ctxn_id_loop_cnt_ = 0;
    uint8_t ctxn_id_index_ = 0;
    uint16_t ctiles_written_ = 0;  // not the same as tile counter: HW has no way to track pending acks
#endif
};

#ifndef COMPILE_FOR_TRISC

template <>
struct noc_traits_t<DataflowBuffer> {

    // Alias the struct defined in noc.h so that noc_traits_t<DataflowBuffer>::src/dst_args_type
    // stays consistent with the DFB-specific Noc overload signatures.
    using src_args_type = DataflowBufferArgs;
    using dst_args_type = DataflowBufferArgs;

    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const DataflowBuffer& src, const Noc&, const src_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "DataflowBuffer without mcast range can only be used as L1 source");
        return src.get_read_ptr() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const DataflowBuffer& dst, const Noc& noc, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "DataflowBuffer without mcast range can only be used as L1 destination");
        return dst.get_write_ptr() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const DataflowBuffer& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(
            address_type == Noc::AddressType::NOC, "DataflowBuffer with mcast range cannot be used as L1 destination");
        auto local_addr = dst.get_write_ptr() + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, local_addr, noc.get_noc_id());
    }
};

#endif

// Arch-specific _impl bodies for DataflowBuffer member functions
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/dataflow_buffer.inl"
#else
#include "internal/tt-1xx/dataflow_buffer.inl"
#endif
