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
#include "experimental/noc.h"
#include "tools/profiler/noc_debugging_metadata.hpp"
#include "tools/profiler/noc_debugging_profiler.hpp"
#endif

#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "experimental/lock.h"

namespace experimental {

class DataflowBuffer {
public:
#ifdef ARCH_QUASAR
    using DFBInterface = LocalDFBInterface;
#else
    using DFBInterface = LocalCBInterface;
#endif

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

#ifdef ARCH_QUASAR
#ifndef COMPILE_FOR_TRISC
    // Implicit sync APIs
    template <typename Src>
    void read_in(const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args);

    template <typename Dst>
    void write_out(const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args);
    // Implicit sync APIs end
#endif
#else  // tt-1xx
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

#ifdef ARCH_QUASAR
    template <bool is_producer>
    void handle_final_credits(uint16_t transactions_issued, uint8_t txn_id_index);
#endif

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
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };
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

}  // namespace experimental

// Arch-specific _impl bodies for DataflowBuffer member functions
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/dataflow_buffer.inl"
#else
#include "internal/tt-1xx/dataflow_buffer.inl"
#endif
