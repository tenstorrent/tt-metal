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

#if __has_include("chlkc_descriptors.h")
#include "chlkc_descriptors.h"
#define DFB_DESCRIPTORS_DEFINED
#endif

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

    // DFBAccessor is backed by a compile-time ID (an implicit CTA).

    // Implicit conversion to uint32_t:
    // This lets a Metal 2.0 kernel pass a DFBAccessor directly to Gen1 (WH/BH) LLK
    // compute APIs that expect a raw CB id.
    // This conversion is constexpr; it's intended for Gen1 use only.
    constexpr operator uint32_t() const noexcept { return id_; }

private:
    uint16_t id_;
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
    DataflowBuffer(DFBAccessor accessor) : DataflowBuffer(static_cast<uint16_t>(accessor)) {}

    // Low-level constructor: prefer DFBAccessor overload above for new kernel code.
    DataflowBuffer(uint16_t logical_dfb_id);

    uint16_t get_id() const { return logical_dfb_id_; }

    // Returns the size of each entry in the DFB
    uint32_t get_entry_size() const;
    uint32_t get_stride_size() const;
    // Returns the total number of entries that can be stored in the DFB
    uint32_t get_total_num_entries() const;
    // Total L1 backing for the global ring (num_entries * entry_size), in bytes.
    uint32_t get_total_size_bytes() const;

    // --- Quasar (tt-2xx) depth / L1 extent queries ---------------------------------
    //
    // A logical DFB on Quasar may use multiple hardware tile counters (TCs) on one RISC
    // (round-robin via tc_idx over tc_slots[0..num_tcs_to_rr-1]). Each TC has its own
    // base_addr and ring extent (limit on DM; ring_size on TRISC). These getters expose
    // three measurement scopes; see also DFB L1 layout diagrams (STRIDED vs ALL).
    //
    //  (1) get_total_*     — entire DFB: num_entries credits, num_entries * entry_size L1.
    //
    //  (2) get_local_*     — active TC only (tc_slots[tc_idx]):
    //        get_local_num_entries()  HW credit depth (buf_capacity) for this TC.
    //        get_local_size_bytes()   linear address interval [base, limit) / ring_size
    //                                 for this TC alone.
    //
    //      local_size_bytes is ONE contiguous L1 interval, but the entries *owned* by this
    //      TC are often NOT adjacent inside it — tile counters can have discontiguous L1:
    //
    //      STRIDED (e.g. 4Sx1S: stride_in_entries = num_producers):
    //        Global ring is interleaved: physical slot (e, p) = e*P + p.
    //        Each TC walks its column with stride_size (= entry_size * P):
    //
    //          L1:  [P0@e0][P1@e0][P2@e0][P3@e0][P0@e1][P1@e1]...
    //          TC0:  ^              ^              ^         (every P-th entry)
    //
    //        TC0's [base0, limit0) spans the full linear ring byte length, but its owned
    //        tiles sit P-1 foreign slots apart — discontiguous within that interval.
    //
    //      ALL (e.g. 4Sx1A: stride_in_entries = 1):
    //        Each TC owns a contiguous block of capacity entries; bases step by
    //        capacity * entry_size:
    //
    //          L1:  [ TC0: cap entries ][ TC1: cap entries ][ TC2: ... ][ TC3: ... ]
    //
    //        Here each TC's owned tiles ARE contiguous within its local interval.
    //
    //  (3) get_ring_span_* — bounding box across ALL TC slots on this RISC:
    //        Contiguous L1 from tc_slots[0].base through the end of the last TC ring
    //        (limit[last] - base[0] on DM; base[last] + ring_size[last] - base[0] on TRISC).
    //        get_ring_span_num_entries() = ring_span_bytes / entry_size (address slots in
    //        that bounding interval, not per-TC owned entry count).
    //
    //        ring_span is the minimal contiguous L1 interval covering every TC window.
    //        In STRIDED layouts TC bases are staggered by entry_size with overlapping
    //        windows; in ALL layouts TC blocks are disjoint and packed. Neither layout
    //        guarantees that every byte in ring_span belongs to one TC's owned entries
    //        (STRIDED: foreign interleaved slots; ALL: only the active TC's block is
    //        "yours" at a given time during round-robin).
    //
    // On tt-1xx (WH/BH) there is no per-TC view; all four getters alias get_total_*.
    uint32_t get_local_num_entries() const;
    uint32_t get_local_size_bytes() const;
    uint32_t get_ring_span_bytes() const;
    uint32_t get_ring_span_num_entries() const;

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) { reserve_back_impl(num_entries); }
    void push_back(uint16_t num_entries) { push_back_impl(num_entries); }
    void wait_front(uint16_t num_entries) { wait_front_impl(num_entries); }
    void pop_front(uint16_t num_entries) { pop_front_impl(num_entries); }
    // Explicit sync APIs end

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Test-only free helpers (defined in internal/tt-2xx/dataflow_buffer_test_helpers.h,
    // NOT part of the public DFB API). Granted friend access to advance the
    // implicit-sync shadow state alongside the HW counter. See that header for
    // semantics + usage rules.
    friend void preload_posted_counter(DataflowBuffer&, uint16_t);
    friend void preload_acked_counter(DataflowBuffer&, uint16_t);
#endif

#ifndef COMPILE_FOR_TRISC
#ifndef ARCH_QUASAR
    bool pages_reservable_at_back(int32_t num_pages) const;
    bool pages_available_at_front(int32_t num_pages) const;
#endif
#endif

#ifdef DFB_DESCRIPTORS_DEFINED
    // JIT descriptor values from chlkc_descriptors.h (indexed by logical_dfb_id_).
    // PACK TRISC uses pack_* arrays; UNPACK/MATH TRISC and DM use unpack_*.
    constexpr uint32_t get_tile_size() const {
#if defined(UCK_CHLKC_PACK)
        return pack_tile_size[logical_dfb_id_];
#else
        return unpack_tile_size[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_tile_r_dim() const {
#if defined(UCK_CHLKC_PACK)
        return pack_tile_r_dim[logical_dfb_id_];
#else
        return unpack_tile_r_dim[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_tile_c_dim() const {
#if defined(UCK_CHLKC_PACK)
        return pack_tile_c_dim[logical_dfb_id_];
#else
        return unpack_tile_c_dim[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_tile_hw() const { return get_tile_r_dim() * get_tile_c_dim(); }

    constexpr uint32_t get_tile_num_faces() const {
#if defined(UCK_CHLKC_PACK)
        return pack_tile_num_faces[logical_dfb_id_];
#else
        return unpack_tile_num_faces[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_face_r_dim() const {
#if defined(UCK_CHLKC_PACK)
        return pack_tile_face_r_dim[logical_dfb_id_];
#else
        return unpack_tile_face_r_dim[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_partial_face() const {
#if defined(UCK_CHLKC_PACK)
        return pack_partial_face[logical_dfb_id_];
#else
        return unpack_partial_face[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_narrow_tile() const {
#if defined(UCK_CHLKC_PACK)
        return pack_narrow_tile[logical_dfb_id_];
#else
        return unpack_narrow_tile[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_num_faces_r_dim() const {
#if defined(UCK_CHLKC_PACK)
        return pack_num_faces_r_dim[logical_dfb_id_];
#else
        return unpack_num_faces_r_dim[logical_dfb_id_];
#endif
    }

    constexpr uint32_t get_num_faces_c_dim() const {
#if defined(UCK_CHLKC_PACK)
        return pack_num_faces_c_dim[logical_dfb_id_];
#else
        return unpack_num_faces_c_dim[logical_dfb_id_];
#endif
    }

    constexpr DataFormat get_dataformat() const {
#if defined(UCK_CHLKC_PACK)
        return static_cast<DataFormat>(pack_dst_format[logical_dfb_id_]);
#else
        return static_cast<DataFormat>(unpack_src_format[logical_dfb_id_]);
#endif
    }

#if !defined(UCK_CHLKC_PACK)
    constexpr DataFormat get_unpack_dst_format() const {
        return static_cast<DataFormat>(unpack_dst_format[logical_dfb_id_]);
    }
#endif

// pack_* format arrays are only emitted for PACK TRISC and DM (see genfiles.cpp).
#if defined(UCK_CHLKC_PACK) || (!defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_UNPACK))
    constexpr DataFormat get_pack_src_format() const {
        return static_cast<DataFormat>(pack_src_format[logical_dfb_id_]);
    }
#endif

#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_UNPACK)
    constexpr DataFormat get_pack_dst_format() const {
        return static_cast<DataFormat>(pack_dst_format[logical_dfb_id_]);
    }
#endif

#endif // DFB_DESCRIPTORS_DEFINED

#ifdef COMPILE_FOR_TRISC
// This can be enabled on Quasar once GH issue #49608 is resolved.
#ifndef ARCH_QUASAR
    uint32_t get_tile_address(uint32_t tile_index);

    // Reads one scalar element from a tile at specified tile_index. element_offset is an index into the tile as a T[]
    // array from its L1 base address (not a byte offset); each step is sizeof(T) bytes
    // (default T=uint32_t → 4-byte words).
    // Values are mailbox-broadcast to all TRISC threads as a zero-extended uint32_t; MATH/PACK cast back to T.
    template <typename T = uint32_t>
    T read_tile_value(uint32_t tile_index, uint32_t element_offset);
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

    constexpr uint32_t address_units_to_bytes(uint32_t units) const {
        return units << cb_addr_shift;
    }

    uint16_t logical_dfb_id_;

    // MATH TRISC does not own fifo state (see trisc firmware: cb_interface / g_dfb_interface
    // exist only on UNPACK/PACK). Compute kernels still construct DataflowBuffer on all TRISC
    // threads; MATH carries logical_dfb_id_ only and no-ops sync / runtime-interface accessors.
#if !(defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_MATH))
    DFBInterface& local_dfb_interface_;
#endif

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

#ifndef COMPILE_FOR_TRISC
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/noc_zero_l1.inl"
#else
#include "internal/tt-1xx/noc_zero_l1.inl"
#endif
#include "internal/noc_zero_dram.inl"
#endif
