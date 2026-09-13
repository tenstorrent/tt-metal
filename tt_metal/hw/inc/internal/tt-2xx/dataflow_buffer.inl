// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Defines the _impl bodies for DataflowBuffer on tt-2xx architectures

#ifdef ARCH_QUASAR

#if defined(COMPILE_FOR_TRISC)
#include "ckernel_trisc_common.h"
#ifdef UCK_CHLKC_PACK
#include "llk_io_pack.h"
#endif
#ifdef UCK_CHLKC_UNPACK
#include "llk_io_unpack.h"
#endif
#endif

#include "api/kernel_thread_globals.h"
#include "internal/scoped_lock_cache_ops.h"  // scoped_lock_acquire/release_cache_ops

#include <type_traits>

#if DFB_IS_COMPUTE_MATH
template <DFBAccess Pap, DFBAccess Cap>
inline DataflowBuffer<Pap, Cap>::DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
}
#else
template <DFBAccess Pap, DFBAccess Cap>
inline DataflowBuffer<Pap, Cap>::DataflowBuffer(uint16_t logical_dfb_id)
    : logical_dfb_id_(logical_dfb_id), local_dfb_interface_(get_local_dfb_interface(logical_dfb_id)) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
    if constexpr (!kKnown) {
        // A pattern-agnostic DataflowBuffer keeps the wire bits at runtime and only supports rings
        // where nothing is BLOCKED: its share is always 1 and, a split op being a whole block, it is
        // never split (producer_split() / consumer_split() are false for it).
        ASSERT(local_dfb_interface_.block_size <= 1u);
    }
    // Declare this DFB's L1 extent to the NOC-debug tracker so a write into it without holding the
    // lock can be flagged.
    RECORD_SCOPED_LOCK_EVENT(
        NocDebuggingEventMetadata::NocDebugEventType::DFB_REGION_START,
        address_units_to_bytes(local_dfb_interface_.tc_slots[0].base_addr),
        get_ring_span_bytes());
}
#endif

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_entry_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_stride_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.stride_size);
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_total_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return local_dfb_interface_.num_entries;
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_total_size_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return get_total_num_entries() * address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_local_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    const dfb::PackedTileCounter packed_tc =
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    const uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC)
    return static_cast<uint32_t>(ckernel::trisc::tile_counters[tc_id].f.buf_capacity);
#else
    const uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    return static_cast<uint32_t>(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id));
#endif
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_local_size_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    const auto& slot = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
#if defined(COMPILE_FOR_TRISC)
    return address_units_to_bytes(slot.ring_size);
#else
    return slot.limit - slot.base_addr;
#endif
#endif
}

namespace {

#if !DFB_IS_COMPUTE_MATH

#ifndef COMPILE_FOR_TRISC
// The wire stride in entries (see DataflowBuffer::wire_stride_tiles for what it means per side). The
// host serializes stride_size = entry_size * stride with stride >= 1 (checked at DM init).
inline uint32_t dfb_dm_stride_entries(const LocalDFBInterface& intf) { return intf.stride_size / intf.entry_size; }

// A split op's n tiles belong to every one of this hart's counters, per_tc each. The host only
// builds rings where the block divides evenly over the counters.
inline uint16_t dfb_dm_per_tc(const LocalDFBInterface& intf, uint16_t n) {
    ASSERT(n % intf.num_tcs_to_rr == 0);
    return static_cast<uint16_t>(n / intf.num_tcs_to_rr);
}

// Spin until every counter can absorb (producer) or supply (consumer) per_tc tiles. Used when
// one transaction spans all the counters (split_tc) or posts to all of them (broadcast_tc).
template <bool is_write>
inline void dfb_dm_all_tc_wait(const LocalDFBInterface& intf, uint32_t per_tc) {
    bool ready = false;
    while (!ready) {
        ready = true;
        for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
            dfb::PackedTileCounter ptc = intf.tc_slots[i].packed_tile_counter;
            const uint8_t tid = dfb::get_tensix_id(ptc);
            const uint8_t cid = dfb::get_counter_id(ptc);
            ASSERT(overlay::fast_llk_intf_get_capacity(tid, cid) >= per_tc);
            const uint32_t level = is_write ? overlay::fast_llk_intf_get_free_space(tid, cid)
                                            : overlay::fast_llk_intf_get_occupancy(tid, cid);
            if (level < per_tc) {
                ready = false;
                break;
            }
        }
    }
}

// Implicit-sync availability on one counter. The HW free-space / occupancy registers only reflect
// transfers whose ISR credit has already landed, so a transfer this hart issued moments ago still
// looks free (producer) or still counts as available (consumer). Measure against this hart's own
// bookkeeping instead: `mine` is how many entries this hart has issued on (producer) or claimed
// from (consumer) the counter, and the op needs `need` more. A producer has room when
// capacity - (mine - acked) >= need; a consumer has data when posted - mine >= need. The HW
// values are 16-bit and wrap, hence the modular subtractions.
template <bool is_write>
inline void dfb_dm_implicit_wait(const LocalDFBInterface& intf, uint8_t slot, uint16_t mine, uint16_t need) {
    dfb::PackedTileCounter ptc = intf.tc_slots[slot].packed_tile_counter;
    const uint8_t tid = dfb::get_tensix_id(ptc);
    const uint8_t cid = dfb::get_counter_id(ptc);
    if constexpr (is_write) {
        const uint32_t capacity = overlay::fast_llk_intf_get_capacity(tid, cid);
        ASSERT(capacity >= need);
        while (static_cast<uint32_t>(static_cast<uint16_t>(mine - overlay::fast_llk_intf_read_acked(tid, cid))) + need >
               capacity);
    } else {
        while (static_cast<uint16_t>(overlay::fast_llk_intf_read_posted(tid, cid) - mine) < need);
    }
}

// Post (producer) or ack (consumer) per_tc credits on every counter, each counter's peer
// gets its share of the one transaction.
template <bool is_write>
inline void dfb_dm_all_tc_credit(const LocalDFBInterface& intf, uint16_t per_tc) {
    for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = intf.tc_slots[i].packed_tile_counter;
        const uint8_t tid = dfb::get_tensix_id(ptc);
        const uint8_t cid = dfb::get_counter_id(ptc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tid, cid) >= per_tc);
        if constexpr (is_write) {
            overlay::fast_llk_intf_inc_posted(tid, cid, per_tc);
        } else {
            overlay::fast_llk_intf_inc_acked(tid, cid, per_tc);
        }
    }
}

// Each tile counter keeps its own cursor, a bookmark into its tiles of the ring. Move one
// counter's bookmark past an n-tile op of its own: n - 1 strides to the op's last tile, then
// `jump` bytes to this counter's next tile (past the other counters'); a bookmark that runs off
// the end of the ring wraps to its base.
template <bool is_write>
inline void dfb_dm_step_slot(const LocalDFBInterface& intf, DFBTCSlot& slot, uint32_t n) {
    uint32_t& ptr = is_write ? slot.wr_ptr : slot.rd_ptr;
    ptr += (n - 1u) * intf.stride_size + intf.jump;
    if (ptr >= slot.limit) {
        ptr = slot.base_addr;
    }
}

// After a split transaction: every counter just received (producer) or gave up (consumer) its
// per_tc share of the block, so every slot's bookmark steps past it. tc_idx stays at index 0
// because each TC takes part in a split transaction, so the next one is still addressed from
// slot 0's cursor.
template <bool is_write>
inline void dfb_dm_all_slots_advance(LocalDFBInterface& intf, uint32_t per_tc) {
    for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb_dm_step_slot<is_write>(intf, intf.tc_slots[i], per_tc);
    }
}

// One op is always this hart's whole share of a counter visit (asserted by the caller), so every
// op hands off: step this counter's bookmark past the op and rotate to the next counter, whose
// bookmark is already waiting exactly where the stream continues.
template <bool is_write>
inline void dfb_dm_advance_slot(LocalDFBInterface& intf, uint32_t n) {
    dfb_dm_step_slot<is_write>(intf, intf.tc_slots[intf.tc_idx], n);
    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

// BROADCAST producer: every counter received all n entries, so only slot 0's bookmark is kept.
// Step it past the n contiguous entries (wrapping to base); tc_idx stays at 0.
inline void dfb_dm_broadcast_advance(LocalDFBInterface& intf, uint32_t n) {
    DFBTCSlot& slot = intf.tc_slots[0];
    slot.wr_ptr += n * intf.stride_size;
    if (slot.wr_ptr >= slot.limit) {
        slot.wr_ptr = slot.base_addr;
    }
}
#endif  // !COMPILE_FOR_TRISC

#if defined(COMPILE_FOR_TRISC)
// Tiles one op puts on / takes from a single counter: the whole op, or its share of it on a split hart.
inline uint32_t dfb_trisc_per_counter(const LocalDFBInterface& intf, uint32_t n) {
    return intf.split_tc ? n / intf.num_tcs_to_rr : n;
}
#endif

inline uint32_t dfb_ring_span_address_units(const LocalDFBInterface& intf) {
    const uint8_t last = static_cast<uint8_t>(intf.num_tcs_to_rr - 1);
#if defined(COMPILE_FOR_TRISC)
    const auto& first = intf.tc_slots[0];
    const auto& last_slot = intf.tc_slots[last];
    return (last_slot.base_addr + last_slot.ring_size) - first.base_addr;
#else
    return intf.tc_slots[last].limit - intf.tc_slots[0].base_addr;
#endif
}
#endif  // !DFB_IS_COMPUTE_MATH

}  // namespace

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_ring_span_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_));
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_ring_span_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    const uint32_t entry_bytes = get_entry_size();
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_)) / entry_bytes;
#endif
}

// The stride field as serialized for this hart, in entries. On a STRIDED side (and on every Tensix
// hart) it is the spacing between the entries the hart touches in turn; on a BLOCKED DM side it is
// the per-tile hop its cursor arithmetic uses to land on its next block, not a spacing inside the
// block. Only the share getters below and the cursor code should read it.
#ifndef COMPILE_FOR_TRISC
template <DFBAccess Pap, DFBAccess Cap>
inline bool DataflowBuffer<Pap, Cap>::producer_broadcast() const {
    if constexpr (kKnown) {
        // The host sets the wire flag only for a DM -> DM ALL ring (one counter per consumer). A
        // DM -> Tensix ALL producer owns a single counter that the remapper fans out, and with one
        // counter the broadcast arm IS the plain one-counter post (jump == stride, so the bookmark
        // math agrees too). Pin that layout so a host change cannot desync the two.
        if constexpr (Cap == DFBAccess::ALL) {
            ASSERT(local_dfb_interface_.broadcast_tc || local_dfb_interface_.num_tcs_to_rr == 1);
        }
        return Cap == DFBAccess::ALL;
    } else {
        return local_dfb_interface_.broadcast_tc != 0;
    }
}
#endif

template <DFBAccess Pap, DFBAccess Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::wire_stride_tiles() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#elif defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.stride_size_tiles;
#else
    return static_cast<uint16_t>(dfb_dm_stride_entries(local_dfb_interface_));
#endif
}

// Spacing between two consecutive entries of one op's share. A BLOCKED side's share is a whole
// block and so contiguous whatever hop the wire stride encodes for its cursor; only a STRIDED side
// of a mixed ring walks a share whose entries sit the wire stride apart.
template <DFBAccess Pap, DFBAccess Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_produce_stride_tiles() const {
    if constexpr (kProducerBlocked) {
        return 1u;
    } else {
        return wire_stride_tiles();
    }
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_consume_stride_tiles() const {
    if constexpr (kConsumerBlocked) {
        return 1u;
    } else {
        return wire_stride_tiles();
    }
}

// The share is derived, not sent: the ring's block_size and this hart's stride say how many of
// the block's tiles this hart handles in one counter visit. UNKNOWN (raw-id) DataflowBuffers only
// support rings where nothing is BLOCKED.
template <DFBAccess Pap, DFBAccess Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_produce_share() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#else
    const uint32_t block = local_dfb_interface_.block_size;
    const uint32_t stride = wire_stride_tiles();
    if constexpr (kProducerBlocked) {
        return static_cast<uint16_t>(block);                                   // whole block per op
    } else if constexpr (kConsumerBlocked) {
        return static_cast<uint16_t>((block > stride) ? block / stride : 1u);  // my part of each consumer block
    } else {
        return 1u;
    }
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_consume_share() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#else
    const uint32_t block = local_dfb_interface_.block_size;
    const uint32_t stride = wire_stride_tiles();
    if constexpr (kConsumerBlocked) {
        return static_cast<uint16_t>(block);                                   // whole block per op
    } else if constexpr (kProducerBlocked) {
        return static_cast<uint16_t>((block > stride) ? block / stride : 1u);  // my part of each producer block
    } else {
        return 1u;
    }
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::reserve_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("RBW");
    dfb::PackedTileCounter packed_tc =
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_produce_share() || !kShareStrict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_wait_for_free_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (producer_broadcast()) {
        // BROADCAST: every consumer reads every entry, so every counter must have room for all of them.
        dfb_dm_all_tc_wait<true>(local_dfb_interface_, num_entries);
    } else if (producer_split()) {
        // SPLIT: the block belongs to every counter -- wait for each one's share.
        dfb_dm_all_tc_wait<true>(local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, num_entries));
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (overlay::fast_llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
    }
#endif
    WAYPOINT("RBD");
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::push_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_produce_share() || !kShareStrict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_push_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (producer_broadcast()) {
        // BROADCAST: post the full count to every counter (every consumer reads every entry).
        dfb_dm_all_tc_credit<true>(local_dfb_interface_, num_entries);
        dfb_dm_broadcast_advance(local_dfb_interface_, num_entries);
    } else if (producer_split()) {
        // SPLIT: post each counter its share of the block and step every bookmark past it.
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, num_entries);
        dfb_dm_all_tc_credit<true>(local_dfb_interface_, per_tc);
        dfb_dm_all_slots_advance<true>(local_dfb_interface_, per_tc);
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        dfb_dm_advance_slot</*is_write=*/true>(local_dfb_interface_, num_entries);
    }
#endif
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::wait_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("WFW");
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_consume_share() || !kShareStrict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_wait_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (consumer_split()) {
        // SPLIT: the block belongs to every counter -- wait for each one's share.
        dfb_dm_all_tc_wait<false>(local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, num_entries));
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (overlay::fast_llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
    }
#endif
    WAYPOINT("WFD");
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::pop_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_consume_share() || !kShareStrict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_pop_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (consumer_split()) {
        // SPLIT: ack each counter its share of the block and step every bookmark past it.
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, num_entries);
        dfb_dm_all_tc_credit<false>(local_dfb_interface_, per_tc);
        dfb_dm_all_slots_advance<false>(local_dfb_interface_, per_tc);
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::fast_llk_intf_inc_acked(tensix_id, tc_id, num_entries);
        dfb_dm_advance_slot</*is_write=*/false>(local_dfb_interface_, num_entries);
    }
#endif
#endif
}

#if !defined(COMPILE_FOR_TRISC)
template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::wait_relay_consumer_caught_up() const {
    // Same posted==acked drain as finish()'s DM path; scoped for PrefetcherPipe relay handoff.
    bool all_acked = false;
    while (!all_acked) {
        all_acked = true;
        for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
            const dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
            const uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
            const uint8_t tc_id = dfb::get_counter_id(packed_tc);
            if (overlay::fast_llk_intf_read_acked(tensix_id, tc_id) !=
                overlay::fast_llk_intf_read_posted(tensix_id, tc_id)) {
                all_acked = false;
            }
        }
    }
}
#endif

template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::finish_impl() {
#if !DFB_IS_COMPUTE_MATH
#ifndef COMPILE_FOR_TRISC
    // A STRIDED side's implicit-sync share must be complete: finish() mid-share means the kernel
    // issued a number of entries that is not a multiple of the share.
    ASSERT(pshare_pos_ == 0 && cshare_pos_ == 0);
    if (ptiles_read_ > 0) {
        handle_final_credits<true>(ptiles_read_, ptxn_id_index_);
    }
    if (ctiles_written_ > 0) {
        handle_final_credits<false>(ctiles_written_, ctxn_id_index_);
    }
#endif
    bool all_acked = false;
    WAYPOINT("AAW");
    while (!all_acked) {
        all_acked = true;
        for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
            dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
            uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK))
            // TRISC drain: finish() must not return until this TC is empty (posted == 0).
            // On TRISC, tile_counters[].f.posted/.acked are live occupancy / free-space
            // (tiles-available / space-available), NOT the cumulative read_posted/read_acked
            // totals used by the DM overlay path below. The consumer also skips TCs this TRISC
            // doesn't own via tensix_trisc_mask, which exists only in the UNPACK-side
            // LocalDFBInterface, so the gate sits under an inner UNPACK guard (the PACK struct
            // has no such member).
#ifdef UCK_CHLKC_UNPACK
            if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
                continue;
            }
#endif
            const uint32_t tiles_avail = ckernel::trisc::tile_counters[tc_id].f.posted & 0xFFFFu;
            if (tiles_avail != 0) {
                all_acked = false;
            }
#elif !defined(COMPILE_FOR_TRISC)
            uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
            const uint32_t read_posted = overlay::fast_llk_intf_read_posted(tensix_id, tc_id);
            const uint32_t read_acked = overlay::fast_llk_intf_read_acked(tensix_id, tc_id);
            if (read_acked != read_posted) {
                all_acked = false;
            }
#endif
        }
    }
    WAYPOINT("AAD");
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_write_ptr_impl() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#elif defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    {
        const auto& slot = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
        return slot.base_addr + dfb_slot_cursor_offset_units(local_dfb_interface_, slot, slot.wr_entry_idx);
    }
#elif !defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr;
#else
    // Unpack TRISC does not use wr_ptr; return ring base for any accidental caller.
    ASSERT(false);
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
#endif
}

template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_read_ptr_impl() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#elif defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    {
        const auto& slot = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
        return slot.base_addr + dfb_slot_cursor_offset_units(local_dfb_interface_, slot, slot.rd_entry_idx);
    }
#elif !defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr;
#else
    // Pack TRISC does not use rd_ptr; return ring base for any accidental caller.
    ASSERT(false);
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
#endif
}

#ifdef COMPILE_FOR_TRISC
template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_tile_address(uint32_t tile_index) {
    uint32_t address = 0;
#if defined(UCK_CHLKC_UNPACK)
    {
        const auto& slot = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
        // Linear (front + tile_index * stride), no wrap. Safe because wait_front(n) must
        // not straddle slot.limit; tile_index is in [0, n).
        const uint32_t base_address =
            slot.base_addr + dfb_slot_cursor_offset_units(local_dfb_interface_, slot, slot.rd_entry_idx);
        const uint32_t offset_address = static_cast<uint32_t>(local_dfb_interface_.stride_size) * tile_index;
        address = address_units_to_bytes(base_address + offset_address);
        mailbox_write(ckernel::ThreadId::MathThreadId, address);
        mailbox_write(ckernel::ThreadId::PackThreadId, address);
        mailbox_write(ckernel::ThreadId::IsolateSfpuThreadId, address);
    }
#elif defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_ISOLATE_SFPU)
    address = mailbox_read(ckernel::ThreadId::UnpackThreadId);
#endif
    return address;
}

template <DFBAccess Pap, DFBAccess Cap>
template <typename T>
T DataflowBuffer<Pap, Cap>::read_tile_value(uint32_t tile_index, uint32_t element_offset) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "read_tile_value: T must be 1, 2, or 4 bytes");
    static_assert(
        (std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>),
        "read_tile_value: T must be an unsigned integral type");

    T value = T{};
#if defined(UCK_CHLKC_UNPACK)
    {
        const auto& slot = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
        // Same linear addressing as get_tile_address: the wait_front window does not wrap.
        const uint32_t base_address =
            slot.base_addr + dfb_slot_cursor_offset_units(local_dfb_interface_, slot, slot.rd_entry_idx);
        const uint32_t offset_address = static_cast<uint32_t>(local_dfb_interface_.stride_size) * tile_index;
        const uint32_t byte_address = address_units_to_bytes(base_address + offset_address);
        value = reinterpret_cast<volatile T*>(byte_address)[element_offset];
        mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(value));
        mailbox_write(ckernel::ThreadId::PackThreadId, static_cast<uint32_t>(value));
        mailbox_write(ckernel::ThreadId::IsolateSfpuThreadId, static_cast<uint32_t>(value));
    }
#elif defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_ISOLATE_SFPU)
    value = static_cast<T>(mailbox_read(ckernel::ThreadId::UnpackThreadId));
#endif
    return value;
}
#endif  // COMPILE_FOR_TRISC

#ifndef COMPILE_FOR_TRISC
template <DFBAccess Pap, DFBAccess Cap>
template <bool is_producer>
inline void DataflowBuffer<Pap, Cap>::handle_final_credits(uint32_t tiles_issued, uint8_t txn_id_index) {
    // Determine the txn_id for the last batch. If tiles_issued lands exactly on
    // a boundary, txn_id_index has already wrapped past it, so step back one slot.
    uint8_t tail_txn_idx = (tiles_issued % local_dfb_interface_.num_entries_per_txn_id == 0)
                                ? static_cast<uint8_t>((txn_id_index + local_dfb_interface_.num_txn_ids - 1) % local_dfb_interface_.num_txn_ids)
                                : txn_id_index;
    uint8_t tail_txn_id = local_dfb_interface_.txn_ids[tail_txn_idx];

    uint8_t N = local_dfb_interface_.num_tcs_to_rr;
    dfb::PackedTileCounter ptc0 = local_dfb_interface_.tc_slots[0].packed_tile_counter;
    // finish() cannot return until the ISR has credited every tile this hart issued. To know
    // what to wait for, compute how many tiles each counter should have been credited. Every op
    // is one whole share (`visit` tiles): normally the hart fills one counter with a share, then
    // the next, wrapping around all N; under SPLIT every counter gets 1/N of every op; under
    // BROADCAST every counter gets all of every op.
    // Computed in 32 bits from the exact running total, then reduced to the HW counter's 16 bits
    // for the modular comparison below -- so a launch moving more than 65535 tiles still agrees
    // with the hardware.
    const uint32_t visit = is_producer ? get_produce_share() : get_consume_share();
    const bool split = is_producer ? producer_split() : consumer_split();
    const bool broadcast = is_producer && producer_broadcast();
    const uint32_t NV = N * visit;
    auto expected_for_slot = [&](uint8_t i) -> uint16_t {
        if (broadcast) {
            return static_cast<uint16_t>(tiles_issued);
        }
        if (split) {
            return static_cast<uint16_t>(tiles_issued / N);
        }
        const uint32_t full = (tiles_issued / NV) * visit;
        const uint32_t rem = tiles_issued % NV;
        const uint32_t start = i * visit;
        uint32_t part = 0u;
        if (rem > start) {
            const uint32_t into_slot = rem - start;
            part = (into_slot > visit) ? visit : into_slot;
        }
        return static_cast<uint16_t>(full + part);
    };
    const uint16_t expected_slot0 = expected_for_slot(0);

    auto read_actual_slot0 = [&]() -> uint16_t {
        if constexpr (is_producer) {
            return static_cast<uint16_t>(
                overlay::fast_llk_intf_read_posted(dfb::get_tensix_id(ptc0), dfb::get_counter_id(ptc0)));
        } else {
            return static_cast<uint16_t>(
                overlay::fast_llk_intf_read_acked(dfb::get_tensix_id(ptc0), dfb::get_counter_id(ptc0)));
        }
    };

    // Wait until this DM's tail transactions have been picked up by the NoC.
    // A transaction passes through three observable states:
    //   not dispatched → tack == 0, tiles == 0
    //   in-flight      → tack >  0
    //   completed      → tack == 0, tiles >  0   ← break here
    // Also exits early if the ISR fires (collective batch done).
    WAYPOINT("WTP1");
    // Modular comparison: read_actual_slot0() and expected_slot0 are both
    // uint16; their wrapped difference interpreted as int16 is negative when
    // actual is "behind" expected. See prepare_implicit_read for rationale.
    while (static_cast<int16_t>(read_actual_slot0() - expected_slot0) < 0) {
        uint64_t tack, tiles;
        if constexpr (is_producer) {
            tack  = __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(OVERLAY_RD_CMD_BUF, tail_txn_id);
            tiles = __builtin_riscv_ttrocc_cmdbuf_read_tiles_to_process_tr_ack_tr_id(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tack  = __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(OVERLAY_WR_CMD_BUF, tail_txn_id);
            tiles = __builtin_riscv_ttrocc_cmdbuf_read_tiles_to_process_wr_sent_tr_id(OVERLAY_WR_CMD_BUF, tail_txn_id);
        }
        if (tack == 0 && tiles > 0) {
            break;
        }
    }

    // Rendezvous: every participating DM has now issued its tail transaction and seen
    // the NoC pick it up. This must be unconditional — gating the barrier on
    // read_actual_slot0() < expected_slot0 is racy because the ISR can fire between
    // different threads' checks, causing some to enter the barrier and others to skip
    // it. Once past this point, tiles_to_process on the tail txn_id reflects the
    // contributions of all producers / consumers for this collective batch.
    // Producer and consumer kernels co-reside with different thread counts, so each must
    // rendezvous on its own barrier — sharing one deadlocks. They are distinct kernels and every
    // kernel gets its own barrier slot, so plain sync_threads() already keeps them apart.
    sync_threads();

    // ISR already handled the collective batch — modular check (see WTP1).
    if (static_cast<int16_t>(read_actual_slot0() - expected_slot0) >= 0) {
        return;
    }

    // Spin giving the ISR a chance to fire. Break when the tail txn_id's tiles_to_process
    // is a genuine partial batch (below the global ISR-programmed threshold). The ISR will
    // never post credits for it, so we fall through to the manual posting below.
    uint16_t global_threshold = local_dfb_interface_.threshold;
    WAYPOINT("WTP2");
    while (static_cast<int16_t>(read_actual_slot0() - expected_slot0) < 0) {
        uint64_t tiles;
        if constexpr (is_producer) {
            tiles = __builtin_riscv_ttrocc_cmdbuf_read_tiles_to_process_tr_ack_tr_id(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tiles = __builtin_riscv_ttrocc_cmdbuf_read_tiles_to_process_wr_sent_tr_id(OVERLAY_WR_CMD_BUF, tail_txn_id);
        }
        if (tiles > 0 && tiles < global_threshold) {
            break;
        }
    }

    // Manually post missing credits if ISR did not fire.
    // Modular: int16(actual - expected) < 0 means actual is behind expected,
    // and the unsigned difference (expected - actual) is the number of missing
    // increments — correct across the uint16 wrap because both operands wrap.
    uint16_t actual_slot0 = read_actual_slot0();
    if (static_cast<int16_t>(actual_slot0 - expected_slot0) < 0) {
        for (uint8_t i = 0; i < N; i++) {
            dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
            uint8_t tensix_id = dfb::get_tensix_id(ptc);
            uint8_t tc_id     = dfb::get_counter_id(ptc);
            uint16_t expected = expected_for_slot(i);
            if constexpr (is_producer) {
                // Modular int16 comparison: posted (16-bit HW) wraps at 65 536, so
                // `actual < expected` is wrong at wrap; cast the difference to int16_t
                // to get a signed modular distance. Negative = behind, ≥0 = caught up.
                uint16_t actual = static_cast<uint16_t>(overlay::fast_llk_intf_read_posted(tensix_id, tc_id));
                if (static_cast<int16_t>(actual - expected) < 0) {
                    overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, static_cast<uint16_t>(expected - actual));
                }
            } else {
                uint16_t actual = static_cast<uint16_t>(overlay::fast_llk_intf_read_acked(tensix_id, tc_id));
                if (static_cast<int16_t>(actual - expected) < 0) {
                    overlay::fast_llk_intf_inc_acked(tensix_id, tc_id, static_cast<uint16_t>(expected - actual));
                }
            }
        }
    }
}


// Lock the `n` held entries. The locked region starts at the write pointer (scoped_write_lock) or the
// read pointer (scoped_read_lock), with entries spaced by stride_size: for the ALL access pattern
// stride_size == entry_size, so the locked entries are contiguous; for STRIDED stride_size > entry_size,
// so they are non-contiguous. For each held entry, do two things:
//     - cache op: invalidate the L2 range on acquire (both lock kinds); flush on release (write lock
//       only)
//     - record the scoped-lock event
template <DFBAccess Pap, DFBAccess Cap>
template <bool is_write>
inline typename DataflowBuffer<Pap, Cap>::ScopedLockRegion DataflowBuffer<Pap, Cap>::lock_acquire_impl(uint16_t num_entries) {
    // A lock covers this hart's whole share (or n consecutive entries on a plain ring): its entries
    // sit one stride_tiles apart from the cursor (contiguous on a BLOCKED side, `stride` apart on the
    // STRIDED side of a mixed ring), the same spacing the implicit-sync path and in-order pack use.
    ASSERT(num_entries == (is_write ? get_produce_share() : get_consume_share()) || !kShareStrict);
    const auto& s = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
    const uint32_t entry = local_dfb_interface_.entry_size;
    const uint32_t stride = entry * (is_write ? get_produce_stride_tiles() : get_consume_stride_tiles());
    // Snapshot the start pointer + this slot's wrap bounds so release replays the identical walk.
    const ScopedLockRegion region{is_write ? s.wr_ptr : s.rd_ptr, s.base_addr, s.limit};
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        RECORD_SCOPED_LOCK_EVENT(NocDebuggingEventMetadata::NocDebugEventType::DFB_LOCK, addr, entry);
        // TODO: with concurrent ALL consumers, this invalidates the same shared cache line once per
        // consumer; the redundant invalidations could be deduplicated (e.g. first-locker-per-round).
        // Currently this invalidates the L2 range, which also drops the matching L1 D$ line on all
        // DM cores.
        scoped_lock_acquire_cache_ops(addr, entry);
        addr += stride;
        if (addr >= region.limit) {
            addr = region.base;
        }
    }
    return region;
}

template <DFBAccess Pap, DFBAccess Cap>
template <bool is_write>
inline void DataflowBuffer<Pap, Cap>::lock_release_impl(ScopedLockRegion region, uint16_t num_entries) {
    const uint32_t entry = local_dfb_interface_.entry_size;
    const uint32_t stride = entry * (is_write ? get_produce_stride_tiles() : get_consume_stride_tiles());
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        // Flush on release only for a write lock. A read lock never writes.
        if constexpr (is_write) {
            // Currently this flushes l2, which writes back + drops the matching L1 D$ line on all DM cores.
            scoped_lock_release_cache_ops(addr, entry);
        }
        RECORD_SCOPED_LOCK_EVENT(NocDebuggingEventMetadata::NocDebugEventType::DFB_UNLOCK, addr, entry);
        addr += stride;
        if (addr >= region.limit) {
            addr = region.base;
        }
    }
}

// Consumer barrier: waits outbound write from DFB writes to arrive at their destination
// Falls back to a full barrier when no txn_ids are assigned
template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::write_barrier_impl(const Noc &noc) const {
    if (local_dfb_interface_.num_txn_ids == 0) {
        noc.async_write_barrier();
        return;
    } else {
        for (uint8_t i = 0; i < local_dfb_interface_.num_txn_ids; i++) {
            // Uses internal API rather than user facing noc.async_write_barrier() since it ASSERTs that the txn_id comes
            // from the user tnx ID pool and the DFB txn ids are internal only.
            noc_async_write_barrier_with_trid(local_dfb_interface_.txn_ids[i], noc.get_noc_id());
        }
    }
}

// Preamble for implicit-sync read: spin until previous reads are posted and there is space in the tile counters.
// Returns the txn_id to stamp on the next NOC read.
template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::prepare_implicit_read(uint32_t num_tiles) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    const uint32_t txn_id = local_dfb_interface_.txn_ids[ptxn_id_index_];
    WAYPOINT("PIRW");
    // Modular comparison: posted (16-bit HW) wraps at 65 536, and so does the
    // kernel-side expectation `ptxn_id_loop_cnt_ * per_tc` if both operands are
    // reduced to uint16 before subtracting. Interpreting the wrapped difference
    // as int16 gives a signed modular distance — negative means posted is behind
    // expected, non-negative means it has caught up. Safe as long as the gap
    // never exceeds half the wrap range (~32K), which is guaranteed by the
    // bounded txn-id ring depth.
    while (static_cast<int16_t>(
        static_cast<uint16_t>(overlay::fast_llk_intf_read_posted(tensix_id, tc_id)) -
        static_cast<uint16_t>(ptxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc)) < 0);
    // One op is this hart's whole share; wait for room for all of it. The HW free-space register
    // only discounts reads that have already POSTED, so a read this hart issued but whose ISR credit
    // is still pending would look like a free slot: count this hart's own issues instead (see
    // dfb_dm_implicit_wait). Under BROADCAST every consumer's counter must have room for the share;
    // under SPLIT each counter takes its part; otherwise only this round's counter matters, which
    // holds one share per full round so far.
    ASSERT(num_tiles == get_produce_share());
    const uint8_t num_tcs = local_dfb_interface_.num_tcs_to_rr;
    if (producer_broadcast()) {
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<true>(local_dfb_interface_, i, static_cast<uint16_t>(ptiles_read_), num_tiles);
        }
    } else if (producer_split()) {
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(num_tiles));
        const uint16_t mine = static_cast<uint16_t>(ptiles_read_ / num_tcs);
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<true>(local_dfb_interface_, i, mine, per_tc);
        }
    } else {
        const uint32_t round = static_cast<uint32_t>(num_tcs) * num_tiles;
        const uint16_t mine = static_cast<uint16_t>((ptiles_read_ / round) * num_tiles);
        dfb_dm_implicit_wait<true>(local_dfb_interface_, local_dfb_interface_.tc_idx, mine, num_tiles);
    }
    WAYPOINT("PIRD");
    return txn_id;
}

// Postamble for implicit-sync read: advance wr_ptr, tile/txn counters, and tc_idx.
template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::commit_implicit_read(uint32_t num_tiles) {
    // Runs once per op (a whole share); the ISR posts the credits, this only moves the bookmarks.
    const uint32_t block = num_tiles;
    if (producer_broadcast()) {
        dfb_dm_broadcast_advance(local_dfb_interface_, block);
    } else if (producer_split()) {
        dfb_dm_all_slots_advance<true>(
            local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(block)));
    } else {
        dfb_dm_advance_slot</*is_write=*/true>(local_dfb_interface_, block);
    }
    ptiles_read_ += block;
    if (ptiles_read_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ptxn_id_index_ = (ptxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ptxn_id_loop_cnt_++;
    }
}

// Preamble for implicit-sync write: spin until previous writes are acked and data is available in the tile counters.
// Returns the txn_id to stamp on the next NOC write.
template <DFBAccess Pap, DFBAccess Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::prepare_implicit_write(uint32_t num_tiles) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    const uint32_t txn_id = local_dfb_interface_.txn_ids[ctxn_id_index_];
    WAYPOINT("PIWW");
    // Modular comparison — see prepare_implicit_read for the rationale. Same
    // trick applied to the acked side.
    while (static_cast<int16_t>(
        static_cast<uint16_t>(overlay::fast_llk_intf_read_acked(tensix_id, tc_id)) -
        static_cast<uint16_t>(ctxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc)) < 0);
    // One op is this hart's whole share; wait until all of it is there. The HW occupancy register
    // still counts entries this hart has claimed but whose ACK is batched and pending, so it could
    // hand the same posted entry out twice: count this hart's own claims instead (see
    // dfb_dm_implicit_wait). Under SPLIT each counter supplies its part; otherwise only this round's
    // counter matters, which has supplied one share per full round so far.
    ASSERT(num_tiles == get_consume_share());
    const uint8_t num_tcs = local_dfb_interface_.num_tcs_to_rr;
    if (consumer_split()) {
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(num_tiles));
        const uint16_t mine = static_cast<uint16_t>(ctiles_written_ / num_tcs);
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<false>(local_dfb_interface_, i, mine, per_tc);
        }
    } else {
        const uint32_t round = static_cast<uint32_t>(num_tcs) * num_tiles;
        const uint16_t mine = static_cast<uint16_t>((ctiles_written_ / round) * num_tiles);
        dfb_dm_implicit_wait<false>(local_dfb_interface_, local_dfb_interface_.tc_idx, mine, num_tiles);
    }
    WAYPOINT("PIWD");
    return txn_id;
}

// Postamble for implicit-sync write: advance rd_ptr, tile/txn counters, and tc_idx.
template <DFBAccess Pap, DFBAccess Cap>
inline void DataflowBuffer<Pap, Cap>::commit_implicit_write(uint32_t num_tiles) {
    // Runs once per op (a whole share); the ISR acks the credits, this only moves the bookmarks.
    const uint32_t block = num_tiles;
    if (consumer_split()) {
        dfb_dm_all_slots_advance<false>(
            local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(block)));
    } else {
        dfb_dm_advance_slot</*is_write=*/false>(local_dfb_interface_, block);
    }
    ctiles_written_ += block;
    if (ctiles_written_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ctxn_id_index_ = (ctxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ctxn_id_loop_cnt_++;
    }
}

// Out-of-line definitions of Noc DFB-specific implicit-sync overloads.
// These are member functions of Noc but must be defined here because they need the complete
// DataflowBuffer type (circular dependency: dataflow_buffer.h includes noc.h, not vice versa).

template <NocOptions opts, typename Src, DFBAccess Pap, DFBAccess Cap>
std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_read(
    const Src& src,
    DataflowBuffer<Pap, Cap>& dst,
    const typename noc_traits_t<Src>::src_args_type& src_args,
    const DataflowBufferArgs& dst_args) const {
    // Implicit sync lands data at the bookmark and moves it by whole entries; offset_bytes is
    // ignored, so a non-zero offset would land data in the wrong place while still posting
    // whole-entry credits.
    ASSERT(dst_args.offset_bytes == 0);
    const uint32_t entry_bytes = dst.get_entry_size();
    const uint32_t share = dst.get_produce_share();
    auto issue = [&](uint32_t txn_id, uint32_t l1_addr, uint32_t bytes) {
        noc_async_read_set_trid(txn_id, noc_id_);
        while (noc_available_transactions(noc_id_, txn_id) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));
        noc_async_read<NOC_MAX_BURST_SIZE + 1, true>(
            get_src_ptr<AddressType::NOC>(src, src_args), l1_addr, bytes, noc_id_, NOC_UNICAST_WRITE_VC);
    };
    if constexpr (Pap == DFBAccess::BLOCKED) {
        // A BLOCKED producer moves its whole block per call: contiguous in L1 and in the tensor, one
        // NoC transaction (the host sizes the implicit-sync ISR threshold for one packet per block).
        const uint32_t txn_id = dst.prepare_implicit_read(share);
        issue(txn_id, dst.get_noc_write_addr(), entry_bytes * share);
        dst.commit_implicit_read(share);
    } else {
        // A STRIDED producer moves one entry per call (one NoC transaction, which is how the host
        // sizes the ISR threshold for this side) and the DFB completes the share itself: the first
        // entry waits for room for the whole share and fixes the txn id, entry i lands at
        // bookmark + i * stride, the last entry moves the bookmark. The kernel only supplies the
        // page of each entry, exactly as on a plain ring.
        if (dst.pshare_pos_ == 0) {
            dst.pshare_txn_id_ = dst.prepare_implicit_read(share);
        }
        const uint32_t stride_bytes = entry_bytes * dst.get_produce_stride_tiles();
        issue(dst.pshare_txn_id_, dst.get_noc_write_addr() + dst.pshare_pos_ * stride_bytes, entry_bytes);
        if (++dst.pshare_pos_ == share) {
            dst.pshare_pos_ = 0;
            dst.commit_implicit_read(share);
        }
    }
}

template <NocOptions opts, typename Dst, DFBAccess Pap, DFBAccess Cap>
std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_write(
    DataflowBuffer<Pap, Cap>& src,
    const Dst& dst,
    const DataflowBufferArgs& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    // Same contract as async_read above, from the consumer side: a BLOCKED consumer drains its
    // whole block per call, a STRIDED consumer one entry per call with the DFB completing the
    // share; offset_bytes is ignored.
    ASSERT(src_args.offset_bytes == 0);
    const uint32_t entry_bytes = src.get_entry_size();
    const uint32_t share = src.get_consume_share();
    auto issue = [&](uint32_t txn_id, uint32_t src_addr, uint32_t bytes) {
        const uint64_t dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, bytes, -1, false, noc_id_);
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_noc_addr, src_addr, bytes);
        ncrisc_noc_fast_write_any_len<noc_mode, true, /*one_packet*/ false>(
            noc_id_,
            write_cmd_buf,
            src_addr,
            dst_noc_addr,
            bytes,
            NOC_UNICAST_WRITE_VC,
            /*mcast*/ false,
            /*linked*/ false,
            /*num_dests*/ 1,
            /*multicast_path_reserve*/ true,
            /*posted*/ false,
            txn_id);
    };
    if constexpr (Cap == DFBAccess::BLOCKED) {
        const uint32_t txn_id = src.prepare_implicit_write(share);
        issue(txn_id, src.get_noc_read_addr(), entry_bytes * share);
        src.commit_implicit_write(share);
    } else {
        if (src.cshare_pos_ == 0) {
            src.cshare_txn_id_ = src.prepare_implicit_write(share);
        }
        const uint32_t stride_bytes = entry_bytes * src.get_consume_stride_tiles();
        issue(src.cshare_txn_id_, src.get_noc_read_addr() + src.cshare_pos_ * stride_bytes, entry_bytes);
        if (++src.cshare_pos_ == share) {
            src.cshare_pos_ = 0;
            src.commit_implicit_write(share);
        }
    }
}

#else  // COMPILE_FOR_TRISC

template <DFBAccess Pap, DFBAccess Cap>
template <bool>
inline typename DataflowBuffer<Pap, Cap>::ScopedLockRegion DataflowBuffer<Pap, Cap>::lock_acquire_impl(uint16_t) { return {}; }
template <DFBAccess Pap, DFBAccess Cap>
template <bool>
inline void DataflowBuffer<Pap, Cap>::lock_release_impl(ScopedLockRegion, uint16_t) {}

#endif  // !COMPILE_FOR_TRISC

#endif  // ARCH_QUASAR
