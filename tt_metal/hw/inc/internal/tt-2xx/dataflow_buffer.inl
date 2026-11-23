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
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline DataflowBuffer<Pap, Cap>::DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
}
#else
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline DataflowBuffer<Pap, Cap>::DataflowBuffer(uint16_t logical_dfb_id)
    : logical_dfb_id_(logical_dfb_id), local_dfb_interface_(get_local_dfb_interface(logical_dfb_id)) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
    if constexpr (!pattern_known) {
        // UNKNOWN (raw id) only supports rings with no BLOCKED side.
        ASSERT(local_dfb_interface_.block_size <= 1u);
    } else {
        stride_tiles_cache_ = wire_stride_tiles();
        if constexpr (producer_blocked != consumer_blocked) {
            // Exactly one BLOCKED side: the other side moves block_size / stride entries per op.
            const uint32_t block = local_dfb_interface_.block_size;
            peer_share_cache_ =
                static_cast<uint16_t>((block > stride_tiles_cache_) ? block / stride_tiles_cache_ : 1u);
        }
    }
    // Declare this DFB's L1 extent to the NOC-debug tracker so a write into it without holding the
    // lock can be flagged.
    RECORD_SCOPED_LOCK_EVENT(
        NocDebuggingEventMetadata::NocDebugEventType::DFB_REGION_START,
        address_units_to_bytes(local_dfb_interface_.tc_slots[0].base_addr),
        get_ring_span_bytes());
}
#endif

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_entry_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_stride_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.stride_size);
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_total_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return local_dfb_interface_.num_entries;
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_total_size_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return get_total_num_entries() * address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
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

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
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

#if !DFB_IS_COMPUTE_MATH && !defined(UCK_CHLKC_ISOLATE_SFPU)

#ifndef COMPILE_FOR_TRISC
// The stride in entries (the host sends it in bytes).
inline uint32_t dfb_dm_stride_entries(const LocalDFBInterface& intf) { return intf.stride_size / intf.entry_size; }

// Split op: n tiles shared evenly over this hart's counters.
inline uint16_t dfb_dm_per_tc(const LocalDFBInterface& intf, uint16_t n) {
    ASSERT(n % intf.num_tcs_to_rr == 0);
    return static_cast<uint16_t>(n / intf.num_tcs_to_rr);
}

// Implicit sync: one op of `block` entries is done, so move this DM's place in the txn-id batches.
inline void dfb_implicit_advance(const LocalDFBInterface& intf, DFBImplicitProgress& progress, uint32_t block) {
    progress.share = static_cast<uint8_t>(block);
    progress.pos = static_cast<uint8_t>(progress.pos + block);
    ASSERT(progress.pos <= intf.num_entries_per_txn_id);
    if (progress.pos == intf.num_entries_per_txn_id) {
        progress.pos = 0;
        progress.loop++;
        progress.txn_idx = (progress.txn_idx + 1u == intf.num_txn_ids) ? 0 : progress.txn_idx + 1u;
    }
}

// Wait until every counter has room for (producer) or holds (consumer) per_tc tiles.
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

// Wait until this counter has room (producer) or data (consumer) for `need` more entries.
// The HW free-space / occupancy value is not used directly: it only updates when the ISR lands the
// credits, so entries this hart just sent or took would still look free. `mine` is this hart's own
// count of those, and the check is done against it.
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

// Post (producer) or ack (consumer) per_tc credits on every counter.
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

// Move one counter's cursor past an n-tile op: to the op's last tile, then `jump` bytes to
// this counter's next tile. Wraps to the ring base at the end.
inline void dfb_dm_step_slot(const LocalDFBInterface& intf, DFBTCSlot& slot, uint32_t n) {
    slot.ptr += (n - 1u) * intf.stride_size + intf.jump;
    if (slot.ptr >= slot.limit) {
        slot.ptr = slot.base_addr;
    }
}

// Split op: every counter took part (per_tc tiles each), so step every cursor and stay on
// counter 0.
inline void dfb_dm_all_slots_advance(LocalDFBInterface& intf, uint32_t per_tc) {
    for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb_dm_step_slot(intf, intf.tc_slots[i], per_tc);
    }
}

// Normal op: step the current counter's cursor, then move on to the next counter.
inline void dfb_dm_advance_slot(LocalDFBInterface& intf, uint32_t n) {
    dfb_dm_step_slot(intf, intf.tc_slots[intf.tc_idx], n);
    intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
}

// Broadcast op: every counter got all n entries; only counter 0's cursor is used.
inline void dfb_dm_broadcast_advance(LocalDFBInterface& intf, uint32_t n) {
    DFBTCSlot& slot = intf.tc_slots[0];
    slot.ptr += n * intf.stride_size;
    if (slot.ptr >= slot.limit) {
        slot.ptr = slot.base_addr;
    }
}
#endif  // !COMPILE_FOR_TRISC

#if defined(COMPILE_FOR_TRISC)
// Tiles one op moves on a single counter (the whole op, or 1/N of it when split).
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
#endif  // !DFB_IS_COMPUTE_MATH && !defined(UCK_CHLKC_ISOLATE_SFPU)

}  // namespace

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_ring_span_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_));
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_ring_span_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    const uint32_t entry_bytes = get_entry_size();
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_)) / entry_bytes;
#endif
}

#ifndef COMPILE_FOR_TRISC
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline bool DataflowBuffer<Pap, Cap>::producer_broadcast() const {
    if constexpr (pattern_known) {
        if constexpr (Cap == dfb::AccessPattern::ALL) {
            ASSERT(local_dfb_interface_.broadcast_tc || local_dfb_interface_.num_tcs_to_rr == 1);
        }
        return Cap == dfb::AccessPattern::ALL;
    } else {
        return local_dfb_interface_.broadcast_tc != 0;
    }
}
#endif

// The stride the host serialized for this hart, in entries: entry spacing on a STRIDED side,
// the per-tile cursor hop on a BLOCKED DM side.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::wire_stride_tiles() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#elif defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.stride_size_tiles;
#else
    return static_cast<uint16_t>(dfb_dm_stride_entries(local_dfb_interface_));
#endif
}

// Spacing between the entries of one op: 1 on a BLOCKED side (a whole block), the stride on a
// STRIDED side facing BLOCKED peers.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool IsProducer>
inline uint16_t DataflowBuffer<Pap, Cap>::side_stride_tiles() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#else
    if constexpr (IsProducer ? producer_blocked : consumer_blocked) {
        return 1u;
    } else if constexpr (pattern_known) {
        return stride_tiles_cache_;
    } else {
        return wire_stride_tiles();
    }
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_producer_stride_tiles() const {
    return side_stride_tiles<true>();
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_consumer_stride_tiles() const {
    return side_stride_tiles<false>();
}

// Tiles per op on this side, derived from block_size and this hart's stride.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool IsProducer>
inline uint16_t DataflowBuffer<Pap, Cap>::side_share() const {
#if DFB_IS_COMPUTE_MATH
    return 1u;
#else
    constexpr bool this_side_blocked = IsProducer ? producer_blocked : consumer_blocked;
    constexpr bool peer_side_blocked = IsProducer ? consumer_blocked : producer_blocked;
    if constexpr (this_side_blocked) {
        return local_dfb_interface_.block_size;
    } else if constexpr (peer_side_blocked) {
        return peer_share_cache_;
    } else {
        return 1u;
    }
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_producer_share() const {
    return side_share<true>();
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint16_t DataflowBuffer<Pap, Cap>::get_consumer_share() const {
    return side_share<false>();
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::reserve_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("RBW");
    dfb::PackedTileCounter packed_tc =
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_producer_share() || !share_strict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_wait_for_free_tiles<Pap, Cap>(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (producer_broadcast()) {
        // BROADCAST: every consumer reads every entry, so every counter must have room for all of them.
        dfb_dm_all_tc_wait<true>(local_dfb_interface_, num_entries);
    } else if (producer_split) {
        // SPLIT: the block belongs to every counter, wait for each one's share.
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

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::push_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_producer_share() || !share_strict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_push_tiles<0x1, Pap, Cap>(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (producer_broadcast()) {
        // BROADCAST: post the full count to every counter (every consumer reads every entry).
        dfb_dm_all_tc_credit<true>(local_dfb_interface_, num_entries);
        dfb_dm_broadcast_advance(local_dfb_interface_, num_entries);
    } else if (producer_split) {
        // SPLIT: post each counter its share of the block and step every bookmark past it.
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, num_entries);
        dfb_dm_all_tc_credit<true>(local_dfb_interface_, per_tc);
        dfb_dm_all_slots_advance(local_dfb_interface_, per_tc);
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        dfb_dm_advance_slot(local_dfb_interface_, num_entries);
    }
#endif
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::wait_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("WFW");
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_consumer_share() || !share_strict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_wait_tiles<Pap, Cap>(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (consumer_split) {
        // SPLIT: the block belongs to every counter, wait for each one's share.
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

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::pop_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    ASSERT(num_entries == get_consumer_share() || !share_strict);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= dfb_trisc_per_counter(local_dfb_interface_, num_entries));
    llk_pop_tiles<0x3, Pap, Cap>(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (consumer_split) {
        // SPLIT: ack each counter its share of the block and step every bookmark past it.
        const uint16_t per_tc = dfb_dm_per_tc(local_dfb_interface_, num_entries);
        dfb_dm_all_tc_credit<false>(local_dfb_interface_, per_tc);
        dfb_dm_all_slots_advance(local_dfb_interface_, per_tc);
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::fast_llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::fast_llk_intf_inc_acked(tensix_id, tc_id, num_entries);
        dfb_dm_advance_slot(local_dfb_interface_, num_entries);
    }
#endif
#endif
}

#if !defined(COMPILE_FOR_TRISC)
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
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

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_write_ptr_impl() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#elif defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    return dfb_window_tile_addr_units(local_dfb_interface_, 0);
#elif !defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].ptr;
#else
    // Unpack TRISC does not use wr_ptr; return ring base for any accidental caller.
    ASSERT(false);
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
#endif
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_read_ptr_impl() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#elif defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    return dfb_window_tile_addr_units(local_dfb_interface_, 0);
#elif !defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].ptr;
#else
    // Pack TRISC does not use rd_ptr; return ring base for any accidental caller.
    ASSERT(false);
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
#endif
}

#ifndef COMPILE_FOR_TRISC
template <typename Accessor>
struct PageView;
namespace dfb {
// The remote side of the transfer is a tensor accessor, not a plain address.
template <typename T, typename = void>
inline constexpr bool noc_endpoint_has_dspec_v = false;
template <typename T>
inline constexpr bool noc_endpoint_has_dspec_v<T, std::void_t<decltype(T::DSpec::is_interleaved)>> = true;
template <typename Accessor>
inline constexpr bool noc_endpoint_has_dspec_v<PageView<Accessor>, void> = noc_endpoint_has_dspec_v<Accessor>;
// The transfer arguments name a page (page_id), not one plain address.
template <typename Args, typename = void>
inline constexpr bool noc_args_have_page_id_v = false;
template <typename Args>
inline constexpr bool noc_args_have_page_id_v<Args, std::void_t<decltype(std::declval<Args&>().page_id)>> = true;
// The remote side is an interleaved tensor, its pages are spread over banks.
template <typename T, typename = void>
inline constexpr bool noc_endpoint_is_interleaved_v = false;
template <typename T>
inline constexpr bool noc_endpoint_is_interleaved_v<T, std::void_t<decltype(T::DSpec::is_interleaved)>> =
    T::DSpec::is_interleaved;
template <typename Accessor>
inline constexpr bool noc_endpoint_is_interleaved_v<PageView<Accessor>, void> = noc_endpoint_is_interleaved_v<Accessor>;
// The remote side is a DRAM-interleaved tensor, each bank holds a contiguous run of its pages.
template <typename T, typename = void>
inline constexpr bool noc_endpoint_is_dram_interleaved_v = false;
template <typename T>
inline constexpr bool noc_endpoint_is_dram_interleaved_v<T, std::void_t<decltype(T::DSpec::is_interleaved)>> =
    T::is_dram && T::DSpec::is_interleaved;
template <typename Accessor>
inline constexpr bool noc_endpoint_is_dram_interleaved_v<PageView<Accessor>, void> =
    noc_endpoint_is_dram_interleaved_v<Accessor>;

template <typename RunCallback>
inline void for_each_run(uint32_t block, uint32_t runs, RunCallback&& run) {
    for (uint32_t r = 0; r < runs; ++r) {
        run(bank_major_run_offset(block, runs, r), bank_major_run_length(block, runs, r), r);
    }
}

template <typename RunCallback>
inline void for_each_bank_run(uint32_t block, RunCallback&& run) {
    for_each_run(block, bank_major_num_runs(block, NUM_DRAM_BANKS), run);
}

template <typename PageCallback>
inline void for_each_page(uint32_t block, PageCallback&& page) {
    for (uint32_t i = 0; i < block; ++i) {
        page(i);
    }
}
}  // namespace dfb
#endif

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::entry_addr(uint32_t addr) const {
#ifndef COMPILE_FOR_TRISC
    // The cursor's address counts entries in page order; a bank-major ring keeps the entry elsewhere.
    if constexpr (Pap == dfb::AccessPattern::BLOCKED || Cap == dfb::AccessPattern::BLOCKED) {
        if (dfb_block_is_permuted(local_dfb_interface_)) {
            const uint32_t entry_bytes = local_dfb_interface_.entry_size;
            const uint32_t ring_base =
                local_dfb_interface_.tc_slots[0].base_addr - local_dfb_interface_.first_entry_in_block * entry_bytes;
            const uint32_t rel = addr - ring_base;
            const uint32_t entry = dfb::bank_major_entry(
                local_dfb_interface_.block_size, local_dfb_interface_.block_banks, rel / entry_bytes);
            return ring_base + entry * entry_bytes + rel % entry_bytes;
        }
    }
#endif
    return addr;
}

#ifdef COMPILE_FOR_TRISC
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::get_tile_address(uint32_t tile_index) {
    uint32_t address = 0;
#if defined(UCK_CHLKC_UNPACK)
    {
        // No wrap: wait_front(n) must not straddle slot.limit and tile_index is in [0, n).
        address = address_units_to_bytes(dfb_window_tile_addr_units(local_dfb_interface_, tile_index));
        mailbox_write(ckernel::ThreadId::MathThreadId, address);
        mailbox_write(ckernel::ThreadId::PackThreadId, address);
        mailbox_write(ckernel::ThreadId::IsolateSfpuThreadId, address);
    }
#elif defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_ISOLATE_SFPU)
    address = mailbox_read(ckernel::ThreadId::UnpackThreadId);
#endif
    return address;
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <typename T>
T DataflowBuffer<Pap, Cap>::read_tile_value(uint32_t tile_index, uint32_t element_offset) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "read_tile_value: T must be 1, 2, or 4 bytes");
    static_assert(
        (std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>),
        "read_tile_value: T must be an unsigned integral type");

    T value = T{};
#if defined(UCK_CHLKC_UNPACK)
    {
        // Same addressing as get_tile_address: the wait_front window does not wrap.
        const uint32_t byte_address =
            address_units_to_bytes(dfb_window_tile_addr_units(local_dfb_interface_, tile_index));
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
// Lock the `n` held entries. The locked region starts at the write pointer (scoped_write_lock) or the
// read pointer (scoped_read_lock), with entries spaced by stride_size: for the ALL access pattern
// stride_size == entry_size, so the locked entries are contiguous; for STRIDED stride_size > entry_size,
// so they are non-contiguous. For each held entry, do two things:
//     - cache op: invalidate the L2 range on acquire (both lock kinds); flush on release (write lock
//       only)
//     - record the scoped-lock event
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool is_write>
inline typename DataflowBuffer<Pap, Cap>::ScopedLockRegion DataflowBuffer<Pap, Cap>::lock_acquire_impl(uint16_t num_entries) {
    // A lock covers one op: num_entries entries from the cursor, stride_tiles apart.
    ASSERT(num_entries == (is_write ? get_producer_share() : get_consumer_share()) || !share_strict);
    const auto& s = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
    const uint32_t entry = local_dfb_interface_.entry_size;
    const uint32_t stride = entry * (is_write ? get_producer_stride_tiles() : get_consumer_stride_tiles());
    // Snapshot the start pointer + this slot's wrap bounds so release replays the identical walk.
    const ScopedLockRegion region{s.ptr, s.base_addr, s.limit};
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        RECORD_SCOPED_LOCK_EVENT(NocDebuggingEventMetadata::NocDebugEventType::DFB_LOCK, entry_addr(addr), entry);
        // TODO: with concurrent ALL consumers, this invalidates the same shared cache line once per
        // consumer; the redundant invalidations could be deduplicated (e.g. first-locker-per-round).
        // Currently this invalidates the L2 range, which also drops the matching L1 D$ line on all
        // DM cores.
        scoped_lock_acquire_cache_ops(entry_addr(addr), entry);
        addr += stride;
        if (addr >= region.limit) {
            addr = region.base;
        }
    }
    return region;
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool is_write>
inline void DataflowBuffer<Pap, Cap>::lock_release_impl(ScopedLockRegion region, uint16_t num_entries) {
    const uint32_t entry = local_dfb_interface_.entry_size;
    const uint32_t stride = entry * (is_write ? get_producer_stride_tiles() : get_consumer_stride_tiles());
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        // Flush on release only for a write lock. A read lock never writes.
        if constexpr (is_write) {
            // Currently this flushes l2, which writes back + drops the matching L1 D$ line on all DM cores.
            scoped_lock_release_cache_ops(entry_addr(addr), entry);
        }
        RECORD_SCOPED_LOCK_EVENT(NocDebuggingEventMetadata::NocDebugEventType::DFB_UNLOCK, entry_addr(addr), entry);
        addr += stride;
        if (addr >= region.limit) {
            addr = region.base;
        }
    }
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::write_barrier_impl(const Noc& noc) const {
    write_barrier_impl(noc.get_noc_id());
}

// Consumer barrier: waits outbound write from DFB writes to arrive at their destination
// Falls back to a full barrier when no txn_ids are assigned
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::write_barrier_impl(uint8_t noc_id) const {
    if (local_dfb_interface_.num_txn_ids == 0) {
        noc_async_write_barrier(noc_id);
        return;
    } else {
        for (uint8_t i = 0; i < local_dfb_interface_.num_txn_ids; i++) {
            // Uses internal API rather than user facing noc.async_write_barrier() since it ASSERTs that the txn_id comes
            // from the user tnx ID pool and the DFB txn ids are internal only.
            noc_async_write_barrier_with_trid(local_dfb_interface_.txn_ids[i], noc_id);
        }
    }
}

// Preamble for implicit-sync read: spin until previous reads are posted and there is space in the tile counters.
// Returns the txn_id to stamp on the next NOC read.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::prepare_implicit_read(uint32_t num_tiles) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    const DFBImplicitProgress& progress = implicit_progress();
    const uint16_t per_tc = local_dfb_interface_.num_entries_per_txn_id_per_tc;
    const uint32_t txn_id = local_dfb_interface_.txn_ids[progress.txn_idx];
    WAYPOINT("PIRW");
    // Modular comparison: posted (16-bit HW) wraps at 65 536, and so does the
    // kernel-side expectation `loop * per_tc` if both operands are
    // reduced to uint16 before subtracting. Interpreting the wrapped difference
    // as int16 gives a signed modular distance — negative means posted is behind
    // expected, non-negative means it has caught up. Safe as long as the gap
    // never exceeds half the wrap range (~32K), which is guaranteed by the
    // bounded txn-id ring depth.
    while (static_cast<int16_t>(
        static_cast<uint16_t>(overlay::fast_llk_intf_read_posted(tensix_id, tc_id)) -
        static_cast<uint16_t>(progress.loop * per_tc)) < 0);
    // Wait until there is room for the whole op. Which counters to check depends on the ring:
    // broadcast waits on every counter for all of it, split waits on every counter for its part,
    // otherwise only this round's counter. `mine` is what this hart already sent to that counter.
    ASSERT(num_tiles == get_producer_share());
    const uint8_t num_tcs = local_dfb_interface_.num_tcs_to_rr;
    if (producer_broadcast()) {
        // Broadcast: every op went to every counter, so this is everything issued.
        const uint16_t mine =
            static_cast<uint16_t>(progress.loop * local_dfb_interface_.num_entries_per_txn_id + progress.pos);
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<true>(local_dfb_interface_, i, mine, num_tiles);
        }
    } else if (producer_split) {
        const uint16_t op_per_tc = dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(num_tiles));
        const uint16_t mine = static_cast<uint16_t>(progress.loop * per_tc + progress.pos / num_tcs);
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<true>(local_dfb_interface_, i, mine, op_per_tc);
        }
    } else {
        const uint32_t round = static_cast<uint32_t>(num_tcs) * num_tiles;
        const uint16_t mine = static_cast<uint16_t>(progress.loop * per_tc + (progress.pos / round) * num_tiles);
        dfb_dm_implicit_wait<true>(local_dfb_interface_, local_dfb_interface_.tc_idx, mine, num_tiles);
    }
    WAYPOINT("PIRD");
    return txn_id;
}

// Postamble for implicit-sync read: advance wr_ptr, tile/txn counters, and tc_idx.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::commit_implicit_read(uint32_t num_tiles) {
    // Runs once per op; the ISR posts the credits, this only moves the bookmarks.
    const uint32_t block = num_tiles;
    if (producer_broadcast()) {
        dfb_dm_broadcast_advance(local_dfb_interface_, block);
    } else if (producer_split) {
        dfb_dm_all_slots_advance(
            local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(block)));
    } else {
        dfb_dm_advance_slot(local_dfb_interface_, block);
    }
    dfb_implicit_advance(local_dfb_interface_, implicit_progress(), block);
}

// Preamble for implicit-sync write: spin until previous writes are acked and data is available in the tile counters.
// Returns the txn_id to stamp on the next NOC write.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline uint32_t DataflowBuffer<Pap, Cap>::prepare_implicit_write(uint32_t num_tiles) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
    const DFBImplicitProgress& progress = implicit_progress();
    const uint16_t per_tc = local_dfb_interface_.num_entries_per_txn_id_per_tc;
    const uint32_t txn_id = local_dfb_interface_.txn_ids[progress.txn_idx];
    WAYPOINT("PIWW");
    // Modular comparison — see prepare_implicit_read for the rationale. Same
    // trick applied to the acked side.
    while (static_cast<int16_t>(
        static_cast<uint16_t>(overlay::fast_llk_intf_read_acked(tensix_id, tc_id)) -
        static_cast<uint16_t>(progress.loop * per_tc)) < 0);
    // Wait until the whole op has arrived. Split waits on every counter for its part, otherwise
    // only on this round's counter. `mine` is what this hart already took from that counter.
    ASSERT(num_tiles == get_consumer_share());
    const uint8_t num_tcs = local_dfb_interface_.num_tcs_to_rr;
    if (consumer_split) {
        const uint16_t op_per_tc = dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(num_tiles));
        const uint16_t mine = static_cast<uint16_t>(progress.loop * per_tc + progress.pos / num_tcs);
        for (uint8_t i = 0; i < num_tcs; i++) {
            dfb_dm_implicit_wait<false>(local_dfb_interface_, i, mine, op_per_tc);
        }
    } else {
        const uint32_t round = static_cast<uint32_t>(num_tcs) * num_tiles;
        const uint16_t mine = static_cast<uint16_t>(progress.loop * per_tc + (progress.pos / round) * num_tiles);
        dfb_dm_implicit_wait<false>(local_dfb_interface_, local_dfb_interface_.tc_idx, mine, num_tiles);
    }
    WAYPOINT("PIWD");
    return txn_id;
}

// Postamble for implicit-sync write: advance rd_ptr, tile/txn counters, and tc_idx.
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void DataflowBuffer<Pap, Cap>::commit_implicit_write(uint32_t num_tiles) {
    // Runs once per op; the ISR acks the credits, this only moves the bookmarks.
    const uint32_t block = num_tiles;
    if (consumer_split) {
        dfb_dm_all_slots_advance(
            local_dfb_interface_, dfb_dm_per_tc(local_dfb_interface_, static_cast<uint16_t>(block)));
    } else {
        dfb_dm_advance_slot(local_dfb_interface_, block);
    }
    dfb_implicit_advance(local_dfb_interface_, implicit_progress(), block);
}

// Out-of-line definitions of Noc DFB-specific implicit-sync overloads.
// These are member functions of Noc but must be defined here because they need the complete
// DataflowBuffer type (circular dependency: dataflow_buffer.h includes noc.h, not vice versa).

template <NocOptions opts, typename Src, dfb::AccessPattern Pap, dfb::AccessPattern Cap>
NOC_TRANSFER_INLINE std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_read(
    const Src& src,
    DataflowBuffer<Pap, Cap>& dst,
    const typename noc_traits_t<Src>::src_args_type& src_args,
    const DataflowBufferArgs& dst_args) const {
    // Implicit sync always lands data at the cursor; offset_bytes is ignored.
    ASSERT(dst_args.offset_bytes == 0);
    const uint32_t entry_bytes = dst.get_entry_size();
    const uint32_t share = dst.get_producer_share();
    // Reads `bytes` from page_offset pages past src_args into L1 at l1_addr, tagged txn_id.
    auto issue = [&](uint32_t txn_id, uint32_t l1_addr, uint32_t bytes, uint32_t page_offset) {
        noc_async_read_set_trid(txn_id, noc_id_);
        while (noc_available_transactions(noc_id_, txn_id) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));
        auto issue_at = [&](uint64_t src_noc_addr) {
            noc_async_read_impl<NOC_MAX_BURST_SIZE + 1, true>(src_noc_addr, l1_addr, bytes, noc_id_, NOC_UNICAST_WRITE_VC);
        };
        auto args = src_args;
        uint64_t noc_offset = 0;
        if constexpr (dfb::noc_args_have_page_id_v<decltype(args)>) {
            args.page_id += page_offset;
        } else {
            noc_offset = static_cast<uint64_t>(page_offset) * entry_bytes;
        }
        if constexpr (noc_addrgen_push_v<Src>) {
            ASSERT(noc_offset == 0);
            // The ordinary issue above always traces (noc_async_read_impl<..., true>); so does the pushed one.
            issue_read_maybe_pushed</*trace=*/true>(src, args, l1_addr, bytes, NOC_UNICAST_WRITE_VC, issue_at);
        } else {
            const uint64_t src_noc_addr = get_src_ptr<AddressType::NOC>(src, args);
            issue_at(src_noc_addr + noc_offset);
        }
    };
    if constexpr (Pap == dfb::AccessPattern::BLOCKED) {
        // BLOCKED producer: one whole block per call, in the transactions the host chose for this side.
        const uint32_t txn_id = dst.prepare_implicit_read(share);
        const uint32_t block_addr = dst.get_noc_write_addr();
        const uint32_t runs = dst.local_dfb_interface_.runs_per_block;
        // One transaction per page, each to its own slot: always right, used when nothing faster is.
        auto page_by_page = [&] {
            ASSERT(runs == share);
            dfb::for_each_page(share, [&](uint32_t i) {
                issue(txn_id, dst.entry_addr(block_addr + i * entry_bytes), entry_bytes, i);
            });
        };
        if constexpr (dfb::noc_endpoint_is_dram_interleaved_v<Src>) {
            if (dst.local_dfb_interface_.block_banks == NUM_DRAM_BANKS &&
                runs == dfb::bank_major_num_runs(share, NUM_DRAM_BANKS)) {
                // One transaction per DRAM bank run (pages first+r, first+r+N, ...).
                dfb::for_each_bank_run(share, [&](uint32_t entry_offset, uint32_t num_entries, uint32_t r) {
                    issue(txn_id, block_addr + entry_offset * entry_bytes, num_entries * entry_bytes, r);
                });
            } else {
                // The ring is not bank-major.
                page_by_page();
            }
        } else if constexpr (
            dfb::noc_endpoint_is_interleaved_v<Src> ||
            (!dfb::noc_endpoint_has_dspec_v<Src> && dfb::noc_args_have_page_id_v<src_args_t<Src>>)) {
            // Pages spread over L1 banks, or a layout this kernel cannot see.
            page_by_page();
        } else if (!dfb_block_is_permuted(dst.local_dfb_interface_)) {
            // A contiguous block into a page-order ring: `runs` contiguous pieces (normally one).
            dfb::for_each_run(share, runs, [&](uint32_t entry_offset, uint32_t num_entries, uint32_t) {
                issue(txn_id, block_addr + entry_offset * entry_bytes, num_entries * entry_bytes, entry_offset);
            });
        } else {
            // A contiguous block into a bank-major ring.
            page_by_page();
        }
        dst.commit_implicit_read(share);
    } else {
        // STRIDED producer: one entry per call, as one NoC transaction. The op's txn id stays
        // txn_ids[txn_idx] until the commit after its last entry.
        DFBImplicitProgress& progress = dst.implicit_progress();
        if (progress.share_pos == 0) {
            dst.prepare_implicit_read(share);
        }
        const uint32_t txn_id = dst.local_dfb_interface_.txn_ids[progress.txn_idx];
        const uint32_t stride_bytes = entry_bytes * dst.get_producer_stride_tiles();
        issue(txn_id, dst.write_cursor_addr(progress.share_pos * stride_bytes), entry_bytes, 0);
        if (++progress.share_pos == share) {
            progress.share_pos = 0;
            dst.commit_implicit_read(share);
        }
    }
}

template <NocOptions opts, typename Dst, dfb::AccessPattern Pap, dfb::AccessPattern Cap>
NOC_TRANSFER_INLINE std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_write(
    DataflowBuffer<Pap, Cap>& src,
    const Dst& dst,
    const DataflowBufferArgs& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    // Consumer side of the async_read contract: a BLOCKED consumer writes one whole block per
    // call, otherwise one entry per call; offset_bytes is ignored.
    ASSERT(src_args.offset_bytes == 0);
    const uint32_t entry_bytes = src.get_entry_size();
    const uint32_t share = src.get_consumer_share();
    // Writes `bytes` from L1 at src_addr to page_offset pages past dst_args, tagged txn_id.
    auto issue = [&](uint32_t txn_id, uint32_t src_addr, uint32_t bytes, uint32_t page_offset) {
        auto issue_at = [&](uint64_t dst_noc_addr) {
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
        auto args = dst_args;
        uint64_t noc_offset = 0;
        if constexpr (dfb::noc_args_have_page_id_v<decltype(args)>) {
            args.page_id += page_offset;
        } else {
            noc_offset = static_cast<uint64_t>(page_offset) * entry_bytes;
        }
        if constexpr (noc_addrgen_push_v<Dst>) {
            ASSERT(noc_offset == 0);
            issue_write_maybe_pushed</*posted=*/false, /*use_trid=*/true, /*trace=*/true>(
                dst, args, src_addr, bytes, NOC_UNICAST_WRITE_VC, txn_id, issue_at);
        } else {
            const uint64_t dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, args);
            issue_at(dst_noc_addr + noc_offset);
        }
    };
    if constexpr (Cap == dfb::AccessPattern::BLOCKED) {
        // BLOCKED consumer: the mirror image of the BLOCKED producer read above.
        const uint32_t txn_id = src.prepare_implicit_write(share);
        const uint32_t block_addr = src.get_noc_read_addr();
        const uint32_t runs = src.local_dfb_interface_.runs_per_block;
        auto page_by_page = [&] {
            ASSERT(runs == share);
            dfb::for_each_page(share, [&](uint32_t i) {
                issue(txn_id, src.entry_addr(block_addr + i * entry_bytes), entry_bytes, i);
            });
        };
        if constexpr (dfb::noc_endpoint_is_dram_interleaved_v<Dst>) {
            if (src.local_dfb_interface_.block_banks == NUM_DRAM_BANKS &&
                runs == dfb::bank_major_num_runs(share, NUM_DRAM_BANKS)) {
                dfb::for_each_bank_run(share, [&](uint32_t entry_offset, uint32_t num_entries, uint32_t r) {
                    issue(txn_id, block_addr + entry_offset * entry_bytes, num_entries * entry_bytes, r);
                });
            } else {
                page_by_page();
            }
        } else if constexpr (
            dfb::noc_endpoint_is_interleaved_v<Dst> ||
            (!dfb::noc_endpoint_has_dspec_v<Dst> && dfb::noc_args_have_page_id_v<dst_args_t<Dst>>)) {
            page_by_page();
        } else if (!dfb_block_is_permuted(src.local_dfb_interface_)) {
            dfb::for_each_run(share, runs, [&](uint32_t entry_offset, uint32_t num_entries, uint32_t) {
                issue(txn_id, block_addr + entry_offset * entry_bytes, num_entries * entry_bytes, entry_offset);
            });
        } else {
            page_by_page();
        }
        src.commit_implicit_write(share);
    } else {
        // STRIDED consumer: one entry per call, as one NoC transaction. The op's txn id stays
        // txn_ids[txn_idx] until the commit after its last entry.
        DFBImplicitProgress& progress = src.implicit_progress();
        if (progress.share_pos == 0) {
            src.prepare_implicit_write(share);
        }
        const uint32_t txn_id = src.local_dfb_interface_.txn_ids[progress.txn_idx];
        const uint32_t stride_bytes = entry_bytes * src.get_consumer_stride_tiles();
        issue(txn_id, src.read_cursor_addr(progress.share_pos * stride_bytes), entry_bytes, 0);
        if (++progress.share_pos == share) {
            progress.share_pos = 0;
            src.commit_implicit_write(share);
        }
    }
}

#else  // COMPILE_FOR_TRISC

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool>
inline typename DataflowBuffer<Pap, Cap>::ScopedLockRegion DataflowBuffer<Pap, Cap>::lock_acquire_impl(uint16_t) { return {}; }
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
template <bool>
inline void DataflowBuffer<Pap, Cap>::lock_release_impl(ScopedLockRegion, uint16_t) {}

#endif  // !COMPILE_FOR_TRISC

#endif  // ARCH_QUASAR
