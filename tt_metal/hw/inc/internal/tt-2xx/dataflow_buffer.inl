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

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_MATH)
#define DFB_IS_COMPUTE_MATH 1
#else
#define DFB_IS_COMPUTE_MATH 0
#endif

#if DFB_IS_COMPUTE_MATH
inline DataflowBuffer::DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
}
#else
inline DataflowBuffer::DataflowBuffer(uint16_t logical_dfb_id)
    : logical_dfb_id_(logical_dfb_id), local_dfb_interface_(get_local_dfb_interface(logical_dfb_id)) {
    dfb_ensure_ready(g_dfb_config_base_addr, static_cast<uint8_t>(logical_dfb_id));
    // Declare this DFB's L1 extent to the NOC-debug tracker so a write into it without holding the
    // lock can be flagged.
    RECORD_SCOPED_LOCK_EVENT(
        NocDebuggingEventMetadata::NocDebugEventType::DFB_REGION_START,
        address_units_to_bytes(local_dfb_interface_.tc_slots[0].base_addr),
        get_ring_span_bytes());
}
#endif

inline uint32_t DataflowBuffer::get_entry_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

inline uint32_t DataflowBuffer::get_stride_size() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(local_dfb_interface_.stride_size);
#endif
}

inline uint32_t DataflowBuffer::get_total_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return local_dfb_interface_.num_entries;
#endif
}

inline uint32_t DataflowBuffer::get_total_size_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return get_total_num_entries() * address_units_to_bytes(local_dfb_interface_.entry_size);
#endif
}

inline uint32_t DataflowBuffer::get_local_num_entries() const {
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
    return static_cast<uint32_t>(overlay::llk_intf_get_capacity(tensix_id, tc_id));
#endif
#endif
}

inline uint32_t DataflowBuffer::get_local_size_bytes() const {
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
// Tiles this hart collects from one counter before rotating counters to the next.
inline uint32_t dfb_dm_tiles_to_collect(const LocalDFBInterface& intf) {
    if (intf.split_tc) {
        return 1u;
    }
    const uint32_t stride_entries =
        (intf.entry_size && intf.stride_size >= intf.entry_size) ? (intf.stride_size / intf.entry_size) : 1u;
    return (intf.block_size > stride_entries) ? (intf.block_size / stride_entries) : 1u;
}

// How many of a transaction's n tiles land on each tile counter. With split_tc set and n
// dividing evenly over the counters, each counter gets an equal share (split = true); in any
// other case one counter keeps the full count (split = false).
struct DfbTcShare {
    uint16_t per_tc;
    bool split;
};
inline DfbTcShare dfb_dm_per_tc_share(const LocalDFBInterface& intf, uint16_t n) {
    const bool split = intf.split_tc && (n % intf.num_tcs_to_rr == 0);
    return {split ? static_cast<uint16_t>(n / intf.num_tcs_to_rr) : n, split};
}

// Producer-side: spin until every counter has room for per_tc tiles. Used when one
// transaction spans all the counters (split_tc) or posts to all of them (broadcast_tc).
template <bool fast>
inline void dfb_dm_all_tc_wait(const LocalDFBInterface& intf, uint32_t per_tc) {
    bool ready = false;
    while (!ready) {
        ready = true;
        for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
            dfb::PackedTileCounter ptc = intf.tc_slots[i].packed_tile_counter;
            const uint8_t tid = dfb::get_tensix_id(ptc);
            const uint8_t cid = dfb::get_counter_id(ptc);
            if constexpr (!fast) {
                ASSERT(overlay::llk_intf_get_capacity(tid, cid) >= per_tc);
            }
            const uint32_t level = fast ? overlay::fast_llk_intf_get_free_space(tid, cid)
                                        : overlay::llk_intf_get_free_space(tid, cid);
            if (level < per_tc) {
                ready = false;
                break;
            }
        }
    }
}

// Producer-side: post per_tc credits on every counter, each counter's consumer gets its
// share of the one transaction.
inline void dfb_dm_all_tc_credit(const LocalDFBInterface& intf, uint16_t per_tc) {
    for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = intf.tc_slots[i].packed_tile_counter;
        const uint8_t tid = dfb::get_tensix_id(ptc);
        const uint8_t cid = dfb::get_counter_id(ptc);
        ASSERT(overlay::llk_intf_get_capacity(tid, cid) >= per_tc);
        overlay::llk_intf_inc_posted(tid, cid, per_tc);
    }
}

// Producer-side, after a split transaction: every counter just received its share of the
// block, so step every slot's write cursor forward by that share (each wraps back to its own
// base). tc_idx stays at index 0 because each TC receives tiles in a split transaction, so
// the next send is still addressed from slot 0's cursor.
inline void dfb_dm_all_slots_advance(LocalDFBInterface& intf, uint32_t step_bytes) {
    for (uint8_t i = 0; i < intf.num_tcs_to_rr; i++) {
        uint32_t& ptr = intf.tc_slots[i].wr_ptr;
        ptr += step_bytes;
        if (ptr >= intf.tc_slots[i].limit) {
            ptr = intf.tc_slots[i].base_addr;
        }
    }
}

// Each tile counter keeps its own cursor, a bookmark into its tiles of the ring. The hart
// collects `tiles_to_collect` tiles from one counter, then it hands off: it parks this
// counter's bookmark `jump` bytes ahead (its next tiles sit past the other counters')
// and rotates to the next counter, whose bookmark is already waiting exactly where the
// stream continues. A bookmark that runs off the end of the ring wraps to its base.
template <bool is_write>
inline void dfb_dm_advance_slot(LocalDFBInterface& intf, uint32_t n) {
    auto& slot = intf.tc_slots[intf.tc_idx];
    const uint32_t tiles_to_collect = dfb_dm_tiles_to_collect(intf);
    const uint32_t current_tile_count =
        (tiles_to_collect > 1u) ? (static_cast<uint32_t>(intf.tiles_collected) + n) : 1u;
    ASSERT(current_tile_count <= tiles_to_collect || n >= tiles_to_collect);
    const bool hand_off = current_tile_count >= tiles_to_collect;
    const uint32_t advance = (n - 1u) * intf.stride_size + (hand_off ? intf.jump : intf.stride_size);
    if constexpr (is_write) {
        slot.wr_ptr += advance;
        if (slot.wr_ptr >= slot.limit) {
            slot.wr_ptr = slot.base_addr;
        }
    } else {
        slot.rd_ptr += advance;
        if (slot.rd_ptr >= slot.limit) {
            slot.rd_ptr = slot.base_addr;
        }
    }
    if (hand_off) {
        intf.tiles_collected = 0;
        intf.tc_idx = (intf.tc_idx + 1) % intf.num_tcs_to_rr;
    } else {
        intf.tiles_collected = static_cast<uint16_t>(current_tile_count);
    }
}
#endif  // !COMPILE_FOR_TRISC

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

inline uint32_t DataflowBuffer::get_ring_span_bytes() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_));
#endif
}

inline uint32_t DataflowBuffer::get_ring_span_num_entries() const {
#if DFB_IS_COMPUTE_MATH
    return 0;
#else
    const uint32_t entry_bytes = get_entry_size();
    return address_units_to_bytes(dfb_ring_span_address_units(local_dfb_interface_)) / entry_bytes;
#endif
}

inline void DataflowBuffer::reserve_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("RBW");
    dfb::PackedTileCounter packed_tc =
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    llk_wait_for_free_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (__builtin_expect(local_dfb_interface_.broadcast_tc || local_dfb_interface_.split_tc, 0)) {
        // BROADCAST: every TC receives the full count; SPLIT: each TC receives its share of the full count.
        dfb_dm_all_tc_wait<false>(local_dfb_interface_, dfb_dm_per_tc_share(local_dfb_interface_, num_entries).per_tc);
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (overlay::llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
    }
#endif
    WAYPOINT("RBD");
#endif
}

inline void DataflowBuffer::push_back_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    llk_push_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (__builtin_expect(local_dfb_interface_.broadcast_tc || local_dfb_interface_.split_tc, 0)) {
        // BROADCAST: post the full count to every TC (every consumer reads every entry).
        // SPLIT: post each TC its share, a count that doesn't divide evenly can't be split.
        const DfbTcShare share = dfb_dm_per_tc_share(local_dfb_interface_, num_entries);
        dfb_dm_all_tc_credit(local_dfb_interface_, share.per_tc);
        if (share.split) {
            dfb_dm_all_slots_advance(local_dfb_interface_, share.per_tc * local_dfb_interface_.stride_size);
        } else {
            local_dfb_interface_.tc_slots[0].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
            if (local_dfb_interface_.tc_slots[0].wr_ptr >= local_dfb_interface_.tc_slots[0].limit) {
                local_dfb_interface_.tc_slots[0].wr_ptr = local_dfb_interface_.tc_slots[0].base_addr;
            }
        }
        // tc_idx deliberately not advanced
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        dfb_dm_advance_slot</*is_write=*/true>(local_dfb_interface_, num_entries);
    }
#endif
#endif
}

inline void DataflowBuffer::wait_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    WAYPOINT("WFW");
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    llk_wait_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
    while (overlay::llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
#endif
    WAYPOINT("WFD");
#endif
}

inline void DataflowBuffer::pop_front_impl(uint16_t num_entries) {
#if !DFB_IS_COMPUTE_MATH
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
        return;
    }
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    llk_pop_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
    overlay::llk_intf_inc_acked(tensix_id, tc_id, num_entries);
    dfb_dm_advance_slot</*is_write=*/false>(local_dfb_interface_, num_entries);
#endif
#endif
}

inline void DataflowBuffer::finish_impl() {
#if !DFB_IS_COMPUTE_MATH
#ifndef COMPILE_FOR_TRISC
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

inline uint32_t DataflowBuffer::get_write_ptr_impl() const {
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

inline uint32_t DataflowBuffer::get_read_ptr_impl() const {
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

#ifndef COMPILE_FOR_TRISC
template <bool is_producer>
inline void DataflowBuffer::handle_final_credits(uint16_t tiles_issued, uint8_t txn_id_index) {
    // Determine the txn_id for the last batch. If tiles_issued lands exactly on
    // a boundary, txn_id_index has already wrapped past it, so step back one slot.
    uint8_t tail_txn_idx = (tiles_issued % local_dfb_interface_.num_entries_per_txn_id == 0)
                                ? static_cast<uint8_t>((txn_id_index + local_dfb_interface_.num_txn_ids - 1) % local_dfb_interface_.num_txn_ids)
                                : txn_id_index;
    uint8_t tail_txn_id = local_dfb_interface_.txn_ids[tail_txn_idx];

    uint8_t N = local_dfb_interface_.num_tcs_to_rr;
    dfb::PackedTileCounter ptc0 = local_dfb_interface_.tc_slots[0].packed_tile_counter;
    // finish() cannot return until the ISR has credited every tile this hart issued. To know
    // what to wait for, compute how many tiles each counter should have been credited: the
    // hart fills one counter with `visit` tiles, then the next, wrapping around all N.
    // (split_tc is the exception: every transaction is shared equally, so each counter just
    // gets 1/N of the total.)
    const uint16_t tiles_to_collect = static_cast<uint16_t>(dfb_dm_tiles_to_collect(local_dfb_interface_));
    const uint16_t visit = (tiles_to_collect > implicit_txn_tiles_) ? tiles_to_collect : implicit_txn_tiles_;
    const uint16_t NV = static_cast<uint16_t>(N * visit);
    auto expected_for_slot = [&](uint8_t i) -> uint16_t {
        if (local_dfb_interface_.split_tc) {
            return static_cast<uint16_t>(tiles_issued / N);
        }
        const uint16_t full = static_cast<uint16_t>((tiles_issued / NV) * visit);
        const uint16_t rem = tiles_issued % NV;
        const uint16_t start = static_cast<uint16_t>(i * visit);
        uint16_t part = 0u;
        if (rem > start) {
            const uint16_t into_slot = static_cast<uint16_t>(rem - start);
            part = (into_slot > visit) ? visit : into_slot;
        }
        return static_cast<uint16_t>(full + part);
    };
    uint16_t expected_slot0 = expected_for_slot(0);

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
            tack  = CMDBUF_TR_ACK_TRID(OVERLAY_RD_CMD_BUF, tail_txn_id);
            tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tack  = CMDBUF_WR_SENT_TRID(OVERLAY_WR_CMD_BUF, tail_txn_id);
            tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
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
    // Producer and consumer kernels co-reside with different thread counts, so each
    // side uses its own barrier (0 = producer, 1 = consumer) — sharing one deadlocks.
    sync_threads(is_producer ? 0 : 1);

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
            tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
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
template <bool is_write>
inline DataflowBuffer::ScopedLockRegion DataflowBuffer::lock_acquire_impl(uint16_t num_entries) {
    ASSERT(
        dfb_dm_tiles_to_collect(local_dfb_interface_) <= 1u &&
        (local_dfb_interface_.block_size > 1
             ? local_dfb_interface_.stride_size == local_dfb_interface_.entry_size
             : local_dfb_interface_.jump == local_dfb_interface_.stride_size));
    const auto& s = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx];
    const uint32_t stride = local_dfb_interface_.stride_size;
    const uint32_t entry = local_dfb_interface_.entry_size;
    // Snapshot the start pointer + this slot's wrap bounds so release replays the identical walk.
    const ScopedLockRegion region{is_write ? s.wr_ptr : s.rd_ptr, s.base_addr, s.limit};
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        RECORD_SCOPED_LOCK_EVENT(NocDebuggingEventMetadata::NocDebugEventType::DFB_LOCK, addr, entry);
        // TODO: with concurrent ALL consumers, this invalidates the same shared cache line once per
        // consumer; the redundant invalidations could be deduplicated (e.g. first-locker-per-round).
        // invalidate_l2 also drops the matching L1 D$ line on all DM cores.
        invalidate_l2_cache_range(addr, entry);
        addr += stride;
        if (addr >= region.limit) {
            addr = region.base;
        }
    }
    return region;
}

template <bool is_write>
inline void DataflowBuffer::lock_release_impl(ScopedLockRegion region, uint16_t num_entries) {
    const uint32_t stride = local_dfb_interface_.stride_size;
    const uint32_t entry = local_dfb_interface_.entry_size;
    uint32_t addr = region.start;
    for (uint16_t k = 0; k < num_entries; ++k) {
        // Flush on release only for a write lock. A read lock never writes.
        if constexpr (is_write) {
            // flush_l2 writes back + drops the matching L1 D$ line on all DM cores.
            flush_l2_cache_range(addr, entry);
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
inline void DataflowBuffer::write_barrier_impl(const Noc &noc) const {
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
inline uint32_t DataflowBuffer::prepare_implicit_read(uint32_t num_tiles) {
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
    // The kernel says how many tiles this transaction fills; wait for room for all of them.
    // Under SPLIT the transaction spans every TC, so wait for each TC's share on each TC. A count
    // that doesn't divide evenly can't be split, so wait for the full count instead.
    // A broadcasting block producer posts the whole count to every TC, so wait for it on each.
    if (__builtin_expect(local_dfb_interface_.split_tc, 0)) {
        dfb_dm_all_tc_wait<true>(
            local_dfb_interface_, dfb_dm_per_tc_share(local_dfb_interface_, num_tiles).per_tc);
    } else if (__builtin_expect(local_dfb_interface_.broadcast_tc && num_tiles > 1, 0)) {
        dfb_dm_all_tc_wait<true>(local_dfb_interface_, num_tiles);
    } else {
        while (overlay::fast_llk_intf_get_free_space(tensix_id, tc_id) < num_tiles);
    }
    WAYPOINT("PIRD");
    return txn_id;
}

// Postamble for implicit-sync read: advance wr_ptr, tile/txn counters, and tc_idx.
inline void DataflowBuffer::commit_implicit_read(uint32_t num_tiles) {
    // Runs once per transaction; the kernel said how many tiles it carried.
    const uint32_t block = num_tiles;
    implicit_txn_tiles_ = static_cast<uint16_t>(block);
    if (__builtin_expect(local_dfb_interface_.split_tc, 0)) {
        dfb_dm_all_slots_advance(
            local_dfb_interface_,
            dfb_dm_per_tc_share(local_dfb_interface_, static_cast<uint16_t>(block)).per_tc *
                local_dfb_interface_.stride_size);
    } else if (__builtin_expect(local_dfb_interface_.broadcast_tc && block > 1, 0)) {
        local_dfb_interface_.tc_slots[0].wr_ptr += local_dfb_interface_.stride_size * block;
        if (local_dfb_interface_.tc_slots[0].wr_ptr >= local_dfb_interface_.tc_slots[0].limit) {
            local_dfb_interface_.tc_slots[0].wr_ptr = local_dfb_interface_.tc_slots[0].base_addr;
        }
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
inline uint32_t DataflowBuffer::prepare_implicit_write(uint32_t num_tiles) {
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
    // The kernel says how many tiles this transaction drains; wait until they are available.
    while (overlay::fast_llk_intf_get_occupancy(tensix_id, tc_id) < num_tiles);
    WAYPOINT("PIWD");
    return txn_id;
}

// Postamble for implicit-sync write: advance rd_ptr, tile/txn counters, and tc_idx.
inline void DataflowBuffer::commit_implicit_write(uint32_t num_tiles) {
    // Runs once per transaction; the kernel said how many tiles it drained.
    const uint32_t block = num_tiles;
    implicit_txn_tiles_ = static_cast<uint16_t>(block);
    dfb_dm_advance_slot</*is_write=*/false>(local_dfb_interface_, block);
    ctiles_written_ += block;
    if (ctiles_written_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ctxn_id_index_ = (ctxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ctxn_id_loop_cnt_++;
    }
}

// Out-of-line definitions of Noc DFB-specific implicit-sync overloads.
// These are member functions of Noc but must be defined here because they need the complete
// DataflowBuffer type (circular dependency: dataflow_buffer.h includes noc.h, not vice versa).

template <NocOptions opts, typename Src>
std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_read(
    const Src& src,
    DataflowBuffer& dst,
    const typename noc_traits_t<Src>::src_args_type& src_args,
    const DataflowBufferArgs& dst_args) const {
    // Implicit sync reads land at get_noc_write_addr() and commit_implicit_read() advances the
    // cursor by whole entries; offset_bytes is ignored, so a non-zero offset would land data in
    // the wrong place while still posting whole-entry credits.
    ASSERT(dst_args.offset_bytes == 0);
    uint32_t txn_id = dst.prepare_implicit_read(dst_args.num_tiles);
    noc_async_read_set_trid(txn_id, noc_id_);
    while (noc_available_transactions(noc_id_, txn_id) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));
    // Move as many tiles as the kernel says one of its transactions carries.
    noc_async_read<NOC_MAX_BURST_SIZE + 1, true>(
        get_src_ptr<AddressType::NOC>(src, src_args),
        // Use cached addresses for NOC APIs
        dst.get_noc_write_addr(),
        dst.get_entry_size() * dst_args.num_tiles,
        noc_id_,
        NOC_UNICAST_WRITE_VC);
    dst.commit_implicit_read(dst_args.num_tiles);
}

template <NocOptions opts, typename Dst>
std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_write(
    DataflowBuffer& src,
    const Dst& dst,
    const DataflowBufferArgs& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    // Same contract as async_read above: implicit sync always transfers whole entries from
    // get_noc_read_addr() and ignores offset_bytes.
    ASSERT(src_args.offset_bytes == 0);
    uint32_t txn_id = src.prepare_implicit_write(src_args.num_tiles);
    // Use cached addresses for NOC APIs
    auto src_addr = src.get_noc_read_addr();
    auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
    // Drain as many tiles as the kernel says one of its transactions carries.
    const uint32_t txn_bytes = src.get_entry_size() * src_args.num_tiles;
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, txn_bytes, -1, false, noc_id_);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_noc_addr, src_addr, txn_bytes);
    // DPRINT("Issue the write\n");
    ncrisc_noc_fast_write_any_len<noc_mode, true, /*one_packet*/false>(
        noc_id_,
        write_cmd_buf,
        src_addr,
        dst_noc_addr,
        txn_bytes,
        NOC_UNICAST_WRITE_VC,
        false,   // mcast
        false,   // linked
        1,       // num_dests
        true,    // multicast_path_reserve
        false,   // posted == false (NocOptions::POSTED not set)
        txn_id);
    src.commit_implicit_write(src_args.num_tiles);
}

#else  // COMPILE_FOR_TRISC

template <bool>
inline DataflowBuffer::ScopedLockRegion DataflowBuffer::lock_acquire_impl(uint16_t) { return {}; }
template <bool>
inline void DataflowBuffer::lock_release_impl(ScopedLockRegion, uint16_t) {}

#endif  // !COMPILE_FOR_TRISC

#endif  // ARCH_QUASAR
