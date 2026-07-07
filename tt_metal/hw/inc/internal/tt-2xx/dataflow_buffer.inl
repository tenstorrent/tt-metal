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

inline DataflowBuffer::DataflowBuffer(uint16_t logical_dfb_id)
    : local_dfb_interface_(g_dfb_interface[logical_dfb_id]), logical_dfb_id_(logical_dfb_id) {}

inline uint32_t DataflowBuffer::get_entry_size() const { return local_dfb_interface_.entry_size; }

inline uint32_t DataflowBuffer::get_stride_size() const { return local_dfb_interface_.stride_size; }

inline uint32_t DataflowBuffer::get_total_num_entries() const { return local_dfb_interface_.num_entries; }

inline void DataflowBuffer::reserve_back_impl(uint16_t num_entries) {
    WAYPOINT("RBW");
    dfb::PackedTileCounter packed_tc =
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    llk_wait_for_free_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
        // DM-DM BLOCKED: wait until every consumer TC has free space (throttled by slowest consumer)
        bool ready = false;
        while (!ready) {
            ready = true;
            for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                ASSERT(overlay::llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
                if (overlay::llk_intf_get_free_space(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) < num_entries) {
                    ready = false;
                    break;
                }
            }
        }
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (overlay::llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
    }
#endif
    WAYPOINT("RBD");
}

inline void DataflowBuffer::push_back_impl(uint16_t num_entries) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
    llk_push_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
    if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
        // DM-DM BLOCKED: post to all N TCs; wr_ptr tracked on slot 0
        for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
            dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
            ASSERT(overlay::llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
            overlay::llk_intf_inc_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), num_entries);
        }
        local_dfb_interface_.tc_slots[0].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[0].wr_ptr >= local_dfb_interface_.tc_slots[0].limit) {
            local_dfb_interface_.tc_slots[0].wr_ptr = local_dfb_interface_.tc_slots[0].base_addr;
        }
        // tc_idx deliberately not advanced
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(overlay::llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        overlay::llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr >= local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
        }

        local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
    }
#endif
}

inline void DataflowBuffer::wait_front_impl(uint16_t num_entries) {
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
}

inline void DataflowBuffer::pop_front_impl(uint16_t num_entries) {
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
    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += (num_entries * local_dfb_interface_.stride_size);
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr >= local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
    }
    local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
#endif
}

inline void DataflowBuffer::finish_impl() {
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
            // TRISC drain: finish() must not return until this TC is empty (posted == 0) -
            // covers the consumer (UNPACK), the Tensix->DM producer (PACK), and both sides of
            // an INTRA self-loop. The consumer also skips TCs this TRISC doesn't own via
            // tensix_trisc_mask, which exists only in the UNPACK-side LocalDFBInterface, so the
            // gate sits under an inner UNPACK guard (the PACK struct has no such member).
#ifdef UCK_CHLKC_UNPACK
            if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
                continue;
            }
#endif
            all_acked = all_acked && (ckernel::trisc::tile_counters[tc_id].f.posted == 0);
#elif !defined(COMPILE_FOR_TRISC)
            uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
            all_acked &=
                (overlay::fast_llk_intf_read_acked(tensix_id, tc_id) == overlay::fast_llk_intf_read_posted(tensix_id, tc_id));
#endif
        }
    }
    WAYPOINT("AAD");
}

inline uint32_t DataflowBuffer::get_write_ptr_impl() const {
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr +
           local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_offset;
#elif !defined(COMPILE_FOR_TRISC)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr;
#else
    // Unpack TRISC does not use wr_ptr; return ring base for any accidental caller.
    ASSERT(false);
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
#endif
}

inline uint32_t DataflowBuffer::get_read_ptr_impl() const {
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr +
           local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_offset;
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
inline void DataflowBuffer::handle_final_credits(uint16_t transactions_issued, uint8_t txn_id_index) {
    // Determine the txn_id for the last batch. If transactions_issued lands exactly on
    // a boundary, txn_id_index has already wrapped past it, so step back one slot.
    uint8_t tail_txn_idx = (transactions_issued % local_dfb_interface_.num_entries_per_txn_id == 0)
                                ? static_cast<uint8_t>((txn_id_index + local_dfb_interface_.num_txn_ids - 1) % local_dfb_interface_.num_txn_ids)
                                : txn_id_index;
    uint8_t tail_txn_id = local_dfb_interface_.txn_ids[tail_txn_idx];

    uint8_t N = local_dfb_interface_.num_tcs_to_rr;
    dfb::PackedTileCounter ptc0 = local_dfb_interface_.tc_slots[0].packed_tile_counter;
    uint16_t expected_slot0 = transactions_issued / N + (0u < (transactions_issued % N) ? 1u : 0u);

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
            uint16_t expected = transactions_issued / N + (i < (transactions_issued % N) ? 1u : 0u);
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


// Consumer barrier: waits outbound write from DFB writes to arrive at their destination
// Falls back to a full barrier when no txn_ids are assigned
inline void DataflowBuffer::write_barrier_impl(const Noc &noc) const {
    if (local_dfb_interface_.num_txn_ids == 0) {
        noc.async_write_barrier();
        return;
    } else {
        for (uint8_t i = 0; i < local_dfb_interface_.num_txn_ids; i++) {
            noc.async_write_barrier<NocOptions::TXN_ID>({.trid = local_dfb_interface_.txn_ids[i]});
        }
    }
}

// Preamble for implicit-sync read: spin until previous reads are posted and there is space in the tile counters.
// Returns the txn_id to stamp on the next NOC read.
inline uint32_t DataflowBuffer::prepare_implicit_read() {
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
    while (overlay::fast_llk_intf_get_free_space(tensix_id, tc_id) < 1);
    WAYPOINT("PIRD");
    return txn_id;
}

// Postamble for implicit-sync read: advance wr_ptr, tile/txn counters, and tc_idx.
inline void DataflowBuffer::commit_implicit_read() {
    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += local_dfb_interface_.stride_size;
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr >=
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr =
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
    }
    ptiles_read_++;
    if (ptiles_read_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ptxn_id_index_ = (ptxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ptxn_id_loop_cnt_++;
    }
    local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
}

// Preamble for implicit-sync write: spin until previous writes are acked and data is available in the tile counters.
// Returns the txn_id to stamp on the next NOC write.
inline uint32_t DataflowBuffer::prepare_implicit_write() {
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
    while (overlay::fast_llk_intf_get_occupancy(tensix_id, tc_id) < 1);
    WAYPOINT("PIWD");
    return txn_id;
}

// Postamble for implicit-sync write: advance rd_ptr, tile/txn counters, and tc_idx.
inline void DataflowBuffer::commit_implicit_write() {
    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += local_dfb_interface_.stride_size;
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr >=
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr =
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
    }
    ctiles_written_++;
    if (ctiles_written_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ctxn_id_index_ = (ctxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ctxn_id_loop_cnt_++;
    }
    local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
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
    uint32_t txn_id = dst.prepare_implicit_read();
    noc_async_read_set_trid(txn_id, noc_id_);
    while (noc_available_transactions(noc_id_, txn_id) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));
    // DPRINT("Issue the read\n");
    noc_async_read<NOC_MAX_BURST_SIZE + 1, true>(
        get_src_ptr<AddressType::NOC>(src, src_args),
        dst.get_write_ptr(),
        dst.get_entry_size(),
        noc_id_,
        NOC_UNICAST_WRITE_VC);
    dst.commit_implicit_read();
}

template <NocOptions opts, typename Dst>
std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_write(
    DataflowBuffer& src,
    const Dst& dst,
    const DataflowBufferArgs& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    uint32_t txn_id = src.prepare_implicit_write();
    auto src_addr = src.get_read_ptr();
    auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size_bytes, -1, posted, noc_id_);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_noc_addr, src_addr, src.get_entry_size());
    // DPRINT("Issue the write\n");
    ncrisc_noc_fast_write_any_len<noc_mode, true, /*one_packet*/false>(
        noc_id_,
        write_cmd_buf,
        src_addr,
        dst_noc_addr,
        src.get_entry_size(),
        NOC_UNICAST_WRITE_VC,
        false,   // mcast
        false,   // linked
        1,       // num_dests
        true,    // multicast_path_reserve
        false,   // posted == false (NocOptions::POSTED not set)
        txn_id);
    src.commit_implicit_write();
}

#endif  // !COMPILE_FOR_TRISC

#endif  // ARCH_QUASAR
