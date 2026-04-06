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

namespace experimental {

inline DataflowBuffer::DataflowBuffer(uint16_t logical_dfb_id)
    : local_dfb_interface_(g_dfb_interface[logical_dfb_id]), logical_dfb_id_(logical_dfb_id) {}

inline uint32_t DataflowBuffer::get_entry_size() const { return local_dfb_interface_.entry_size; }

inline uint32_t DataflowBuffer::get_stride_size() const { return local_dfb_interface_.stride_size; }

inline void DataflowBuffer::reserve_back_impl(uint16_t num_entries) {
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
                ASSERT(llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
                if (llk_intf_get_free_space(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) < num_entries) {
                    ready = false;
                    break;
                }
            }
        }
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
    }
#endif
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
            ASSERT(llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
            llk_intf_inc_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), num_entries);
        }
        local_dfb_interface_.tc_slots[0].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[0].wr_ptr == local_dfb_interface_.tc_slots[0].limit) {
            local_dfb_interface_.tc_slots[0].wr_ptr = local_dfb_interface_.tc_slots[0].base_addr;
        }
        // tc_idx deliberately not advanced
    } else {
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        ASSERT(llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
        }

        local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
    }
#endif
}

inline void DataflowBuffer::wait_front_impl(uint16_t num_entries) {
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
    ASSERT(llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
    while (llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
#endif
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
    ASSERT(llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
    llk_intf_inc_acked(tensix_id, tc_id, num_entries);
    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += (num_entries * local_dfb_interface_.stride_size);
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
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
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
            if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
                continue;
            }
            all_acked = all_acked && (ckernel::trisc::tile_counters[tc_id].f.posted == 0);
#elif !defined(COMPILE_FOR_TRISC)
            uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
            all_acked &=
                (fast_llk_intf_read_acked(tensix_id, tc_id) == fast_llk_intf_read_posted(tensix_id, tc_id));
#endif
        }
    }
    WAYPOINT("AAD");
}

inline uint32_t DataflowBuffer::get_write_ptr_impl() const {
    // return byte address (wr_ptr is 16B address on Gen1XX)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr;
}

inline uint32_t DataflowBuffer::get_read_ptr_impl() const {
    // return byte address (rd_ptr is 16B address on Gen1XX)
    return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr;
}

#ifndef COMPILE_FOR_TRISC

template <typename Src>
inline void DataflowBuffer::read_in(
    const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);

    // Wait for entries that were previously read across all transaction ids to be posted. Need to do this because HW
    // doesn't track pending posts. When this condition is met, we know previous reads were committed.
    while (fast_llk_intf_read_posted(tensix_id, tc_id) < (ptxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc));

    // Make sure there is space for the new tile
    while (fast_llk_intf_get_free_space(tensix_id, tc_id) < 1);

    noc.async_read<Noc::TxnIdMode::ENABLED>(src, *this, get_entry_size(), src_args, {}, NOC_UNICAST_WRITE_VC, local_dfb_interface_.txn_ids[ptxn_id_index_]);

    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (local_dfb_interface_.stride_size);
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
    }

    ptiles_read_++;
    // Move to next txn id when we have read in num tiles per DM producer.
    // This is safe because we ensure previously read entries are posted before reading in more data
    if (ptiles_read_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ptxn_id_index_ = (ptxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ptxn_id_loop_cnt_++;
    }

    local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
}

template <typename Dst>
inline void DataflowBuffer::write_out(
    const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
    dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
    uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
    uint8_t tc_id = dfb::get_counter_id(packed_tc);

    // Wait for entries that were previously written across all transaction ids to be acked. Need to do this because HW
    // doesn't track pending acks. When this condition is met, we know previous writes were issued.
    while (fast_llk_intf_read_acked(tensix_id, tc_id) < (ctxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc));

    while (fast_llk_intf_get_occupancy(tensix_id, tc_id) < 1);

    noc.async_write<Noc::TxnIdMode::ENABLED>(*this, dst, get_entry_size(), {}, dst_args, NOC_UNICAST_WRITE_VC, local_dfb_interface_.txn_ids[ctxn_id_index_]);

    local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += (local_dfb_interface_.stride_size);
    if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
    }

    ctiles_written_++;
    // Move to next txn id when the DM has written its threshold per transaction id.
    // This is safe because we ensure previous writes are acked before trying to write more data
    if (ctiles_written_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
        ctxn_id_index_ = (ctxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        ctxn_id_loop_cnt_++;
    }

    local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
}

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
                fast_llk_intf_read_posted(dfb::get_tensix_id(ptc0), dfb::get_counter_id(ptc0)));
        } else {
            return static_cast<uint16_t>(
                fast_llk_intf_read_acked(dfb::get_tensix_id(ptc0), dfb::get_counter_id(ptc0)));
        }
    };

    uint64_t threshold;
    if constexpr (is_producer) {
        threshold = GET_TILES_TO_PROCESS_THRES_TR_ACK(tail_txn_id);
    } else {
        threshold = GET_TILES_TO_PROCESS_THRES_WR_SENT(tail_txn_id);
    }

    // Wait until this DM's tail reads have completed.
    // A transaction passes through three observable states:
    //   not dispatched → tack == 0, tiles == 0
    //   in-flight      → tack >  0
    //   completed      → tack == 0, tiles >  0   ← break here
    // Also exits early if the ISR fires (collective batch done).
    WAYPOINT("WTP1");
    while (read_actual_slot0() < expected_slot0) {
        uint64_t tack, tiles;
        if constexpr (is_producer) {
            tack  = CMDBUF_TR_ACK_TRID(OVERLAY_RD_CMD_BUF, tail_txn_id);
            tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tack  = CMDBUF_WR_SENT_TRID(OVERLAY_WR_CMD_BUF, tail_txn_id);
            tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
        }
        if (tack == 0 && tiles > 0) break;
    }

    // Transactions are completed. Spin giving the ISR a chance to fire.
    // Break when tiles < threshold which is a genuine partial tail the ISR can never handle.
    WAYPOINT("WTP2");
    while (read_actual_slot0() < expected_slot0) {
        uint64_t tiles;
        if constexpr (is_producer) {
            tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
        } else {
            tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
        }
        if (tiles > 0 && tiles < threshold) break;
    }

    // Manually post missing credits if ISR did not fire.
    uint16_t actual_slot0 = read_actual_slot0();
    if (actual_slot0 < expected_slot0) {
        for (uint8_t i = 0; i < N; i++) {
            dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
            uint8_t tensix_id = dfb::get_tensix_id(ptc);
            uint8_t tc_id     = dfb::get_counter_id(ptc);
            uint16_t expected = transactions_issued / N + (i < (transactions_issued % N) ? 1u : 0u);
            if constexpr (is_producer) {
                uint16_t actual = static_cast<uint16_t>(fast_llk_intf_read_posted(tensix_id, tc_id));
                if (actual < expected) {
                    fast_llk_intf_inc_posted(tensix_id, tc_id, expected - actual);
                }
            } else {
                uint16_t actual = static_cast<uint16_t>(fast_llk_intf_read_acked(tensix_id, tc_id));
                if (actual < expected) {
                    fast_llk_intf_inc_acked(tensix_id, tc_id, expected - actual);
                }
            }
        }
    }
}

#endif  // !COMPILE_FOR_TRISC

}  // namespace experimental

#endif  // ARCH_QUASAR
