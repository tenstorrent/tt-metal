// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h"  // For g_dfb_interface extern declaration
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"

// TODO: make this the top level api header but then separate out 1xx and 2xx implementations

#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#include "experimental/noc.h"
#else
#include "ckernel_trisc_common.h"
#ifdef UCK_CHLKC_PACK
#include "llk_io_pack.h"
#endif
#ifdef UCK_CHLKC_UNPACK
#include "llk_io_unpack.h"
#endif
#endif

#include "experimental/lock.h"

namespace experimental {

class DataflowBuffer {
public:
    DataflowBuffer(uint16_t logical_dfb_id) : local_dfb_interface_(g_dfb_interface[logical_dfb_id]), logical_dfb_id_(logical_dfb_id) {}

    uint16_t get_id() const { return logical_dfb_id_; }

    uint32_t get_entry_size() const { return local_dfb_interface_.entry_size; }

    uint32_t get_stride_size() const { return local_dfb_interface_.stride_size; }

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) {
        dfb::PackedTileCounter packed_tc =
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
        ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
        llk_wait_for_free_tiles(logical_dfb_id_, num_entries);
        // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " acked: " <<
        // static_cast<uint32_t>(tile_counters[tc_id].f.acked) << ENDL(); DEVICE_PRINT("reserve_back: tc_id: {} acked:
        // {}\n", static_cast<uint32_t>(tc_id), static_cast<uint32_t>(tile_counters[tc_id].f.acked));
#elif !defined(COMPILE_FOR_TRISC)
        if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
            // DM-DM BLOCKED: wait until every consumer TC has free space (throttled by slowest consumer)
            bool ready = false;
            while (!ready) {
                ready = true;
                for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                    dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                    ASSERT(llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
                    // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " free space: " <<
                    // static_cast<uint32_t>(llk_intf_get_free_space(get_tensix_id(ptc), get_counter_id(ptc))) <<
                    // ENDL(); DEVICE_PRINT("reserve_back: tc_id: {} free space: {}\n", tc_id,
                    // static_cast<uint32_t>(llk_intf_get_free_space(get_tensix_id(ptc), get_counter_id(ptc))));
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
            // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " free space: " <<
            // static_cast<uint32_t>(llk_intf_get_free_space(tensix_id, tc_id)) << ENDL(); DEVICE_PRINT("reserve_back:
            // tc_id: {} free space: {}\n", tc_id, static_cast<uint32_t>(llk_intf_get_free_space(tensix_id, tc_id)));
        }
#endif
    }

    void push_back(uint16_t num_entries) {
        dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
        ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
        llk_push_tiles(logical_dfb_id_, num_entries);
        // DPRINT << "push_bak: tc_id: " << static_cast<uint32_t>(tc_id) << " posted: " <<
        // static_cast<uint32_t>(tile_counters[tc_id].f.posted) << ENDL(); DEVICE_PRINT("push_back: tc_id: {} posted:
        // {}\n", tc_id, static_cast<uint32_t>(tile_counters[tc_id].f.posted));
#elif !defined(COMPILE_FOR_TRISC)
        if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
            // DM-DM BLOCKED: post to all N TCs; wr_ptr tracked on slot 0
            for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                ASSERT(llk_intf_get_capacity(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc)) >= num_entries);
                // DPRINT << "push_back: tc_id: " << static_cast<uint32_t>(tc_id) << " posted: " <<
                // static_cast<uint32_t>(llk_intf_get_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc))) <<
                // ENDL(); DEVICE_PRINT("push_back: tc_id: {} posted: {}\n", tc_id,
                // static_cast<uint32_t>(llk_intf_get_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc))));
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
            DPRINT << "push_back: tensix_id: " << static_cast<uint32_t>(tensix_id)
                   << " tc_id: " << static_cast<uint32_t>(tc_id)
                   << " capacity: " << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id))
                   << " posted: " << static_cast<uint32_t>(llk_intf_get_posted(tensix_id, tc_id)) << ENDL();
            DEVICE_PRINT(
                "push_back: tensix_id: {} tc_id: {} capacity: {} posted: {}\n",
                tensix_id,
                tc_id,
                static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id)),
                static_cast<uint32_t>(llk_intf_get_posted(tensix_id, tc_id)));

            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
            if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
                local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
            }

            local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
        }
#endif
    }

    void wait_front(uint16_t num_entries) {
        dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = dfb::get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
        ASSERT(ckernel::trisc::tile_counters[tc_id].f.buf_capacity >= num_entries);
        if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
            return;
        }
        DPRINT << "wait_front: tc_id: " << static_cast<uint32_t>(tc_id)
               << " num_entries: " << static_cast<uint32_t>(num_entries) << ENDL();
        DEVICE_PRINT("wait_front: tc_id: {} num_entries: {}\n", tc_id, num_entries);
        llk_wait_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        // DPRINT << "wait_front: tensix_id: " << static_cast<uint32_t>(tensix_id)
        //        << " capacity: " << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id))
        //        << " tc_id: " << static_cast<uint32_t>(tc_id)
        //        << " occupancy: " << static_cast<uint32_t>(llk_intf_get_occupancy(tensix_id, tc_id)) << ENDL();
        // DEVICE_PRINT("wait_front: tensix_id: {} capacity: {} tc_id: {} occupancy: {}\n", tensix_id,
        // static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id)), tc_id,
        // static_cast<uint32_t>(llk_intf_get_occupancy(tensix_id, tc_id)));
        ASSERT(llk_intf_get_capacity(tensix_id, tc_id) >= num_entries);
        while (llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
#endif
    }

    void pop_front(uint16_t num_entries) {
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
        // DPRINT << "pop_front: free space: " << (uint32_t)llk_intf_get_free_space(tensix_id, tc_id) << ENDL();
        // DEVICE_PRINT("pop_front: free space: {}\n", (uint32_t)llk_intf_get_free_space(tensix_id, tc_id));
#endif
    }
    // Explicit sync APIs end

    // Implicit sync APIs
    // one tile at a time right now
#ifndef COMPILE_FOR_TRISC
    template <typename Src>
    void read_in(const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args) {
        dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        uint8_t tc_id = dfb::get_counter_id(packed_tc);

        // Wait for entries that were previously read across all transaction ids to be posted. Need to do this because HW doesn't track pending posts
        // When this condition is met, we know previous reads were committed
        DPRINT << "read_in: tensix_id: " << static_cast<uint32_t>(tensix_id)
               << " tc_id: " << static_cast<uint32_t>(tc_id)
               << " posted: " << static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id)) << ENDL();
        DEVICE_PRINT(
            "read_in: tensix_id: {} tc_id: {} posted: {}\n",
            tensix_id,
            tc_id,
            static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id)));
        while (fast_llk_intf_read_posted(tensix_id, tc_id) < (ptxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc));

        // Make sure there is space for the new tile
        while (fast_llk_intf_get_free_space(tensix_id, tc_id) < 1);

        // DPRINT << "issuing the read on " << static_cast<uint32_t>(local_dfb_interface_.txn_ids[ptxn_id_index_]) <<
        // ENDL();
        // DEVICE_PRINT("issuing the read on {}\n", local_dfb_interface_.txn_ids[ptxn_id_index_]);

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
    void write_out(const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
        dfb::PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
        uint8_t tc_id = dfb::get_counter_id(packed_tc);


        // Wait for entries that were previously written across all transaction ids to be acked. Need to do this because HW doesn't track pending acks
        // When this condition is met, we know previous writes were issued
        while (fast_llk_intf_read_acked(tensix_id, tc_id) < (ctxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc));

        while (fast_llk_intf_get_occupancy(tensix_id, tc_id) < 1);

        // DPRINT << "issuing the write on " << static_cast<uint32_t>(local_dfb_interface_.txn_ids[ctxn_id_index_]) <<
        // ENDL();
        // DEVICE_PRINT("issuing the write on {}\n", local_dfb_interface_.txn_ids[ctxn_id_index_]);

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
    // Implicit sync APIs end
#endif

    void finish() {
#ifndef COMPILE_FOR_TRISC
        // Handle case where outstanding transactions do not meet ISR threshold
        // Each DM updates tile counters for the tiles it read/wrote by using its local counter
        // DPRINT << "ptiles_read: " << static_cast<uint32_t>(ptiles_read_) << " ctiles_written: " <<
        // static_cast<uint32_t>(ctiles_written_) << ENDL();
        // DEVICE_PRINT("ptiles_read: {} ctiles_written: {}\n", ptiles_read_, ctiles_written_);
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
                DPRINT << "read acked: " << static_cast<uint32_t>(fast_llk_intf_read_acked(tensix_id, tc_id))
                       << " read posted: " << static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id))
                       << ENDL();
                DEVICE_PRINT(
                    "read acked: {} read posted: {}\n",
                    static_cast<uint32_t>(fast_llk_intf_read_acked(tensix_id, tc_id)),
                    static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id)));
                all_acked &=
                    (fast_llk_intf_read_acked(tensix_id, tc_id) == fast_llk_intf_read_posted(tensix_id, tc_id));
#endif
            }
        }
        WAYPOINT("AAD");
    }

    uint32_t get_write_ptr() const {
        // return byte address (wr_ptr is 16B address on Gen1XX)
        return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr;
    }

    uint32_t get_read_ptr() const {
        // return byte address (rd_ptr is 16B address on Gen1XX)
        return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr;
    }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    // Handle case when final transactions issued by DMs do not meet ISR threshold by manually posting/acking credits.
    // Each DM independently enters finish() so this needs to ensure all DMs have completed their final transactions
    // to avoid manually posting/acking credits that would be handled by the ISR.
    template <bool is_producer>
    void handle_final_credits(uint16_t transactions_issued, uint8_t txn_id_index) {
#ifndef COMPILE_FOR_TRISC
        // Determine the txn_id for the last batch. If transactions_issued lands exactly on
        // a boundary, txn_id_index has already wrapped past it, so step back one slot.
        uint8_t tail_txn_idx =
            (transactions_issued % local_dfb_interface_.num_entries_per_txn_id == 0)
                ? static_cast<uint8_t>(
                      (txn_id_index + local_dfb_interface_.num_txn_ids - 1) % local_dfb_interface_.num_txn_ids)
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
                tack = CMDBUF_TR_ACK_TRID(OVERLAY_RD_CMD_BUF, tail_txn_id);
                tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
            } else {
                tack = CMDBUF_WR_SENT_TRID(OVERLAY_WR_CMD_BUF, tail_txn_id);
                tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
            }
            if (tack == 0 && tiles > 0) {
                break;
            }
        }

        // Transactions are completed. Spin giving the ISR a chance to fire
        // Break when tiles < threshold which is a genuine partial tail the ISR can never handle.
        WAYPOINT("WTP2");
        while (read_actual_slot0() < expected_slot0) {
            uint64_t tiles;
            if constexpr (is_producer) {
                tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
            } else {
                tiles = CMDBUF_READ_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, tail_txn_id);
            }
            if (tiles > 0 && tiles < threshold) {
                break;
            }
        }

        // Manually post missing credits if ISR did not fire.
        uint16_t actual_slot0 = read_actual_slot0();
        // DPRINT << "actual_slot0: " << static_cast<uint32_t>(actual_slot0)
        //        << " expected_slot0: " << static_cast<uint32_t>(expected_slot0) << ENDL();
        // DEVICE_PRINT("actual_slot0: {} expected_slot0: {}\n", actual_slot0, expected_slot0);

        if (actual_slot0 < expected_slot0) {
            for (uint8_t i = 0; i < N; i++) {
                dfb::PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                uint8_t tensix_id = dfb::get_tensix_id(ptc);
                uint8_t tc_id = dfb::get_counter_id(ptc);
                uint16_t expected = transactions_issued / N + (i < (transactions_issued % N) ? 1u : 0u);
                if constexpr (is_producer) {
                    uint16_t actual = static_cast<uint16_t>(fast_llk_intf_read_posted(tensix_id, tc_id));
                    if (actual < expected) {
                        // DPRINT << "inc_posted tc(" << static_cast<uint32_t>(tensix_id) << "," <<
                        // static_cast<uint32_t>(tc_id)
                        //        << ") delta: " << static_cast<uint32_t>(expected - actual) << ENDL();
                        // DEVICE_PRINT("inc_posted tc({}, {}) delta: {}\n", tensix_id, tc_id, expected - actual);
                        fast_llk_intf_inc_posted(tensix_id, tc_id, expected - actual);
                    }
                } else {
                    uint16_t actual = static_cast<uint16_t>(fast_llk_intf_read_acked(tensix_id, tc_id));
                    if (actual < expected) {
                        // DPRINT << "inc_acked tc(" << static_cast<uint32_t>(tensix_id) << "," <<
                        // static_cast<uint32_t>(tc_id)
                        //        << ") delta: " << static_cast<uint32_t>(expected - actual) << ENDL();
                        // DEVICE_PRINT("inc_acked tc({}, {}) delta: {}\n", tensix_id, tc_id, expected - actual);
                        fast_llk_intf_inc_acked(tensix_id, tc_id, expected - actual);
                    }
                }
            }
        }
#endif
    }

    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    LocalDFBInterface& local_dfb_interface_;

    uint16_t logical_dfb_id_;

    // Metadata for implicit sync
    uint16_t ptxn_id_loop_cnt_ = 0;
    uint8_t ptxn_id_index_ = 0;
    uint16_t ptiles_read_ = 0; // isn't the same as reading the tile counter because we don't have a way of tracking pending posts from HW

    uint16_t ctxn_id_loop_cnt_ = 0;
    uint8_t ctxn_id_index_ = 0;
    uint16_t ctiles_written_ = 0; // isn't the same as reading the tile counter because we don't have a way of tracking pending acks from HW
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
            "DataflowBuffer without mcast range can only be used as L1 source");
        return dst.get_write_ptr() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const DataflowBuffer& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(
            address_type == Noc::AddressType::NOC, "DataflowBuffer with mcast range cannot be used as L1 source");
        auto local_addr = dst.get_write_ptr() + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, local_addr, noc.get_noc_id());
    }
};

#endif

}  // namespace experimental
