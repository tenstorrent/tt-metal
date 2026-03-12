// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/dataflow_buffer_interface.h"
#include "internal/dataflow_buffer_init.h"  // For g_dfb_interface extern declaration
#include "api/debug/assert.h"

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

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
        llk_wait_for_free_tiles(logical_dfb_id_, num_entries);
        // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " acked: " << static_cast<uint32_t>(tile_counters[tc_id].f.acked) << ENDL();
#elif !defined(COMPILE_FOR_TRISC)
        if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
            // DM-DM BLOCKED: wait until every consumer TC has free space (throttled by slowest consumer)
            bool ready = false;
            while (!ready) {
                ready = true;
                for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                    PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                    // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " free space: " << static_cast<uint32_t>(llk_intf_get_free_space(get_tensix_id(ptc), get_counter_id(ptc))) << ENDL();
                    if (llk_intf_get_free_space(get_tensix_id(ptc), get_counter_id(ptc)) < num_entries) {
                        ready = false;
                        break;
                    }
                }
            }
        } else {
            uint8_t tensix_id = get_tensix_id(packed_tc);
            while (llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
            // DPRINT << "reserve_back: tc_id: " << static_cast<uint32_t>(tc_id) << " free space: " << static_cast<uint32_t>(llk_intf_get_free_space(tensix_id, tc_id)) << ENDL();
        }
#endif
    }

    void push_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
        llk_push_tiles(logical_dfb_id_, num_entries);
        // DPRINT << "push_bak: tc_id: " << static_cast<uint32_t>(tc_id) << " posted: " << static_cast<uint32_t>(tile_counters[tc_id].f.posted) << ENDL();
#elif !defined(COMPILE_FOR_TRISC)
        if (__builtin_expect(local_dfb_interface_.broadcast_tc, 0)) {
            // DM-DM BLOCKED: post to all N TCs; wr_ptr tracked on slot 0
            for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                PackedTileCounter ptc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                // DPRINT << "push_back: tc_id: " << static_cast<uint32_t>(tc_id) << " posted: " << static_cast<uint32_t>(llk_intf_get_posted(get_tensix_id(ptc), get_counter_id(ptc))) << ENDL();
                llk_intf_inc_posted(get_tensix_id(ptc), get_counter_id(ptc), num_entries);
            }
            local_dfb_interface_.tc_slots[0].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
            if (local_dfb_interface_.tc_slots[0].wr_ptr == local_dfb_interface_.tc_slots[0].limit) {
                local_dfb_interface_.tc_slots[0].wr_ptr = local_dfb_interface_.tc_slots[0].base_addr;
            }
            // tc_idx deliberately not advanced
        } else {
            uint8_t tensix_id = get_tensix_id(packed_tc);
            llk_intf_inc_posted(tensix_id, tc_id, num_entries);
            // DPRINT << "push_back: tensix_id: " << static_cast<uint32_t>(tensix_id) << " tc_id: " << static_cast<uint32_t>(tc_id) << " capacity: "
            //         << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id))
            //         << " posted: " << static_cast<uint32_t>(llk_intf_get_posted(tensix_id, tc_id)) << ENDL();

            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (num_entries * local_dfb_interface_.stride_size);
            if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
                local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
            }

            local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
        }
#endif
    }

    void wait_front(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
        if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
            return;
        }
        llk_wait_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
        uint8_t tensix_id = get_tensix_id(packed_tc);
        // DPRINT << "wait_front: tensix_id: " << static_cast<uint32_t>(tensix_id)
        //        << " capacity: " << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id))
        //        << " tc_id: " << static_cast<uint32_t>(tc_id)
        //        << " occupancy: " << static_cast<uint32_t>(llk_intf_get_occupancy(tensix_id, tc_id)) << ENDL();
        while (llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
#endif
    }

    void pop_front(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tc_id = get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
        if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
            return;
        }
        llk_pop_tiles(logical_dfb_id_, num_entries);
#elif !defined(COMPILE_FOR_TRISC)
        uint8_t tensix_id = get_tensix_id(packed_tc);
        llk_intf_inc_acked(tensix_id, tc_id, num_entries);
        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += (num_entries * local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
        }
        local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
        // DPRINT << "pop_front: free space: " << (uint32_t)llk_intf_get_free_space(tensix_id, tc_id) << ENDL();
#endif
    }
    // Explicit sync APIs end

    // Implicit sync APIs
    // one tile at a time right now
#ifndef COMPILE_FOR_TRISC
    template <typename Src>
    void read_in(const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args) {
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tensix_id = get_tensix_id(packed_tc);
        uint8_t tc_id = get_counter_id(packed_tc);
        DPRINT << "Read in tensix_id: " << static_cast<uint32_t>(tensix_id) << " tc_id: " << static_cast<uint32_t>(tc_id) << ENDL();

        // Wait for entries that were previously read in to be posted. Need to do this because HW doesn't track pending posts
        DPRINT << "Read in waiting for total posted to be " << (ptxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc)
               << " and current posted is " << static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id)) << ENDL();
        while (fast_llk_intf_read_posted(tensix_id, tc_id) < (ptxn_id_loop_cnt_ * local_dfb_interface_.num_entries_per_txn_id_per_tc));

        // make sure there is space for the new tile
        DPRINT << "Free space is " << static_cast<uint32_t>(fast_llk_intf_get_free_space(tensix_id, tc_id)) << ENDL();
        while (fast_llk_intf_get_free_space(tensix_id, tc_id) < 1);

        // ISSUE THE READ
        DPRINT << "read_in: issuing read with txn_id: " << static_cast<uint32_t>(local_dfb_interface_.txn_ids[ptxn_id_index_]) << ENDL();
        noc.async_read<Noc::TxnIdMode::ENABLED>(src, *this, get_entry_size(), src_args, {}, NOC_UNICAST_WRITE_VC, local_dfb_interface_.txn_ids[ptxn_id_index_]);

        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr += (local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
        }

        ptiles_read_++;
       // we need to move to "next" txn id when we have read in num tiles per dm producer
        if (ptiles_read_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
            ptxn_id_index_ = (ptxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
            ptxn_id_loop_cnt_++;
        }

        local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
    }

    template <typename Dst>
    void write_out(const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
        PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].packed_tile_counter;
        uint8_t tensix_id = get_tensix_id(packed_tc);
        uint8_t tc_id = get_counter_id(packed_tc);
        DPRINT << "Write out tensix_id: " << static_cast<uint32_t>(tensix_id) << " tc_id: " << static_cast<uint32_t>(tc_id) << ENDL();

        DPRINT << "Read acked is " << static_cast<uint32_t>(fast_llk_intf_read_acked(tensix_id, tc_id)) << ENDL();

        DPRINT << "Write out waiting for occupancy to be more than 1 and current occupancy is " << static_cast<uint32_t>(fast_llk_intf_get_occupancy(tensix_id, tc_id)) << ENDL();
        while (fast_llk_intf_get_occupancy(tensix_id, tc_id) < 1);

        // ISSUE THE WRITE
        DPRINT << "Write out issuing write with txn_id: " << static_cast<uint32_t>(local_dfb_interface_.txn_ids[ctxn_id_index_]) << ENDL();
        noc.async_write<Noc::TxnIdMode::ENABLED>(*this, dst, get_entry_size(), {}, dst_args, NOC_UNICAST_WRITE_VC, local_dfb_interface_.txn_ids[ctxn_id_index_]);

        local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr += (local_dfb_interface_.stride_size);
        if (local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr == local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].limit) {
            local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr = local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].base_addr;
        }

        ctiles_written_++;
        if (ctiles_written_ % local_dfb_interface_.num_entries_per_txn_id == 0) {
            ctxn_id_index_ = (ctxn_id_index_ + 1) % local_dfb_interface_.num_txn_ids;
        }

        local_dfb_interface_.tc_idx = (local_dfb_interface_.tc_idx + 1) % local_dfb_interface_.num_tcs_to_rr;
    }
    // Implicit sync APIs end
#endif

    // from pov of producer need to make sure all the entries get posted (check the raw posted per TC == raw acked per
    // TC)
    // also that there are no interrupts remaining...
    void finish() {
        bool all_acked = false;
        while (!all_acked) {
            all_acked = true;
            for (uint8_t i = 0; i < local_dfb_interface_.num_tcs_to_rr; i++) {
                PackedTileCounter packed_tc = local_dfb_interface_.tc_slots[i].packed_tile_counter;
                uint8_t tc_id = get_counter_id(packed_tc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_UNPACK)
                if ((local_dfb_interface_.tensix_trisc_mask & (1u << ckernel::csr_read<ckernel::CSR::TRISC_ID>())) == 0) {
                    continue;
                }
                all_acked = all_acked && (ckernel::trisc::tile_counters[tc_id].f.posted == 0);
#elif !defined(COMPILE_FOR_TRISC)
                uint8_t tensix_id = get_tensix_id(packed_tc);
                // DPRINT << "read acked: " << static_cast<uint32_t>(fast_llk_intf_read_acked(tensix_id, tc_id)) << " read posted: " << static_cast<uint32_t>(fast_llk_intf_read_posted(tensix_id, tc_id)) << ENDL();
                all_acked &=
                    (fast_llk_intf_read_acked(tensix_id, tc_id) == fast_llk_intf_read_posted(tensix_id, tc_id));
#endif
            }
        }
    }

    uint32_t get_write_ptr() const {
        // return byte address (wr_ptr is 16B address on Gen1XX)
        return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].wr_ptr + MEM_L1_UNCACHED_BASE;
    }

    uint32_t get_read_ptr() const {
        // return byte address (rd_ptr is 16B address on Gen1XX)
        return local_dfb_interface_.tc_slots[local_dfb_interface_.tc_idx].rd_ptr + MEM_L1_UNCACHED_BASE;
    }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    LocalDFBInterface& local_dfb_interface_;

    uint16_t logical_dfb_id_;

    // Metadata for implicit sync
    uint16_t ptxn_id_loop_cnt_ = 0;
    uint8_t ptxn_id_index_ = 0;
    uint16_t ptiles_read_ = 0; // isn't the same as reading the tile counter because we don't have a way of tracking pending posts from HW

    uint8_t ctxn_id_index_ = 0;
    uint16_t ctiles_written_ = 0; // isn't the same as reading the tile counter because we don't have a way of tracking pending aks from HW
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
