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
#include "api/compute/common_globals.h"
#include "internal/tt-2xx/quasar/tile_counters.h"
#endif

#include "experimental/lock.h"

namespace experimental {

class DataflowBuffer {
public:
    DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id];
        PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[0];
        uint8_t tc_id = get_counter_id(packed_tc);
        uint8_t tensix_id = get_tensix_id(packed_tc);
    }

    uint32_t get_entry_size() const { return g_dfb_interface[logical_dfb_id_].entry_size; }

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = g_dfb_interface[logical_dfb_id_].packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        PACK({
            uint16_t entries_freed;
            do {
                entries_freed = dfb_tile_counters[tc_id].f.acked;
            } while (entries_freed < num_entries);
        })
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        while (llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
#endif
    }

    void push_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id_];
        PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        PACK({
            dfb_tile_counters[tc_id].f.posted = num_entries;
            local_dfb_interface.wr_ptr[counter_idx_] += (num_entries * local_dfb_interface.stride_size);
            // DPRINT << "push_back: updated wr_ptr: " << local_dfb_interface.wr_ptr[counter_idx_] << ENDL();
            if (local_dfb_interface.wr_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
                local_dfb_interface.wr_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
            }

            counter_idx_ = (counter_idx_ + 1) % local_dfb_interface.num_tcs_to_rr;
            // DPRINT << "push_back: updated counter_idx: " << (uint32_t)counter_idx_ << ENDL();
        })
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        llk_intf_inc_posted(tensix_id, tc_id, num_entries);
        local_dfb_interface.wr_ptr[counter_idx_] += (num_entries * local_dfb_interface.stride_size);
        // DPRINT << "push_back: updated wr_ptr: " << local_dfb_interface.wr_ptr[counter_idx_] << ENDL();
        if (local_dfb_interface.wr_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
            local_dfb_interface.wr_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
        }

        counter_idx_ = (counter_idx_ + 1) % local_dfb_interface.num_tcs_to_rr;
        // DPRINT << "push_back: updated counter_idx: " << (uint32_t)counter_idx_ << ENDL();
#endif
    }

    void wait_front(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = g_dfb_interface[logical_dfb_id_].packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        UNPACK({
            uint16_t entries_received;
            do {
                entries_received = dfb_tile_counters[tc_id].f.posted;
            } while (entries_received < num_entries);
        })
#else
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
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id_];
        PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        UNPACK({
            dfb_tile_counters[tc_id].f.acked = num_entries;
            local_dfb_interface.rd_ptr[counter_idx_] += (num_entries * local_dfb_interface.stride_size);
            // DPRINT << "pop_front: updated rd_ptr: " << local_dfb_interface.rd_ptr[counter_idx_] << ENDL();
            if (local_dfb_interface.rd_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
                local_dfb_interface.rd_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
            }
            counter_idx_ = (counter_idx_ + 1) % local_dfb_interface.num_tcs_to_rr;
            // DPRINT << "pop_front: updated counter_idx: " << (uint32_t)counter_idx_ << ENDL();
        })
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        llk_intf_inc_acked(tensix_id, tc_id, num_entries);
        local_dfb_interface.rd_ptr[counter_idx_] += (num_entries * local_dfb_interface.stride_size);
        DPRINT << "pop_front: updated rd_ptr: " << local_dfb_interface.rd_ptr[counter_idx_] << ENDL();
        if (local_dfb_interface.rd_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
            local_dfb_interface.rd_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
        }
        counter_idx_ = (counter_idx_ + 1) % local_dfb_interface.num_tcs_to_rr;
        DPRINT << "pop_front: updated counter_idx: " << (uint32_t)counter_idx_ << ENDL();
#endif
    }
    // Explicit sync APIs end

    // Implicit sync APIs
    void read_in() {}

    void write_out() {}
    // Implicit sync APIs end

    // from pov of producer need to make sure all the entries get posted (check the raw posted per TC == raw acked per
    // TC)
    // also that there are no interrupts remaining...
    void finish() {
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id_];
        bool all_acked = false;
        while (!all_acked) {
            all_acked = true;
            for (uint8_t i = 0; i < local_dfb_interface.num_tcs_to_rr; i++) {
                PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[i];
                uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
                UNPACK({ all_acked = all_acked && (dfb_tile_counters[tc_id].f.posted == 0); })
#else
                uint8_t tensix_id = get_tensix_id(packed_tc);
                all_acked &=
                    (fast_llk_intf_read_acked(tensix_id, tc_id) == fast_llk_intf_read_posted(tensix_id, tc_id));
#endif
            }
        }
    }

    uint32_t get_write_ptr() const {
        // return byte address (wr_ptr is 16B address on Gen1XX)
        uint32_t wr_ptr_bytes = g_dfb_interface[logical_dfb_id_].wr_ptr[counter_idx_];
        return wr_ptr_bytes;
    }

    uint32_t get_read_ptr() const {
        // return byte address (rd_ptr is 16B address on Gen1XX)
        uint32_t rd_ptr_bytes = g_dfb_interface[logical_dfb_id_].rd_ptr[counter_idx_];
        return rd_ptr_bytes;
    }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    uint16_t logical_dfb_id_;
    uint8_t counter_idx_ = 0;

    // TODO: update txn id isr handling
    uint8_t txn_id_index_ = 0;
    uint32_t txn_id_loop_cnt_ = 0;  // try to remove this
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
