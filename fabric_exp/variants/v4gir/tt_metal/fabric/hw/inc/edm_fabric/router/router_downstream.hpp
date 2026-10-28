// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Downstream connections: the routers on this chip that a receiver channel forwards packets to (one per other
// direction in 2D, one in 1D). For each, the receiver keeps
//   - a cursor: the next slot in the downstream sender channel's ring and the number of slots known free there;
//   - the link: the ring's base and the downstream's NoC coordinate and credit register (from runtime args, set at init).
//
// Credits. The downstream's sender channel has a free-slots stream register; this router keeps a local copy of it,
// which the downstream increments as it frees slots. A forward decrements both (the downstream's is its doorbell), and
// the cursor's known-free count is a lower bound of the local copy, re-read only when it reaches zero. So admission
// usually costs no register read (cached downstream credits, LEDGER E1-2: forwarding and multicast -4 to -8).
//
// Forwards go through the NoC API. A router with a single live downstream (E/W-facing, 2D) keeps its forwarding command
// buffers programmed with that downstream's coordinate and credit register for the whole run (stateful API); every
// other forward programs the destination per write (stateless API).
//
// Names and entries. The receiver loop names a downstream as routing does: by its direction in 2D (so a packet's action
// bits are its downstreams), 0 for 1D's one downstream. The tables below keep the host's compact order (the runtime
// args list the downstreams that way); downstream_entry maps a name to its entry, at compile time.

#include "router_common.hpp"

namespace fabric_router {

template <size_t VC>
struct DownstreamTraits {
    static constexpr size_t size = VC == 0 ? VC0_DOWNSTREAM_EDM_SIZE : VC1_DOWNSTREAM_EDM_SIZE;
    static constexpr uint32_t num_slots = VC == 0 ? DOWNSTREAM_SENDER_NUM_BUFFERS_VC0 : DOWNSTREAM_SENDER_NUM_BUFFERS_VC1;
    static constexpr uint32_t ring_bytes = (num_slots - 1) * channel_buffer_size;
    static constexpr uint8_t data_cmd_buf = receiver_channel_forwarding_data_cmd_buf_ids[VC];
    static constexpr uint8_t sync_cmd_buf = receiver_channel_forwarding_sync_cmd_buf_ids[VC];
    static constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;

    template <size_t COMPACT>
    static constexpr uint32_t local_credit_stream = local_downstream_credit_stream(VC, COMPACT);
    template <size_t COMPACT>
    static constexpr uint32_t remote_credit_stream = downstream_sender_free_slots_stream(VC, COMPACT);

    static_assert(size > 0 && size <= max_downstreams);
    static_assert(num_slots <= stream_count_mask);
};

// The downstreams a router can forward to, one bit per name. In 2D, the directions with an entry (the host lists
// DownstreamTraits::size of them); an E/W-facing router inside a mesh never turns a packet onto N/S (packets travel Y
// first), so its N/S downstreams are not live. Intermesh routers (a landing decodes Y first) and routers on a mesh's
// edge (an exit may leave in any direction) can turn onto N/S. In 1D, downstream 0.
template <size_t VC>
constexpr uint32_t live_downstreams() {
    if constexpr (!is_2d) {
        return 0x1u;
    } else {
        constexpr bool ew = my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST;
        constexpr bool turns = is_intermesh_router || is_intramesh_router_on_edge;
        uint32_t live = 0;
        for (size_t entry = 0; entry < DownstreamTraits<VC>::size; entry++) {
            const auto dir = downstream_direction(entry);
            if (!(ew && !turns && (dir == eth_chan_directions::NORTH || dir == eth_chan_directions::SOUTH))) {
                live |= 1u << dir;
            }
        }
        return live;
    }
}

// Whether VC's forwards keep a preset destination in the command buffers: when the VC has exactly one live downstream
// (every 1D router; an E/W-facing 2D router inside a mesh), its coordinate and credit register are programmed once at
// init and each forward sets only what differs (stateful NoC API). Not with DPRINT or the watcher, whose traffic shares
// the command buffers, nor on intermesh or edge routers, whose exits pick a downstream at run time.
template <size_t VC>
constexpr bool forwards_are_stateful() {
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    return VC == 0 && popcount(live_downstreams<VC>()) == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode &&
           !is_intermesh_router && !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    return false;
#endif
}

// Set once at init from the runtime args.
struct DownstreamLink {
    uint32_t ring_base;        // the downstream sender channel's first slot
    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless forwards)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)
};

// Changes with every forward.
struct DownstreamCursor {
    uint32_t next_slot;   // the downstream ring's next slot
    uint32_t free_slots;  // slots known free there (a lower bound of the local credit register)
};

// The downstream links of one VC, filled at init.
template <size_t VC>
struct DownstreamLinks {
    std::array<DownstreamLink, DownstreamTraits<VC>::size> link;
};

// The downstream connections of one VC as the receiver loop uses them: the cursors, and a copy of the links set at
// init. Only forwards touch them, so they stay in memory (the receiver loop takes them by reference), not in registers;
// holding the links by value puts a forward's ring bounds at a fixed offset from its cursor, one load away.
template <size_t VC>
struct Downstreams {
    using T = DownstreamTraits<VC>;
    static constexpr uint32_t live = live_downstreams<VC>();

    std::array<DownstreamCursor, T::size> cursor;
    const DownstreamLinks<VC> links;

    explicit Downstreams(const DownstreamLinks<VC>& l) : links(l) {
        for_each_index<T::size>([&](auto i) { cursor[i] = {l.link[i].ring_base, 0}; });
    }

    // Every downstream in `to` (live names, one bit each) has a free slot. The known counts are tested first; only when
    // one the packet needs is used up are the live local credit registers re-read, as one group.
    FORCE_INLINE bool admit(uint32_t to) {
        if (!any_used_up<live>(to)) {
            return true;
        }
        uint32_t raw[T::size];  // indexed by entry
        std::atomic_signal_fence(std::memory_order_seq_cst);
        read_local_credits<live>(raw);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        set_free_slots<live>(raw);
        return !any_used_up<live>(to);
    }

    // One full-packet copy into downstream D (by name), tagged with `trid`, then its credits.
    template <size_t D>
    FORCE_INLINE void forward(uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        constexpr size_t I = downstream_entry(D);
        DownstreamCursor& c = cursor[I];
        const uint32_t dst = c.next_slot;
        if constexpr (forwards_are_stateful<VC>()) {
#if defined(ARCH_BLACKHOLE)
            // The forward's two commands go on the data and credit command buffers: their ready state is read as a
            // pair, so the two reads overlap (RESULTS §17) instead of each being a lone read inside the NoC API, and a
            // buffer not ready yet is waited for just before its command, as the API would. Only this ERISC issues on
            // them, so a buffer seen ready stays ready until its next command.
            const bool data_ready = noc_cmd_buf_ready(T::noc, T::data_cmd_buf);
            const bool sync_ready = noc_cmd_buf_ready(T::noc, T::sync_cmd_buf);
            if (!data_ready) {
                while (!noc_cmd_buf_ready(T::noc, T::data_cmd_buf));
            }
            ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(T::noc, T::data_cmd_buf, trid);
            ncrisc_noc_write_with_state<
                DM_DEDICATED_NOC,
                /*posted=*/true,
                /*update_counter=*/true,
                /*one_packet=*/false,
                /*skip_cmdbuf_chk=*/true>(T::noc, T::data_cmd_buf, src_addr, dst, size_bytes);
#else
            noc_async_write_one_packet_with_trid_with_state</*update_counter=*/false, /*posted=*/true>(
                src_addr, dst, size_bytes, trid, T::data_cmd_buf, T::noc);
#endif
            // On deadlock-avoiding routers main re-programs the credit address on every forward (the same address
            // as preset; main gives no reason), and so does this router: one command-buffer register write.
            if (!sync_ready) {
                while (!noc_cmd_buf_ready(T::noc, T::sync_cmd_buf));
            }
            noc_inline_dw_write_with_state<
                /*update_addr_lo=*/enable_deadlock_avoidance,
                /*update_counter=*/false,
                /*posted=*/true,
                false,
                false,
                InlineWriteDst::REG,
                /*skip_cmdbuf_chk=*/true>(
                0, get_stream_reg_write_addr<T::template remote_credit_stream<I>>(), T::sync_cmd_buf, T::noc);
        } else {
            noc_async_write_one_packet_with_trid</*update_counter=*/is_blackhole_arch, /*posted=*/true>(
                src_addr,
                (static_cast<uint64_t>(links.link[I].noc_coord) << NOC_ADDR_COORD_SHIFT) | dst,
                size_bytes,
                trid,
                T::data_cmd_buf,
                T::noc);
            noc_inline_dw_write<InlineWriteDst::REG>(
                links.link[I].credit_noc_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xf, T::noc);
        }
        c.next_slot = dst == links.link[I].ring_base + T::ring_bytes ? links.link[I].ring_base : dst + channel_buffer_size;
        c.free_slots--;
        increment_local_update_ptr_val<T::template local_credit_stream<I>>(-1);
    }

    // Forwards to every downstream in `to` (live names), lowest name first.
    FORCE_INLINE void forward_all(uint32_t to, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        forward_each<live>(to, src_addr, size_bytes, trid);
    }

private:
    // Per-packet helpers recurse over a slot mask rather than take for_each_bit closures (router_sender.hpp, "The loop
    // function takes no closures").

    // Whether a downstream in MASK (names) that `to` needs has no known free slot.
    template <uint32_t MASK>
    FORCE_INLINE bool any_used_up(uint32_t to) const {
        if constexpr (MASK != 0) {
            constexpr size_t D = lowest_bit(MASK);
            return ((((to >> D) & 1u) != 0) & (cursor[downstream_entry(D)].free_slots == 0)) |
                   any_used_up<without_lowest_bit(MASK)>(to);
        } else {
            return false;
        }
    }

    // The local credit registers of the downstreams in MASK (names), into raw[] (indexed by entry), lowest name first.
    template <uint32_t MASK>
    FORCE_INLINE static void read_local_credits(uint32_t* raw) {
        if constexpr (MASK != 0) {
            constexpr size_t I = downstream_entry(lowest_bit(MASK));
            raw[I] = read_stream_raw<T::template local_credit_stream<I>>();
            read_local_credits<without_lowest_bit(MASK)>(raw);
        }
    }

    template <uint32_t MASK>
    FORCE_INLINE void set_free_slots(const uint32_t* raw) {
        if constexpr (MASK != 0) {
            constexpr size_t I = downstream_entry(lowest_bit(MASK));
            cursor[I].free_slots = raw[I] & stream_count_mask;
            set_free_slots<without_lowest_bit(MASK)>(raw);
        }
    }

    template <uint32_t MASK>
    FORCE_INLINE void forward_each(uint32_t to, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        if constexpr (MASK != 0) {
            constexpr size_t D = lowest_bit(MASK);
            if (((to >> D) & 1u) != 0) {
                forward<D>(src_addr, size_bytes, trid);
            }
            forward_each<without_lowest_bit(MASK)>(to, src_addr, size_bytes, trid);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Init (out of line, once)
// ---------------------------------------------------------------------------------------------------------------------

// The downstream connections of one VC as the runtime args describe them: which entries (compact order) have a
// downstream (one bit each), and for those, densely in entry order, ring base, location-info and registration addresses
// on the downstream, this router's teardown semaphore for it, and the NoC coordinates (packed, one byte per dense entry).
struct DownstreamArgs {
    uint32_t mask;
    const uint32_t* ring_bases;
    const uint32_t* location_info_addrs;
    const uint32_t* registration_addrs;
    const size_t* teardown_sems;
    uint32_t noc_x_packed;
    uint32_t noc_y_packed;
};

FORCE_INLINE uint32_t dense_noc_x(const DownstreamArgs& a, uint32_t dense) { return (a.noc_x_packed >> (dense * 8)) & 0xFF; }
FORCE_INLINE uint32_t dense_noc_y(const DownstreamArgs& a, uint32_t dense) { return (a.noc_y_packed >> (dense * 8)) & 0xFF; }

// Before the Ethernet handshake: zero the teardown semaphores, set the local credit copies to the ring size, fill the
// cursors and links.
template <size_t VC>
__attribute__((noinline)) void init_downstreams(DownstreamLinks<VC>& links, const DownstreamArgs& a) {
    using T = DownstreamTraits<VC>;
    uint32_t dense = 0;
    for_each_index<T::size>([&](auto i) {
        constexpr size_t I = decltype(i)::value;
        links.link[I] = {0, 0, 0};
        if ((a.mask >> I) & 1u) {
            *reinterpret_cast<volatile uint32_t*>(a.teardown_sems[dense]) = 0;
            init_ptr_val<T::template local_credit_stream<I>>(T::num_slots);
            const uint32_t x = dense_noc_x(a, dense);
            const uint32_t y = dense_noc_y(a, dense);
            links.link[I].ring_base = a.ring_bases[dense];
            links.link[I].noc_coord = static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT);
            links.link[I].credit_noc_addr =
                get_noc_addr(x, y, get_stream_reg_write_addr(T::template remote_credit_stream<I>), T::noc);
            dense++;
        }
    });
}

// Opens dense connection `dense` of `a`: registers this router with the downstream as its producer, its credits to go
// to `credit_register` (this router's local copy of the downstream's free slots). One body for every connection.
__attribute__((noinline)) void open_downstream(const DownstreamArgs& a, uint32_t dense, uint32_t credit_register) {
    // Blackhole opens router-to-router connections with non-posted writes.
    constexpr bool posted = !is_blackhole_arch;
    const uint32_t x = dense_noc_x(a, dense);
    const uint32_t y = dense_noc_y(a, dense);
    tt::tt_fabric::connection::open_start<false, posted, tt::tt_fabric::worker_handshake_noc>(
        a.location_info_addrs[dense], credit_register, a.teardown_sems[dense], x, y);
    tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
        a.registration_addrs[dense], reinterpret_cast<volatile uint32_t*>(a.teardown_sems[dense]), x, y);
}

// After the handshake: open each connection.
template <size_t VC>
__attribute__((noinline)) void open_downstreams(const DownstreamArgs& a) {
    using T = DownstreamTraits<VC>;
    uint32_t dense = 0;
    for_each_index<T::size>([&](auto i) {
        constexpr size_t I = decltype(i)::value;
        if ((a.mask >> I) & 1u) {
            open_downstream(a, dense, get_stream_reg_write_addr(T::template local_credit_stream<I>));
            dense++;
        }
    });
}

// After the open: program the forwarding command buffers of a stateful VC with its one downstream.
template <size_t VC>
__attribute__((noinline)) void preset_downstream_cmd_bufs(const DownstreamArgs& a) {
    using T = DownstreamTraits<VC>;
    if constexpr (forwards_are_stateful<VC>()) {
        constexpr size_t I = downstream_entry(lowest_bit(live_downstreams<VC>()));
        if ((a.mask >> I) & 1u) {
            constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
            const uint32_t dense = popcount(a.mask & ((1u << I) - 1));
            const uint32_t x = dense_noc_x(a, dense);
            const uint32_t y = dense_noc_y(a, dense);
            noc_async_write_one_packet_with_trid_set_state<true>(get_noc_addr(x, y, 0, T::noc), T::data_cmd_buf, T::noc, noc_vc);
            noc_inline_dw_write_set_state<true, true>(
                get_noc_addr(x, y, get_stream_reg_write_addr(T::template remote_credit_stream<I>), T::noc),
                pack_value_for_inc_on_write_stream_reg_write(-1),
                0xF,
                T::sync_cmd_buf,
                T::noc,
                noc_vc);
        }
    }
}

}  // namespace fabric_router
