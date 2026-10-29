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
//
// UDM (FabricTensixConfig::UDM). Packets for this chip go to the fabric's relay, a kernel on a Tensix core that does
// their NoC work on this chip (writes, atomics, reads), not to their NoC destinations. To the receiver the relay is one
// more downstream of VC0 (V4_UDM.md): a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole
// packet, and a free-slots register the relay increments as it frees slots, of which this router keeps a local copy
// (tensix_relay_local_free_slots_stream_id). Its forwards are stateless, as main's.

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
    return VC == 0 && popcount(live_downstreams<VC>()) == 1 && command_buffer_presets &&
           !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && !is_intermesh_router && !is_intermesh_router_on_edge &&
           !is_intramesh_router_on_edge;
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

// A stateless forward's two commands: the packet into `dst` on the node at `link`, tagged with `trid`, then -1 on the
// node's free-slots register (its doorbell).
template <uint8_t NOC, uint8_t DATA_CMD_BUF>
FORCE_INLINE void forward_stateless(
    const DownstreamLink& link, uint32_t dst, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    noc_async_write_one_packet_with_trid</*update_counter=*/is_blackhole_arch, /*posted=*/true>(
        src_addr, (static_cast<uint64_t>(link.noc_coord) << NOC_ADDR_COORD_SHIFT) | dst, size_bytes, trid, DATA_CMD_BUF, NOC);
    noc_inline_dw_write<InlineWriteDst::REG>(link.credit_noc_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xf, NOC);
}

// UDM's relay: its ring and this router's copy of its free-slots register.
struct RelayTraits {
    static constexpr uint32_t num_slots = LOCAL_RELAY_NUM_BUFFERS;
    static constexpr uint32_t ring_bytes = (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t local_credit_stream = tensix_relay_local_free_slots_stream_id;
};
struct NoRelay {};  // the relay's members on every other router

// The relay's members of VC0's downstream state: present only in UDM.
template <size_t VC, typename T>
using IfRelay = std::conditional_t<udm_mode && VC == 0, T, NoRelay>;

// The downstream links of one VC, filled at init.
template <size_t VC>
struct DownstreamLinks {
    std::array<DownstreamLink, DownstreamTraits<VC>::size> link;
    [[no_unique_address]] IfRelay<VC, DownstreamLink> relay;
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
    [[no_unique_address]] IfRelay<VC, DownstreamCursor> relay_cursor;  // UDM: the relay's (its link is links.relay)

    explicit Downstreams(const DownstreamLinks<VC>& l) : links(l) {
        for_each_index<T::size>([&](auto i) { cursor[i] = {l.link[i].ring_base, 0}; });
        if constexpr (udm_mode && VC == 0) {
            relay_cursor = {l.relay.ring_base, 0};
        }
    }

    // UDM: the relay has a free slot. Its known count is tested first; its local credit register is re-read only when
    // that is used up.
    FORCE_INLINE bool admit_relay() {
        if (relay_cursor.free_slots == 0) {
            relay_cursor.free_slots = read_stream_raw<RelayTraits::local_credit_stream>() & stream_count_mask;
        }
        return relay_cursor.free_slots != 0;
    }

    // UDM: the whole packet into the relay's next slot, tagged with `trid` (the slot's completion waits for it), then
    // its credits.
    FORCE_INLINE void forward_to_relay(uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        const uint32_t dst = relay_cursor.next_slot;
        forward_stateless<T::noc, T::data_cmd_buf>(links.relay, dst, src_addr, size_bytes, trid);
        relay_cursor.next_slot =
            dst == links.relay.ring_base + RelayTraits::ring_bytes ? links.relay.ring_base : dst + channel_buffer_size;
        relay_cursor.free_slots--;
        increment_local_update_ptr_val<RelayTraits::local_credit_stream>(-1);
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
            forward_stateless<T::noc, T::data_cmd_buf>(links.link[I], dst, src_addr, size_bytes, trid);
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

// UDM: the relay connection as the runtime args describe it (erisc_datamover_builder.cpp, pack_adaptor_to_relay_rt_args).
struct RelayArgs {
    uint32_t present;            // the host connected a relay (every UDM router but a dispatch link's, which has no UDM)
    uint32_t ring_base;          // the relay's first slot
    uint32_t noc_x;
    uint32_t noc_y;
    uint32_t registration;       // the relay's connection handshake word
    uint32_t location_info;      // the relay's location-info block for this connection
    uint32_t free_slots_stream;  // the relay's free-slots register
    // A word on this router the host reserves for the relay connection (its buffer index; unused by a persistent
    // connection). It is the connection's teardown word: the open writes its address to the relay and zeroes it, and
    // the relay would ack a close there, which never comes (main passes no word, so its open zeroes L1 address 0).
    uint32_t teardown_word;
};
struct NoRelayArgs {};

// UDM, before the Ethernet handshake: this router's copy of the relay's free slots set to its ring size, the link
// filled. Nothing on other routers (the types are then NoRelay, NoRelayArgs).
template <typename Link, typename Args>
__attribute__((noinline)) void init_relay(Link& link, const Args& a) {
    if constexpr (udm_mode) {
        ASSERT(a.present);
        init_ptr_val<RelayTraits::local_credit_stream>(RelayTraits::num_slots);
        link.ring_base = a.ring_base;
        link.noc_coord = static_cast<uint32_t>(get_noc_addr(a.noc_x, a.noc_y, 0) >> NOC_ADDR_COORD_SHIFT);
        link.credit_noc_addr = get_noc_addr(
            a.noc_x, a.noc_y, get_stream_reg_write_addr(a.free_slots_stream), tt::tt_fabric::edm_to_downstream_noc);
    }
}

// UDM, after the downstream routers: open the relay connection (persistent; never closed), as a router's is opened.
template <typename Args>
__attribute__((noinline)) void open_relay(const Args& a) {
    if constexpr (udm_mode) {
        constexpr bool posted = !is_blackhole_arch;
        tt::tt_fabric::connection::open_start<false, posted, tt::tt_fabric::worker_handshake_noc>(
            a.location_info, get_stream_reg_write_addr(RelayTraits::local_credit_stream), a.teardown_word, a.noc_x,
            a.noc_y);
        tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
            a.registration, reinterpret_cast<volatile uint32_t*>(a.teardown_word), a.noc_x, a.noc_y);
    }
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
