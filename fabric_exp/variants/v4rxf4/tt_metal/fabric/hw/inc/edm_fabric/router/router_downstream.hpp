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
// usually costs no register read (cached downstream credits, LEDGER E1-2).
//
// Forwards go through the NoC API. Where VC0 has a single live downstream (E/W-facing 2D routers inside a mesh, 1D),
// its forwarding command buffers stay programmed with that downstream's coordinate for the whole run (stateful API),
// and another VC whose forwards use the same buffers and go to the same downstream forwards on that preset too; every
// other forward programs the destination per write (stateless API).
//
// Names, entries and dense indices, one term each:
//   - name: how the receiver loop and routing name a downstream: its direction in 2D (so a packet's action bits are its
//     downstreams), 0 for 1D's one downstream;
//   - entry: its index in the host's compact order, which the tables below keep (downstream_entry maps a name to its
//     entry at compile time);
//   - dense index: its index among the entries the runtime args list (the entries that have a downstream).
//
// UDM (FabricTensixConfig::UDM). Packets for this chip go to the fabric's relay, a kernel on a Tensix core that does
// their NoC work on this chip (writes, atomics, reads), not to their NoC destinations. To the receiver the relay is one
// more downstream (Relay, V4_UDM.md).

#include "router_common.hpp"

namespace fabric_router {

// ---------------------------------------------------------------------------------------------------------------------
// Downstreams of a VC
// ---------------------------------------------------------------------------------------------------------------------

// Byte enables of a whole-word inline write (the credit register).
constexpr uint32_t INLINE_WRITE_ALL_BYTES = 0xF;

// The host's downstream args exist per VC for the VCs that forward to other routers: VC0 and VC1
// (VC<n>_DOWNSTREAM_EDM_SIZE, DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY, the runtime args' per-VC blocks).
constexpr std::array<size_t, 2> DOWNSTREAM_EDM_COUNT_BY_VC = {VC0_DOWNSTREAM_EDM_SIZE, VC1_DOWNSTREAM_EDM_SIZE};

// The downstream routers of VC, as the host sets them.
template <size_t VC>
struct DownstreamTraits {
    static_assert(VC < DOWNSTREAM_EDM_COUNT_BY_VC.size(), "the host lists downstream routers for VC0 and VC1");
    static constexpr size_t count = DOWNSTREAM_EDM_COUNT_BY_VC[VC];
    static constexpr uint32_t num_slots = DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[VC];
    // The last slot's offset from the ring's base (the ring wraps after it).
    static constexpr uint32_t last_slot_offset = (num_slots - 1) * channel_buffer_size;
    static constexpr uint8_t data_cmd_buf = receiver_channel_forwarding_data_cmd_buf_ids[VC];
    static constexpr uint8_t sync_cmd_buf = receiver_channel_forwarding_sync_cmd_buf_ids[VC];
    static constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;

    // This router's local copy of entry ENTRY's free-slots register, and the register on the downstream.
    template <size_t ENTRY>
    static constexpr uint32_t local_credit_stream = local_downstream_credit_stream(VC, ENTRY);
    template <size_t ENTRY>
    static constexpr uint32_t remote_credit_stream = downstream_sender_free_slots_stream(VC, ENTRY);

    static_assert(count > 0 && count <= max_downstreams);
    static_assert(num_slots <= stream_count_mask);
};

// The downstreams a router can forward to, one bit per name. In 2D, the directions with an entry (the host lists
// DownstreamTraits::count of them); an E/W-facing router inside a mesh never turns a packet onto N/S (packets travel Y
// first), so its N/S downstreams are not live. Intermesh routers (a landing decodes Y first) and routers on a mesh's
// edge (an exit may leave in any direction) can turn onto N/S. In 1D, downstream 0.
template <size_t VC>
constexpr uint32_t live_downstreams() {
    if constexpr (!is_2d) {
        return 0x1u;
    } else {
        constexpr bool east_west = my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST;
        constexpr bool turns = is_intermesh_router || is_intramesh_router_on_edge;
        uint32_t live = 0;
        for (size_t entry = 0; entry < DownstreamTraits<VC>::count; entry++) {
            const auto direction = downstream_direction(entry);
            if (!(east_west && !turns &&
                  (direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH))) {
                live |= 1u << direction;
            }
        }
        return live;
    }
}

// Forward command-buffer presets. VC0's forwards keep their command buffers programmed with the destination when VC0
// has exactly one live downstream (every 1D router; an E/W-facing 2D router inside a mesh): its coordinate and credit
// register are programmed once at init (preset_downstream_cmd_bufs) and each forward sets only what differs (stateful
// NoC API). Not with DPRINT or the watcher, whose traffic shares the command buffers, nor on intermesh or edge routers,
// whose exits pick a downstream at run time.
//
// Another VC's forwards may share VC0's command buffers (the host's RX_CH_<n>_FWD_DATA_ / FWD_SYNC_CMD_BUF_ID). Such a
// VC forwards on VC0's preset when it uses both of them and goes to the same one live downstream: the coordinate is the
// same, and each forward sets its own ring slot and credit register. A VC that shares a buffer with VC0 but cannot use
// the preset would overwrite it, so VC0 then keeps none: every forward is stateless.
constexpr bool command_buffers_can_keep_presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && !is_intermesh_router &&
    !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    false;
#endif

// The VCs whose forwards go to downstream routers here (DownstreamConnections).
constexpr bool vc_forwards_to_routers(size_t vc) { return vc == 0 || (vc == 1 && has_vc1); }

// Whether VC's forwards use VC0's command buffers: both of them, or at least one.
template <size_t VC>
constexpr bool uses_both_vc0_forward_cmd_bufs() {
    return DownstreamTraits<VC>::data_cmd_buf == DownstreamTraits<0>::data_cmd_buf &&
           DownstreamTraits<VC>::sync_cmd_buf == DownstreamTraits<0>::sync_cmd_buf;
}
template <size_t VC>
constexpr bool uses_a_vc0_forward_cmd_buf() {
    constexpr uint8_t data = DownstreamTraits<VC>::data_cmd_buf;
    constexpr uint8_t sync = DownstreamTraits<VC>::sync_cmd_buf;
    return data == DownstreamTraits<0>::data_cmd_buf || data == DownstreamTraits<0>::sync_cmd_buf ||
           sync == DownstreamTraits<0>::data_cmd_buf || sync == DownstreamTraits<0>::sync_cmd_buf;
}

// Whether another VC (VC1, where this router has it) forwards on VC0's preset, given that VC0 keeps one.
template <size_t VC>
constexpr bool rides_vc0_preset() {
    return VC != 0 && uses_both_vc0_forward_cmd_bufs<VC>() && live_downstreams<VC>() == live_downstreams<0>();
}
// Whether another VC would overwrite VC0's preset: it uses one of VC0's buffers without riding the preset.
template <size_t VC>
constexpr bool clobbers_vc0_preset() {
    if constexpr (VC == 0 || !vc_forwards_to_routers(VC)) {
        return false;
    } else {
        return uses_a_vc0_forward_cmd_buf<VC>() && !rides_vc0_preset<VC>();
    }
}
constexpr bool vc0_keeps_forward_preset() {
    return command_buffers_can_keep_presets && popcount(live_downstreams<0>()) == 1 && !clobbers_vc0_preset<1>();
}

// Whether VC's forwards are stateful (on VC0's preset).
template <size_t VC>
constexpr bool forwards_are_stateful() {
    if constexpr (!vc0_keeps_forward_preset()) {
        return false;
    } else if constexpr (VC == 0) {
        return true;
    } else {
        return rides_vc0_preset<VC>();
    }
}

// Whether a stateful forward sets its credit register: where another VC's forwards share the preset (their credit
// registers differ), and on deadlock-avoiding routers, where main re-programs it on every forward (the same address as
// preset; main gives no reason).
constexpr bool another_vc_rides_vc0_preset() {
    if constexpr (vc_forwards_to_routers(1)) {
        return forwards_are_stateful<1>();
    } else {
        return false;
    }
}
constexpr bool forwards_set_credit_register = enable_deadlock_avoidance || another_vc_rides_vc0_preset();

// Set once at init from the runtime args.
struct DownstreamLink {
    uint32_t ring_base;        // the downstream sender channel's first slot
    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless forwards)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)

    // The NoC address of `addr` on the downstream.
    FORCE_INLINE uint64_t noc_addr(uint32_t addr) const {
        return (static_cast<uint64_t>(noc_coord) << NOC_ADDR_COORD_SHIFT) | addr;
    }
};

// The slot after `slot` in a ring from `ring_base` whose last slot is at `last_slot_offset`.
FORCE_INLINE uint32_t next_ring_slot(uint32_t slot, uint32_t ring_base, uint32_t last_slot_offset) {
    return slot == ring_base + last_slot_offset ? ring_base : slot + channel_buffer_size;
}

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
        src_addr, link.noc_addr(dst), size_bytes, trid, DATA_CMD_BUF, NOC);
    noc_inline_dw_write<InlineWriteDst::REG>(
        link.credit_noc_addr, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, NOC);
}

// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole packet, and a free-slots
// register the relay increments as it frees slots, of which this router keeps a local copy. Its forwards are stateless,
// as main's, on VC0's forward command buffer (UDM routers have VC0 only).
struct Relay {
    static constexpr uint32_t num_slots = LOCAL_RELAY_NUM_BUFFERS;
    static constexpr uint32_t last_slot_offset = (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t local_credit_stream = tensix_relay_local_free_slots_stream_id;

    DownstreamLink link;  // set at init
    DownstreamCursor cursor;

    // The relay has a free slot. Its known count is tested first; its local credit register is re-read only when that
    // is used up.
    FORCE_INLINE bool admit() {
        if (cursor.free_slots == 0) {
            cursor.free_slots = read_stream_count<local_credit_stream>();
        }
        return cursor.free_slots != 0;
    }

    // The whole packet into the relay's next slot, tagged with `trid` (the slot's completion waits for it), then its
    // credits.
    FORCE_INLINE void forward(uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        using Traits = DownstreamTraits<0>;
        const uint32_t dst = cursor.next_slot;
        forward_stateless<Traits::noc, Traits::data_cmd_buf>(link, dst, src_addr, size_bytes, trid);
        cursor.next_slot = next_ring_slot(dst, link.ring_base, last_slot_offset);
        cursor.free_slots--;
        increment_local_update_ptr_val<local_credit_stream>(-1);
    }
};

// The downstream connections of one VC as the receiver loop uses them: the cursors, and the links set at init. Holding
// the links by value puts a forward's ring bounds at a fixed offset from its cursor, one load away.
template <size_t VC>
struct Downstreams {
    using Traits = DownstreamTraits<VC>;
    static constexpr uint32_t live = live_downstreams<VC>();

    std::array<DownstreamCursor, Traits::count> cursors;  // by entry
    std::array<DownstreamLink, Traits::count> links;      // by entry, set at init

    // Every downstream in `forward_to` (live names, one bit each) has a free slot. The known counts are tested first;
    // only when one the packet needs is used up are the live local credit registers re-read, as one group.
    FORCE_INLINE bool admit(uint32_t forward_to) {
        if (!any_without_free_slot<live>(forward_to)) {
            return true;
        }
        uint32_t raw[Traits::count];  // by entry
        begin_read_group();
        read_local_credits<live>(raw);
        end_read_group();
        set_free_slots<live>(raw);
        return !any_without_free_slot<live>(forward_to);
    }

    // One full-packet copy into downstream NAME, tagged with `trid`, then its credits.
    template <size_t NAME>
    FORCE_INLINE void forward(uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        constexpr size_t entry = downstream_entry(NAME);
        DownstreamCursor& downstream = cursors[entry];
        const uint32_t dst = downstream.next_slot;
        if constexpr (forwards_are_stateful<VC>()) {
#if defined(ARCH_BLACKHOLE)
            // The forward's two commands go on the data and credit command buffers: their ready state is read as a
            // pair, so the two reads overlap (RESULTS §17) instead of each being a lone read inside the NoC API, and a
            // buffer not ready yet is waited for just before its command, as the API would. Only this ERISC issues on
            // them, so a buffer seen ready stays ready until its next command.
            const bool data_ready = noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf);
            const bool sync_ready = noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf);
            if (!data_ready) {
                while (!noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf));
            }
            ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, trid);
            ncrisc_noc_write_with_state<
                DM_DEDICATED_NOC,
                /*posted=*/true,
                /*update_counter=*/true,
                /*one_packet=*/false,
                /*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, src_addr, dst, size_bytes);
#else
            noc_async_write_one_packet_with_trid_with_state</*update_counter=*/false, /*posted=*/true>(
                src_addr, dst, size_bytes, trid, Traits::data_cmd_buf, Traits::noc);
#endif
            // The credit address, where it is set on every forward (forwards_set_credit_register): one command-buffer
            // register write.
            if (!sync_ready) {
                while (!noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf));
            }
            noc_inline_dw_write_with_state<
                /*update_addr_lo=*/forwards_set_credit_register,
                /*update_counter=*/false,
                /*posted=*/true,
                /*update_addr_hi=*/false,
                /*update_val=*/false,
                InlineWriteDst::REG,
                /*skip_cmdbuf_chk=*/true>(
                0, get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(), Traits::sync_cmd_buf, Traits::noc);
        } else {
            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);
        }
        downstream.next_slot = next_ring_slot(dst, links[entry].ring_base, Traits::last_slot_offset);
        downstream.free_slots--;
        increment_local_update_ptr_val<Traits::template local_credit_stream<entry>>(-1);
    }

    // Forwards to every downstream in `forward_to` (live names), lowest name first.
    FORCE_INLINE void forward_all(uint32_t forward_to, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        forward_each<live>(forward_to, src_addr, size_bytes, trid);
    }

private:
    // Per-packet helpers recurse over a mask of names rather than take per-bit closures (router_sender.hpp, "The loop
    // function takes no closures").

    // Whether a downstream in NAMES that `forward_to` needs has no known free slot. Bitwise & and | on the tests: no
    // branch per downstream.
    template <uint32_t NAMES>
    FORCE_INLINE bool any_without_free_slot(uint32_t forward_to) const {
        if constexpr (NAMES != 0) {
            constexpr size_t name = lowest_bit(NAMES);
            return ((((forward_to >> name) & 1u) != 0) & (cursors[downstream_entry(name)].free_slots == 0)) |
                   any_without_free_slot<without_lowest_bit(NAMES)>(forward_to);
        } else {
            return false;
        }
    }

    // The local credit registers of the downstreams in NAMES, into raw[] (by entry), lowest name first.
    template <uint32_t NAMES>
    FORCE_INLINE static void read_local_credits(uint32_t* raw) {
        if constexpr (NAMES != 0) {
            constexpr size_t entry = downstream_entry(lowest_bit(NAMES));
            raw[entry] = read_stream_raw<Traits::template local_credit_stream<entry>>();
            read_local_credits<without_lowest_bit(NAMES)>(raw);
        }
    }

    template <uint32_t NAMES>
    FORCE_INLINE void set_free_slots(const uint32_t* raw) {
        if constexpr (NAMES != 0) {
            constexpr size_t entry = downstream_entry(lowest_bit(NAMES));
            cursors[entry].free_slots = raw[entry] & stream_count_mask;
            set_free_slots<without_lowest_bit(NAMES)>(raw);
        }
    }

    template <uint32_t NAMES>
    FORCE_INLINE void forward_each(uint32_t forward_to, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        if constexpr (NAMES != 0) {
            constexpr size_t name = lowest_bit(NAMES);
            if (((forward_to >> name) & 1u) != 0) {
                forward<name>(src_addr, size_bytes, trid);
            }
            forward_each<without_lowest_bit(NAMES)>(forward_to, src_addr, size_bytes, trid);
        }
    }
};

// The receiver's downstream connections: the downstream routers of each VC that forwards to other routers (VC0's;
// VC1's on multi-mesh routers) and UDM's relay, each where this router has it.
// Placement: only forwards touch them, so they stay in memory (the receiver loop takes them by reference), not in the
// registers every packet needs (RESULTS §98).
struct DownstreamConnections {
    Downstreams<0> vc0;
    [[no_unique_address]] Present<vc_forwards_to_routers(1), Downstreams<1>, 1> vc1;
    [[no_unique_address]] Present<udm_mode, Relay, 2> relay;

    // VC's downstream routers.
    template <size_t VC>
    FORCE_INLINE auto& of() {
        static_assert(vc_forwards_to_routers(VC));
        if constexpr (VC == 0) {
            return vc0;
        } else {
            return vc1;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Init (out of line, once)
// ---------------------------------------------------------------------------------------------------------------------

// The downstream connections of one VC as the runtime args describe them: which entries have a downstream (one bit
// each), and for those, by dense index, ring base, location-info and registration addresses on the downstream, this
// router's teardown semaphore for it, and the NoC coordinates (packed, one byte per dense index).
struct DownstreamArgs {
    uint32_t entries;
    const uint32_t* ring_bases;
    const uint32_t* location_info_addrs;
    const uint32_t* registration_addrs;
    const size_t* teardown_sems;
    uint32_t noc_x_packed;
    uint32_t noc_y_packed;
};

FORCE_INLINE uint32_t dense_noc_x(const DownstreamArgs& args, uint32_t dense) {
    return (args.noc_x_packed >> (dense * 8)) & 0xFF;
}
FORCE_INLINE uint32_t dense_noc_y(const DownstreamArgs& args, uint32_t dense) {
    return (args.noc_y_packed >> (dense * 8)) & 0xFF;
}

// Before the Ethernet handshake: zero the teardown semaphores, set the local credit copies to the ring size, fill the
// links and the cursors.
template <size_t VC>
__attribute__((noinline)) void init_downstreams(Downstreams<VC>& downstreams, const DownstreamArgs& args) {
    using Traits = DownstreamTraits<VC>;
    auto& links = downstreams.links;
    uint32_t dense = 0;
    for_each_index<Traits::count>([&](auto entry_constant) {
        constexpr size_t entry = decltype(entry_constant)::value;
        links[entry] = {0, 0, 0};
        if ((args.entries >> entry) & 1u) {
            *reinterpret_cast<volatile uint32_t*>(args.teardown_sems[dense]) = 0;
            init_ptr_val<Traits::template local_credit_stream<entry>>(Traits::num_slots);
            const uint32_t x = dense_noc_x(args, dense);
            const uint32_t y = dense_noc_y(args, dense);
            links[entry].ring_base = args.ring_bases[dense];
            links[entry].noc_coord = static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT);
            links[entry].credit_noc_addr =
                get_noc_addr(x, y, get_stream_reg_write_addr(Traits::template remote_credit_stream<entry>), Traits::noc);
            dense++;
        }
        downstreams.cursors[entry] = {links[entry].ring_base, 0};
    });
}

// Opens the connection at dense index `dense` of `args`: registers this router with the downstream as its producer,
// its credits to go to `credit_register` (this router's local copy of the downstream's free slots). One body for every
// connection.
__attribute__((noinline)) void open_downstream(const DownstreamArgs& args, uint32_t dense, uint32_t credit_register) {
    // Blackhole opens router-to-router connections with non-posted writes.
    constexpr bool posted = !is_blackhole_arch;
    const uint32_t x = dense_noc_x(args, dense);
    const uint32_t y = dense_noc_y(args, dense);
    tt::tt_fabric::connection::open_start</*SEND_CREDIT_ADDR=*/false, posted, tt::tt_fabric::worker_handshake_noc>(
        args.location_info_addrs[dense], credit_register, args.teardown_sems[dense], x, y);
    tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
        args.registration_addrs[dense], reinterpret_cast<volatile uint32_t*>(args.teardown_sems[dense]), x, y);
}

// After the handshake: open each connection.
template <size_t VC>
__attribute__((noinline)) void open_downstreams(const DownstreamArgs& args) {
    using Traits = DownstreamTraits<VC>;
    uint32_t dense = 0;
    for_each_index<Traits::count>([&](auto entry_constant) {
        constexpr size_t entry = decltype(entry_constant)::value;
        if ((args.entries >> entry) & 1u) {
            open_downstream(args, dense, get_stream_reg_write_addr(Traits::template local_credit_stream<entry>));
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

// UDM, before the Ethernet handshake: this router's copy of the relay's free slots set to its ring size, the link and
// the cursor filled. Nothing on other routers (the types are then absent, NoRelayArgs).
template <typename RelayT, typename Args>
__attribute__((noinline)) void init_relay(RelayT& relay, const Args& args) {
    if constexpr (udm_mode) {
        ASSERT(args.present);
        init_ptr_val<Relay::local_credit_stream>(Relay::num_slots);
        relay.link.ring_base = args.ring_base;
        relay.link.noc_coord = static_cast<uint32_t>(get_noc_addr(args.noc_x, args.noc_y, 0) >> NOC_ADDR_COORD_SHIFT);
        relay.link.credit_noc_addr = get_noc_addr(
            args.noc_x, args.noc_y, get_stream_reg_write_addr(args.free_slots_stream), tt::tt_fabric::edm_to_downstream_noc);
        relay.cursor = {relay.link.ring_base, 0};
    }
}

// UDM, after the downstream routers: open the relay connection (persistent; never closed), as a router's is opened.
template <typename Args>
__attribute__((noinline)) void open_relay(const Args& args) {
    if constexpr (udm_mode) {
        constexpr bool posted = !is_blackhole_arch;
        tt::tt_fabric::connection::open_start</*SEND_CREDIT_ADDR=*/false, posted, tt::tt_fabric::worker_handshake_noc>(
            args.location_info, get_stream_reg_write_addr(Relay::local_credit_stream), args.teardown_word,
            args.noc_x, args.noc_y);
        tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
            args.registration, reinterpret_cast<volatile uint32_t*>(args.teardown_word), args.noc_x, args.noc_y);
    }
}

// After the open: program VC0's forwarding command buffers with its one downstream, where it keeps them programmed
// (vc0_keeps_forward_preset; VCs that forward on the preset set their own ring slot and credit register).
template <size_t VC>
__attribute__((noinline)) void preset_downstream_cmd_bufs(const DownstreamArgs& args) {
    using Traits = DownstreamTraits<VC>;
    if constexpr (forwards_are_stateful<VC>()) {
        constexpr size_t entry = downstream_entry(lowest_bit(live_downstreams<VC>()));
        if ((args.entries >> entry) & 1u) {
            constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
            const uint32_t dense = popcount(args.entries & ((1u << entry) - 1));  // entries below it with a downstream
            const uint32_t x = dense_noc_x(args, dense);
            const uint32_t y = dense_noc_y(args, dense);
            noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(
                get_noc_addr(x, y, 0, Traits::noc), Traits::data_cmd_buf, Traits::noc, noc_vc);
            noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
                get_noc_addr(x, y, get_stream_reg_write_addr(Traits::template remote_credit_stream<entry>), Traits::noc),
                pack_value_for_inc_on_write_stream_reg_write(-1),
                INLINE_WRITE_ALL_BYTES,
                Traits::sync_cmd_buf,
                Traits::noc,
                noc_vc);
        }
    }
}

}  // namespace fabric_router
