// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Downstream connections: the routers on this chip that a receiver channel forwards packets to (one per other
// direction in 2D, one in 1D). For each, the receiver keeps
//   - its progress: the next slot in the downstream sender channel and the number of slots known free there;
//   - the link: the channel's first slot and the downstream's NoC coordinate and credit register (from runtime args,
//     set at init).
//
// Credits. The downstream's sender channel has a free-slots stream register; this router keeps a local copy of it,
// which the downstream increments as it frees slots. A forward decrements both (the downstream's is its doorbell), and
// the progress's known-free count is a lower bound of the local copy, re-read only when it reaches zero. So admission
// usually costs no register read (cached downstream credits, LEDGER E1-2).
//
// Forwards go through the NoC API, on command buffers partially preset at init: the fields no forward changes stay
// programmed, and each forward writes only its own (forwards_use_presets).
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

// Forward command-buffer presets (partial). Every forward to a router is two commands on its VC's forwarding command
// buffers (the host's RX_CH_<n>_FWD_DATA_ / FWD_SYNC_CMD_BUF_ID): the packet on the data buffer, then -1 on the
// destination's free-slots register (its doorbell) on the credit buffer. The fields no forward changes are preset at
// init (preset_downstream_cmd_bufs): the data write's control (posted, static VC) and high address word, the credit's
// control, value (-1), byte enables and high address word. Each forward writes only its own: the data write's
// transaction ID, source, destination node and address, and length; the credit's register and node. One shape serves
// one downstream or several, a downstream picked at run time (intermesh, edge) included, and VCs whose forwards share
// buffers share the preset (their invariant fields are the same).
//
// Not with DPRINT or the watcher, whose traffic shares the command buffers, nor on a one-ERISC router
// (command_buffer_presets), nor on a UDM router, whose relay forwards are stateless on the same data buffer: there
// every forward programs its whole commands (forward_stateless).

// The VCs whose forwards go to downstream routers here (DownstreamConnections).
constexpr bool vc_forwards_to_routers(size_t vc) { return vc == 0 || (vc == 1 && has_vc1); }

// A buffer is preset for one kind of command: VC's data and credit buffers differ, neither is VC0's other one, and
// neither is the local-write buffer or the atomics buffer (write_at_cmd_buf), whose commands program those fields.
template <size_t VC>
constexpr bool forward_cmd_bufs_consistent() {
    if constexpr (!vc_forwards_to_routers(VC)) {
        return true;
    } else {
        using T = DownstreamTraits<VC>;
        constexpr uint8_t local = tt::tt_fabric::local_chip_data_cmd_buf;
        return T::data_cmd_buf != T::sync_cmd_buf && T::data_cmd_buf != DownstreamTraits<0>::sync_cmd_buf &&
               T::sync_cmd_buf != DownstreamTraits<0>::data_cmd_buf && T::data_cmd_buf != local &&
               T::sync_cmd_buf != local && T::data_cmd_buf != write_at_cmd_buf && T::sync_cmd_buf != write_at_cmd_buf;
    }
}

constexpr bool forwards_use_presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && forward_cmd_bufs_consistent<0>() &&
    forward_cmd_bufs_consistent<1>();
#else
    false;
#endif

// Set once at init from the runtime args.
struct DownstreamLink {
    uint32_t first_slot_addr;        // the downstream sender channel's first slot
    uint32_t noc_coord;        // the downstream's NoC coordinate (every forward but a full preset's)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)

    // The NoC address of `addr` on the downstream.
    FORCE_INLINE uint64_t noc_addr(uint32_t addr) const {
        return (static_cast<uint64_t>(noc_coord) << NOC_ADDR_COORD_SHIFT) | addr;
    }
};

// The slot after `slot` in a channel whose slots start at `first_slot_addr` and end at `last_slot_offset` from it.
FORCE_INLINE uint32_t slot_after(uint32_t slot, uint32_t first_slot_addr, uint32_t last_slot_offset) {
    return slot == first_slot_addr + last_slot_offset ? first_slot_addr : slot + channel_buffer_size;
}

// A downstream sender channel's progress: changes with every forward.
struct DownstreamSenderChannelProgress {
    uint32_t free_slots;      // slots known free there (a lower bound of the local credit register)
    uint32_t next_slot_addr;  // its next slot
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

// A forward on the partially preset buffers (forwards_use_presets): the packet into `dst` on the node at `link`, tagged
// with `trid`, then -1 on that node's free-slots register `credit_register`. Each command polls its own buffer just
// before it (PROBE nsbs, V4_NS_PRESETS §8.3: serial polls; no ready state is read ahead or held).
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
FORCE_INLINE void forward_partially_preset(
    const DownstreamLink& link,
    uint32_t credit_register,
    uint32_t dst,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    // Read once: the register writes below store through uint32_t pointers, which may alias the field.
    const uint32_t node = link.noc_coord;
    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/false>(NOC, DATA_CMD_BUF, trid);
    noc_write_to_node_with_state<
        DM_DEDICATED_NOC,
        DATA_CMD_BUF,
        CQ_NOC_SNDL,
        CQ_NOC_SEND,
        CQ_NOC_wait,
        /*update_counter=*/is_blackhole_arch,
        /*posted=*/true>(NOC, src_addr, node, dst, size_bytes);
    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/false>(
        node, credit_register, CREDIT_CMD_BUF, NOC);
}

// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole packet, and a free-slots
// register the relay increments as it frees slots, of which this router keeps a local copy. Its forwards are stateless,
// as main's, on VC0's forward command buffer (UDM routers have VC0 only).
struct Relay {
    static constexpr uint32_t num_slots = LOCAL_RELAY_NUM_BUFFERS;
    static constexpr uint32_t last_slot_offset = (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t local_credit_stream = tensix_relay_local_free_slots_stream_id;

    DownstreamLink link;  // set at init
    DownstreamSenderChannelProgress progress;

    // The relay has a free slot. Its known count is tested first; its local credit register is re-read only when that
    // is used up.
    FORCE_INLINE bool admit() {
        if (progress.free_slots == 0) {
            progress.free_slots = read_stream_count<local_credit_stream>();
        }
        return progress.free_slots != 0;
    }

    // The whole packet into the relay's next slot, tagged with `trid` (the slot's completion waits for it), then its
    // credits.
    FORCE_INLINE void forward(uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        using Traits = DownstreamTraits<0>;
        const uint32_t dst = progress.next_slot_addr;
        forward_stateless<Traits::noc, Traits::data_cmd_buf>(link, dst, src_addr, size_bytes, trid);
        progress.next_slot_addr = slot_after(dst, link.first_slot_addr, last_slot_offset);
        progress.free_slots--;
        increment_local_update_ptr_val<local_credit_stream>(-1);
    }
};

// The downstream connections of one VC as the receiver loop uses them: each downstream sender channel's progress, and
// the links set at init. Holding the links by value puts a forward's slot bounds at a fixed offset from its progress,
// one load away.
template <size_t VC>
struct Downstreams {
    using Traits = DownstreamTraits<VC>;
    static constexpr uint32_t live = live_downstreams<VC>();

    std::array<DownstreamSenderChannelProgress, Traits::count> progress;  // by entry
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
        DownstreamSenderChannelProgress& downstream = progress[entry];
        const uint32_t dst = downstream.next_slot_addr;
        if constexpr (forwards_use_presets) {
            forward_partially_preset<Traits::noc, Traits::data_cmd_buf, Traits::sync_cmd_buf>(
                links[entry],
                get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(),
                dst,
                src_addr,
                size_bytes,
                trid);
        } else {
            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);
        }
        downstream.next_slot_addr = slot_after(dst, links[entry].first_slot_addr, Traits::last_slot_offset);
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
            return ((((forward_to >> name) & 1u) != 0) & (progress[downstream_entry(name)].free_slots == 0)) |
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
            progress[entry].free_slots = raw[entry] & stream_count_mask;
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
// each), and for those, by dense index, first slot address, location-info and registration addresses on the
// downstream, this router's teardown semaphore for it, and the NoC coordinates (packed, one byte per dense index).
struct DownstreamArgs {
    uint32_t entries;
    const uint32_t* first_slot_addrs;
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
// links and the progress.
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
            links[entry].first_slot_addr = args.first_slot_addrs[dense];
            links[entry].noc_coord = static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT);
            links[entry].credit_noc_addr =
                get_noc_addr(x, y, get_stream_reg_write_addr(Traits::template remote_credit_stream<entry>), Traits::noc);
            dense++;
        }
        downstreams.progress[entry] = {/*free_slots=*/0, /*next_slot_addr=*/links[entry].first_slot_addr};
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
    uint32_t first_slot_addr;          // the relay's first slot
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
// its progress set. Nothing on other routers (the types are then absent, NoRelayArgs).
template <typename RelayT, typename Args>
__attribute__((noinline)) void init_relay(RelayT& relay, const Args& args) {
    if constexpr (udm_mode) {
        ASSERT(args.present);
        init_ptr_val<Relay::local_credit_stream>(Relay::num_slots);
        relay.link.first_slot_addr = args.first_slot_addr;
        relay.link.noc_coord = static_cast<uint32_t>(get_noc_addr(args.noc_x, args.noc_y, 0) >> NOC_ADDR_COORD_SHIFT);
        relay.link.credit_noc_addr = get_noc_addr(
            args.noc_x, args.noc_y, get_stream_reg_write_addr(args.free_slots_stream), tt::tt_fabric::edm_to_downstream_noc);
        relay.progress = {/*free_slots=*/0, /*next_slot_addr=*/relay.link.first_slot_addr};
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

// After the opens: preset the forwarding command buffers' invariant fields (forwards_use_presets), VC0's and, where its
// buffers differ, VC1's. The destination node in the presets is a placeholder: every forward writes its own.
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
__attribute__((noinline)) void preset_forward_cmd_bufs() {
    constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
    noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(0, DATA_CMD_BUF, NOC, noc_vc);
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        0, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, CREDIT_CMD_BUF, NOC, noc_vc);
}
// VC's buffers, where it forwards and they are not VC0's (already preset).
template <size_t VC>
FORCE_INLINE void preset_vc_forward_cmd_bufs() {
    if constexpr (vc_forwards_to_routers(VC)) {
        using T = DownstreamTraits<VC>;
        if constexpr (
            VC == 0 || T::data_cmd_buf != DownstreamTraits<0>::data_cmd_buf ||
            T::sync_cmd_buf != DownstreamTraits<0>::sync_cmd_buf) {
            preset_forward_cmd_bufs<T::noc, T::data_cmd_buf, T::sync_cmd_buf>();
        }
    }
}
template <size_t VC>
__attribute__((noinline)) void preset_downstream_cmd_bufs(const DownstreamArgs&) {
    static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
    if constexpr (forwards_use_presets) {
        preset_vc_forward_cmd_bufs<0>();
        preset_vc_forward_cmd_bufs<1>();
    }
}

}  // namespace fabric_router
