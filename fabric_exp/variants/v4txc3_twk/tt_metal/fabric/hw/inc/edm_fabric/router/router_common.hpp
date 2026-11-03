// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compile-time facts both router programs use: stream register ids, the downstream direction maps, channel slot
// layouts and small register helpers. Everything here is derived from the compile-time args
// (fabric_erisc_router_ct_args.hpp); nothing here holds state.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace fabric_router {

using tt::tt_fabric::eth_chan_directions;

// ---------------------------------------------------------------------------------------------------------------------
// Architecture and small constants
// ---------------------------------------------------------------------------------------------------------------------

constexpr bool is_blackhole_arch =
#if defined(ARCH_BLACKHOLE)
    true;
#else
    false;
#endif

constexpr bool is_2d = is_2d_fabric != 0;

// VC1 (multi-mesh fabrics): VC1 sender channels, a VC1 receiver channel, and VC1's downstream connections in the runtime
// args.
constexpr bool has_vc1 =
#if defined(FABRIC_2D_VC1_ACTIVE)
    true;
#else
    false;
#endif

// Tensix mux mode (FabricTensixConfig::MUX; V4_TENSIX_MUX.md): the fabric's mux of each direction feeds that router's
// worker channel with everything for its link and takes the forwards of the other directions' routers. The host's flag
// is also set under UDM, which is its own mode.
constexpr bool tensix_mux_mode = fabric_tensix_extension_mux_mode && !udm_mode;

// The count in a stream credit register (packets received, free slots). A count never exceeds its channel's slot
// count, and every channel static_asserts its slot count against this mask, so the low 11 bits hold it.
constexpr uint32_t stream_count_mask = 0x7FFu;
static_assert(stream_count_mask <= ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));

// The Ethernet firmware's heartbeat word. While the router runs instead of the base firmware, each loop keeps it moving
// every 64 rounds, with UMD's fabric signature (0xDCBA) in the upper half.
#if defined(ARCH_BLACKHOLE)
constexpr uint32_t heartbeat_addr = MEM_SYSENG_ETH_HEARTBEAT;
#else
constexpr uint32_t heartbeat_addr = 0x1F80;  // Wormhole's (main's router; the firmware headers do not name it)
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------------------------------

// The free-slots stream register of each sender channel, by fabric position (the table is fabric-flat: a router's
// compact channel i sits at fabric_position_for_compact_sender(i)).
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_free_slots_streams = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id,
    sender_channel_5_free_slots_stream_id,
    sender_channel_6_free_slots_stream_id,
    sender_channel_7_free_slots_stream_id,
    sender_channel_8_free_slots_stream_id,
    sender_channel_9_free_slots_stream_id};

// The free-slots stream register of this router's compact sender channel CH.
template <size_t CH>
constexpr uint32_t sender_free_slots_stream = sender_free_slots_streams[fabric_position_for_compact_sender(CH)];

// Initialize a stream register that this router uses; k_unused_stream_id marks one it does not (a VC whose credits
// travel through L1 counters), and is out of register range.
template <uint32_t STREAM_ID>
FORCE_INLINE void init_stream_if_used(uint32_t value) {
    if constexpr (STREAM_ID != tt::tt_fabric::k_unused_stream_id) {
        init_ptr_val<STREAM_ID>(value);
    }
}

// The count held in stream register STREAM_ID (masked).
template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t read_stream_count() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) & stream_count_mask;
}

// The raw value of stream register STREAM_ID (the caller masks).
template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t read_stream_raw() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

// ---------------------------------------------------------------------------------------------------------------------
// Downstream routers (2D): the other directions of this chip, in compact order
// ---------------------------------------------------------------------------------------------------------------------
//
// Each sender channel's connection info (its worker's location, the read counter a disconnected worker's credits go
// to), in L1 at a compile-time address.
constexpr std::array<size_t, MAX_NUM_SENDER_CHANNELS> sender_connection_info_addrs = {
    local_sender_channel_0_connection_info_addr, local_sender_channel_1_connection_info_addr,
    local_sender_channel_2_connection_info_addr, local_sender_channel_3_connection_info_addr,
    local_sender_channel_4_connection_info_addr, local_sender_channel_5_connection_info_addr,
    local_sender_channel_6_connection_info_addr, local_sender_channel_7_connection_info_addr,
    local_sender_channel_8_connection_info_addr, local_sender_channel_9_connection_info_addr};

// Each sender channel's connection semaphore (a worker channel's connection state, read every pass), in L1 at a
// compile-time address: reading it needs no pointer.
constexpr std::array<size_t, MAX_NUM_SENDER_CHANNELS> sender_connection_semaphore_addrs = {
    local_sender_channel_0_connection_semaphore_addr, local_sender_channel_1_connection_semaphore_addr,
    local_sender_channel_2_connection_semaphore_addr, local_sender_channel_3_connection_semaphore_addr,
    local_sender_channel_4_connection_semaphore_addr, local_sender_channel_5_connection_semaphore_addr,
    local_sender_channel_6_connection_semaphore_addr, local_sender_channel_7_connection_semaphore_addr,
    local_sender_channel_8_connection_semaphore_addr, local_sender_channel_9_connection_semaphore_addr};

// A router never forwards back out its own facing, so its downstream routers are the other four directions, packed
// densely in enum order (E, W, N, S, Z) with its own direction removed (the codec's compact dispatch order): the
// compact index of a downstream direction.
constexpr auto downstream_directions = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
constexpr uint32_t max_downstreams = downstream_directions.size();

constexpr eth_chan_directions downstream_direction(size_t compact) { return downstream_directions[compact]; }

// The compact index of downstream direction `dir` (dir != my_direction).
constexpr size_t downstream_compact_index(eth_chan_directions dir) {
    return static_cast<size_t>(dir) < static_cast<size_t>(my_direction) ? static_cast<size_t>(dir)
                                                                         : static_cast<size_t>(dir) - 1;
}

// Routing names a downstream by its direction (2D: action bit d is direction d); 1D's one downstream is named 0. The
// downstream tables keep the host's compact order, and downstream_entry maps a name to its entry there.
constexpr size_t downstream_entry(size_t name) {
    return is_2d ? downstream_compact_index(static_cast<eth_chan_directions>(name)) : 0;
}

// The free-slots stream register, on the downstream router, of the sender channel this router feeds (the downstream's
// channel for traffic arriving from this router's direction).
constexpr uint32_t downstream_sender_free_slots_stream(size_t vc, size_t compact) {
    const size_t base = vc == 0 ? 0 : VC1_FABRIC_POSITION_START - 1;
    if constexpr (is_2d) {
        const bool below = my_direction > static_cast<uint32_t>(downstream_direction(compact));
        return sender_free_slots_streams[base + (below ? my_direction : my_direction + 1)];
    } else {
        return sender_channel_1_free_slots_stream_id;
    }
}

// This router's local copy of a downstream's free slots (the downstream credits it as it sends), per VC.
constexpr std::array<uint32_t, max_downstreams> vc0_downstream_credit_streams = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    vc_0_free_slots_from_downstream_edge_4_stream_id};
constexpr std::array<uint32_t, max_downstreams> vc1_downstream_credit_streams = {
    vc_1_free_slots_from_downstream_edge_1_stream_id,
    vc_1_free_slots_from_downstream_edge_2_stream_id,
    vc_1_free_slots_from_downstream_edge_3_stream_id,
    vc_1_free_slots_from_downstream_edge_4_stream_id};

constexpr uint32_t local_downstream_credit_stream(size_t vc, size_t compact) {
    if constexpr (is_2d) {
        return vc == 0 ? vc0_downstream_credit_streams[compact] : vc1_downstream_credit_streams[compact];
    } else {
        return vc0_downstream_credit_streams[0];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Ethernet TX queues
// ---------------------------------------------------------------------------------------------------------------------

// A TX queue's status, read as eth_txq_is_busy reads it (a CMD read first, BH-55), so it can be read in a group with
// other registers; txq_busy tells from it whether a command is still going.
FORCE_INLINE uint32_t read_txq_status(uint8_t q) {
    (void)eth_txq_reg_read(q, ETH_TXQ_CMD);
    return eth_txq_reg_read(q, ETH_TXQ_STATUS);
}
FORCE_INLINE bool txq_busy(uint32_t status) { return ((status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) != 0; }

// ---------------------------------------------------------------------------------------------------------------------
// Compile-time loops
// ---------------------------------------------------------------------------------------------------------------------

// Calls f(std::integral_constant<size_t, I>) for each I in [0, N).
template <size_t N, typename F>
FORCE_INLINE void for_each_index(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) { (f(std::integral_constant<size_t, Is>{}), ...); }(
        std::make_index_sequence<N>{});
}

constexpr uint32_t popcount(uint32_t mask) { return static_cast<uint32_t>(__builtin_popcount(mask)); }

// For plain recursion over a channel mask: the lowest channel in MASK, and MASK without it. The per-pass code recurses
// over its masks instead of iterating with a closure (router_sender.hpp, "The loop function takes no closures").
constexpr size_t lowest_bit(uint32_t mask) { return static_cast<size_t>(__builtin_ctz(mask)); }
constexpr uint32_t without_lowest_bit(uint32_t mask) { return mask & (mask - 1); }

// ---------------------------------------------------------------------------------------------------------------------
// Presence
// ---------------------------------------------------------------------------------------------------------------------

// A member that exists only where its channel or VC is present on this router (by configuration or channel trimming),
// and is an empty Absent<TAG> elsewhere (TAG keeps two absent members of one struct at distinct addresses). Code names
// such a member only under `if constexpr` on the same flag, from template-dependent code, so an absent one never has to
// compile.
template <size_t TAG>
struct Absent {};
template <bool PRESENT, typename T, size_t TAG>
using Present = std::conditional_t<PRESENT, T, Absent<TAG>>;

// `value` where PRESENT, Absent<TAG> elsewhere: to initialize a Present member.
template <bool PRESENT, size_t TAG, typename T>
FORCE_INLINE constexpr Present<PRESENT, T, TAG> present(const T& value) {
    if constexpr (PRESENT) {
        return value;
    } else {
        return {};
    }
}

// One T<I> for each bit I of MASK, lowest first, found by its index at compile time (get<I>()); empty where MASK is 0.
// A channel or VC this router does not run has no member. Code names a member only under `if constexpr` on the same
// mask, from template-dependent code.
template <uint32_t MASK, template <size_t> class T>
struct Each {
    [[no_unique_address]] T<lowest_bit(MASK)> first;
    [[no_unique_address]] Each<without_lowest_bit(MASK), T> rest;

    template <size_t I>
    FORCE_INLINE T<I>& get() {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
    template <size_t I>
    FORCE_INLINE const T<I>& get() const {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
};
template <size_t I>
struct NotHeld : std::false_type {};
template <template <size_t> class T>
struct Each<0, T> {
    // Naming a member this container does not hold is an error where it is instantiated, not where it is written.
    template <size_t I>
    T<I>& get() {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
    template <size_t I>
    const T<I>& get() const {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Placement primitives
// ---------------------------------------------------------------------------------------------------------------------
// Each is compiler-only: it emits no instruction, it only constrains where the compiler may keep or move things.

// Keeps `obj` in memory at this point: its current value is stored before it and read back after it (GCC's documented
// semantics of an asm memory operand). The whole object is affected, so state meant to stay in registers must not
// share an object with it. It says nothing about where `obj` lives between such points. Use: state touched only on some
// passes (a forwarding channel's record, a later VC's receiver state), so the registers go to the state every pass
// touches (V4_PLACEMENT.md).
template <typename T>
FORCE_INLINE void keep_in_memory(T& obj) {
    asm volatile("" : "+m"(obj));
}

// Read group boundaries (std::atomic_signal_fence): a compiler barrier for memory. No access to memory other code could
// see (device registers, L1, any object whose address escapes) moves across either one, so the reads written between
// a begin and an end stay together, with no other such access among them, and their latencies overlap when their
// values are first used after the end. They do not stop arithmetic, or locals kept in registers, moving between the
// reads, and such memory used on both sides of a boundary is stored before it and reloaded after it.
FORCE_INLINE void begin_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }
FORCE_INLINE void end_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }

// An ordering point outside a read group (std::atomic_signal_fence): the same compiler barrier for memory, where code
// relies on the order of the memory accesses around it but they form no group.
FORCE_INLINE void order_point() { std::atomic_signal_fence(std::memory_order_seq_cst); }

// Byte enables of a whole-word inline write (credit registers and words).
constexpr uint32_t INLINE_WRITE_ALL_BYTES = 0xF;

// ---------------------------------------------------------------------------------------------------------------------
// Roles and NoC command buffers
// ---------------------------------------------------------------------------------------------------------------------

// This ERISC's roles: one each on a two-ERISC router, both on a single-ERISC router (channel trimming may leave an
// ERISC neither).
constexpr bool is_sender_role = any_sender_channel_serviced;
constexpr bool is_receiver_role = any_receiver_channel_serviced;

// The VCs whose forwards go to downstream routers here: the host lists downstream routers for VC0 and VC1.
constexpr bool vc_forwards_to_routers(size_t vc) { return vc == 0 || (vc == 1 && has_vc1); }

// Which NoC command buffer each kind of command uses on this ERISC (the host's ids), which kinds keep their buffer
// programmed between commands (a preset, through the stateful NoC API), and the rule they follow. A buffer that a kind
// keeps preset carries that kind only: any other command on it rewrites the preset (VC2's credit on VC0's credit
// preset, job 142; an inline write on the local-write preset, job 351). The atomics buffer carries no preset and no
// write from L1: atomics need the return address the firmware keeps there (job 366); upstream-router credits,
// stateless forward credits and teardowns, whole inline-register writes and increments, go on it too. Each ERISC's
// buffers are its own: two ERISCs issue on different NoCs (V4_BOUNDS.md §4). The table holds the router's own commands;
// an API that ignores its buffer argument is outside it (Blackhole's emulated L1 inline write always issues on
// write_cmd_buf, job 142).
struct CommandBuffers {
    // The host's buffer of each kind.
    static constexpr uint8_t atomics = write_at_cmd_buf;
    static constexpr uint8_t local_writes = tt::tt_fabric::local_chip_data_cmd_buf;               // receiver
    static constexpr const auto& worker_credits = sender_channel_ack_cmd_buf_ids;                // sender, by channel
    static constexpr const auto& forward_data = receiver_channel_forwarding_data_cmd_buf_ids;    // receiver, by VC
    static constexpr const auto& forward_credits = receiver_channel_forwarding_sync_cmd_buf_ids;  // receiver, by VC

    // Which kinds keep presets: none on one ERISC (both roles' commands share its buffers; V4_SINGLE_ERISC.md) or with
    // DPRINT or the watcher (their traffic shares them); forwards also not where every path uses one NoC, or in UDM
    // mode (the relay's stateless forwards share the data buffer).
    static constexpr bool presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
        NUM_ACTIVE_ERISCS > 1;
#else
        false;
#endif
    static constexpr bool forward_presets = presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode;

    // The kinds of command, one bit each; each worker channel's credits are a kind of their own (WORKER_CREDITS << CH).
    enum Kind : uint32_t {
        ATOMICS = 1u << 0,
        LOCAL_WRITES = 1u << 1,
        FORWARD_DATA = 1u << 2,
        FORWARD_CREDITS = 1u << 3,
        WORKER_CREDITS = 1u << 4,
    };

    // The kinds each of the four buffers (WR, RD, WR_REG, AT) carries on this ERISC. Worker credits by the host's
    // worker positions, serviced or not (the host assigns buffers by channel, whatever the trimming). The forwards of
    // every VC that forwards to routers are one kind: VCs on the same buffers share their presets.
    static constexpr std::array<uint32_t, 4> kinds_on = [] {
        std::array<uint32_t, 4> kinds{};
        kinds[atomics] |= ATOMICS;
        if (is_sender_role) {
            for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
                kinds[worker_credits[ch]] |= is_sender_channel_worker[ch] ? WORKER_CREDITS << ch : 0u;
            }
        }
        if (is_receiver_role) {
            kinds[local_writes] |= LOCAL_WRITES;
            for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
                if (vc_forwards_to_routers(vc)) {
                    kinds[forward_data[vc]] |= FORWARD_DATA;
                    kinds[forward_credits[vc]] |= forward_presets ? FORWARD_CREDITS : 0u;  // stateless: on atomics
                }
            }
        }
        return kinds;
    }();
    // The kinds kept preset here.
    static constexpr uint32_t preset_kinds =
        presets ? ~(ATOMICS | (forward_presets ? 0u : FORWARD_DATA | FORWARD_CREDITS)) : 0u;

    // Whether VC's forwards use VC0's two buffers (true where VC forwards to no router here).
    static constexpr bool forwards_on_vc0_bufs(size_t vc) {
        return !vc_forwards_to_routers(vc) ||
               (forward_data[vc] == forward_data[0] && forward_credits[vc] == forward_credits[0]);
    }
};
static_assert(
    [] {
        for (uint32_t kinds : CommandBuffers::kinds_on) {
            if ((kinds & CommandBuffers::preset_kinds) != 0 && popcount(kinds) != 1) {
                return false;
            }
        }
        return true;
    }(),
    "a command buffer kept preset carries one kind of command (the host's *_CMD_BUF_ID)");
static_assert(
    CommandBuffers::kinds_on[CommandBuffers::atomics] == CommandBuffers::ATOMICS,
    "only atomics, inline register writes and teardowns go on the atomics buffer");

}  // namespace fabric_router
