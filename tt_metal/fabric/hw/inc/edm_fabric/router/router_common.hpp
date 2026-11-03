// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compile-time facts both router programs use: the architecture, the VCs and modes, stream registers, the downstream
// direction maps, the Ethernet TX queues, and the roles and NoC command buffers. Everything here is derived from the
// compile-time args (fabric_erisc_router_ct_args.hpp); nothing here holds state. The router-independent helpers are
// router_utils.hpp's.

#include <array>
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"
#include "noc_overlay_parameters.h"
#include "api/alignment.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/edm_fabric_counters.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "hostdevcommon/fabric_common.h"
#include "hostdevcommon/fabric_telemetry_msgs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"

#include "router_utils.hpp"

namespace fabric_router {

// The router is written against the fabric library's names (its packet, channel and connection types, the routing
// codec, the router state and handshake API), so they are visible here unqualified.
using namespace tt::tt_fabric;

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

// VC1 (multi-mesh fabrics): VC1 sender channels, a VC1 receiver channel, and VC1's downstream connections in the
// runtime args.
constexpr bool has_vc1 =
#if defined(FABRIC_2D_VC1_ACTIVE)
    true;
#else
    false;
#endif

// VC2 (neighbour exchange): a worker channel and a receiver channel that only delivers.
constexpr bool has_vc2 = ACTUAL_VC2_SENDER_CHANNELS > 0;

// Tensix mux mode (FabricTensixConfig::MUX): the fabric's mux of each direction feeds that router's worker channel with
// everything for its link and takes the forwards of the other directions' routers. The host's flag is also set under
// UDM, which is its own mode.
constexpr bool tensix_mux_mode = fabric_tensix_extension_mux_mode && !udm_mode;

// The count in a stream credit register (packets received, free slots). A count never exceeds its channel's slot
// count, and every channel static_asserts its slot count against this mask, so the low 11 bits hold it.
constexpr uint32_t stream_count_mask = 0x7FFu;
static_assert(stream_count_mask <= ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));

// The Ethernet firmware's heartbeat word. While the router runs instead of the base firmware, each loop keeps it moving
// every HEARTBEAT_INTERVAL rounds, the count in the lower half and UMD's fabric signature in the upper half (the host's
// contract: hostdevcommon/fabric_common.h).
constexpr uint32_t heartbeat_addr =
    is_blackhole_arch ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE;
constexpr uint32_t HEARTBEAT_INTERVAL = FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS;  // rounds
constexpr uint32_t HEARTBEAT_SIGNATURE = FABRIC_KERNEL_HEARTBEAT_MAGIC;
static_assert((HEARTBEAT_INTERVAL & (HEARTBEAT_INTERVAL - 1)) == 0, "the round count is tested with a mask");

// ---------------------------------------------------------------------------------------------------------------------
// Blocks in L1 the host and the base firmware share with the router
// ---------------------------------------------------------------------------------------------------------------------

// The host's termination signal for this router.
FORCE_INLINE volatile tt::tt_fabric::TerminationSignal* termination_signal() {
    return reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
}

// The routing table in this router's L1 (written by the host).
FORCE_INLINE const tt::tt_fabric::routing_l1_info_t& l1_routing_table() {
    return *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
}

// The FabricTelemetry block the host reads (static info, router state, heartbeats, the dynamic bandwidth).
FORCE_INLINE volatile tt_l1_ptr FabricTelemetry* telemetry_block() {
    return reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------------------------------

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

// A router never forwards back out its own facing, so its downstream routers are the other four directions, packed
// densely in enum order (E, W, N, S, Z) with its own direction removed (the codec's compact dispatch order): the
// compact index of a downstream direction.
constexpr auto downstream_directions = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
constexpr uint32_t max_downstreams = downstream_directions.size();

constexpr eth_chan_directions downstream_direction(size_t compact) { return downstream_directions[compact]; }

// The compact index of downstream direction `dir` (dir != my_direction): a router lists the other directions in enum
// order, skipping its own facing.
constexpr size_t downstream_compact_index(eth_chan_directions dir) {
    return static_cast<size_t>(dir) < static_cast<size_t>(my_direction) ? static_cast<size_t>(dir)
                                                                        : static_cast<size_t>(dir) - 1;
}

// Routing names a downstream by its direction (2D: action bit d is direction d); 1D's one downstream is named 0. The
// downstream tables keep the host's compact order, and downstream_entry maps a name to its entry there.
constexpr size_t downstream_entry(size_t name) {
    return is_2d ? downstream_compact_index(static_cast<eth_chan_directions>(name)) : 0;
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
// keeps preset carries that kind only: any other command on it rewrites the preset (a second worker channel's credit,
// or an inline write, on a preset buffer would). The atomics buffer carries no preset and no write from L1: atomics
// need the return address the firmware keeps there; upstream-router credits, stateless forward credits and teardowns,
// whole inline-register writes and increments, go on it too. Each ERISC's buffers are its own: two ERISCs issue on
// different NoCs. The table holds the router's own commands; an API that ignores its buffer argument is outside it
// (Blackhole's emulated L1 inline write always issues on write_cmd_buf).
struct CommandBuffers {
    // The host's buffer of each kind.
    static constexpr uint8_t atomics = write_at_cmd_buf;
    static constexpr uint8_t local_writes = tt::tt_fabric::local_chip_data_cmd_buf;               // receiver
    static constexpr const auto& worker_credits = sender_channel_ack_cmd_buf_ids;                 // sender, by channel
    static constexpr const auto& forward_data = receiver_channel_forwarding_data_cmd_buf_ids;     // receiver, by VC
    static constexpr const auto& forward_credits = receiver_channel_forwarding_sync_cmd_buf_ids;  // receiver, by VC

    // Which kinds keep presets: none on one ERISC (both roles' commands share its buffers) or with
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

// ---------------------------------------------------------------------------------------------------------------------
// Supported configurations
// ---------------------------------------------------------------------------------------------------------------------
//   - Blackhole; two ERISCs (one role each, a TX queue each) or one ERISC (both roles);
//   - 1D and 2D: lines, meshes, rings and tori (bubble flow control; first-level acks on VC0); strict atomics;
//   - VC0; VC1 on 2D multi-mesh fabrics (intermesh landing, crossover and exits);
//   - VC2 on 2D routers, with or without VC1: neighbour exchange (a worker channel into a receiver channel that only
//     delivers);
//   - Tensix mux mode on one ERISC, VC0 (the fabric's muxes feed the worker channel and take the forwards);
//   - UDM on 2D meshes, one ERISC (packets for this chip go to the fabric's relay);
//   - the speedy path (VC0 worker-only routers whose receiver channel only delivers: it skips decode and admission);
//   - the host's telemetry options: the perf recorder at low resolution and the dynamic stats;
//   - channel trimming: the capture, and a replay's routers with channels compiled out (an ERISC left with nothing to
//     service runs rounds only, router_round.hpp's run_idle).

// UDM as the host builds it (compute_mesh_router_builder.cpp, connect_to_local_tensix_builder): one ERISC (any Tensix
// extension), 2D meshes, one local Tensix connection (the relay), mux mode off.
constexpr bool udm_config_supported = NUM_ACTIVE_ERISCS == 1 && is_2d && !has_vc1 && !enable_deadlock_avoidance &&
                                      !tensix_mux_mode && num_ds_or_local_tensix_connections == 1;

// VC2 as the host builds it (FabricBuilderConfig: Blackhole 2D, not UDM or the mux), with or without VC1: one worker
// channel into a receiver channel of its own that only delivers (neighbour exchange), counter credits.
constexpr bool vc2_config_supported = is_2d && ACTUAL_VC2_SENDER_CHANNELS == 1 && is_vc_forwarding_disabled[2] &&
                                      receiver_channel_uses_counter_credits(VC2_RECEIVER_CHANNEL);

// The speedy path as the host selects it (ENABLE_SPEEDY_VC0): no bubble flow control or first-level acks on VC0, VC0's
// receiver forwarding off. Its credit amortization args do not apply here (credits
// go back per completion group, as everywhere).
constexpr bool speedy_config_supported =
    !enable_deadlock_avoidance && !ENABLE_FIRST_LEVEL_ACK_VC0 && disable_rx_ch0_forwarding && !udm_mode;

static_assert(is_blackhole_arch, "the router supports Blackhole only");
static_assert(NUM_ACTIVE_ERISCS == 1 || NUM_ACTIVE_ERISCS == 2, "the router runs on one ERISC or two");
static_assert(NUM_ACTIVE_ERISCS == 1 || multi_txq_enabled, "on two ERISCs each role needs a TX queue of its own");
static_assert(
    NUM_ACTIVE_ERISCS == 1 || !(is_sender_role && is_receiver_role), "on two ERISCs each ERISC runs one role");
static_assert(NUM_RECEIVER_CHANNELS == 1 + (has_vc1 ? 1 : 0) + (has_vc2 ? 1 : 0), "one receiver channel per VC");
static_assert(!has_vc1 || is_2d, "VC1 needs a 2D fabric");
// VC1 traffic would go router to router around the muxes, but the host gives VC1's sender channels no slots there.
static_assert(!(tensix_mux_mode && has_vc1), "VC1 with the Tensix mux extension is not supported");
static_assert(!has_vc2 || vc2_config_supported, "VC2 is supported as neighbour exchange on 2D routers");
static_assert(
    !ENABLE_FIRST_LEVEL_ACK_VC1 && !ENABLE_FIRST_LEVEL_ACK_VC2, "first-level acks on VC1 / VC2 are not supported");
static_assert(!is_vc_crossover_to_vc1[2], "VC2's crossover onto VC1 is not supported");
static_assert(!udm_mode || udm_config_supported, "UDM is supported on one-ERISC 2D meshes with one relay");
static_assert(
    !super_speedy_mode || speedy_config_supported,
    "the speedy path needs VC0 without bubble flow control, "
    "first-level acks or receiver forwarding");
static_assert(!ENABLE_RISC_CPU_DATA_CACHE, "the CPU data cache is not supported");
static_assert(!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA, "the TX queue's spin-wait send is not supported");
static_assert(
    perf_telemetry_mode == PerfTelemetryRecorderType::NONE ||
        perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH,
    "the perf recorder supports low-resolution bandwidth only");
static_assert(!tensix_mux_mode || NUM_ACTIVE_ERISCS == 1, "Tensix mux mode is supported on one ERISC");
static_assert(
    num_ds_or_local_tensix_connections == 0 || tensix_mux_mode || udm_mode,
    "local Tensix connections are supported in mux mode and UDM only");

}  // namespace fabric_router
