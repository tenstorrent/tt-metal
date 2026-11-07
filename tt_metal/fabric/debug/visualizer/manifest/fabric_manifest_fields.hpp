// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>
#include <tt_stl/type_name.hpp>

#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

// The router facts the manifest reads straight from what the router kernel is fed, one entry each, in one table per
// place a field sits.
namespace tt::tt_fabric::manifest {

// The index a per-channel or per-edge argument's name takes in its {}.
enum class ArgIndex : uint8_t {
    NONE,
    // Over this router's own counts, as most per-channel tables are indexed.
    COMPACT,
    // Over the fabric's maximum counts, as the free-slots stream table is indexed (StreamAssignment).
    FABRIC_POSITION,
    // The edge's VC in the first {}, and its edge number in the second.
    VC_AND_EDGE,
};

// The channel or edge a per-channel or per-edge field is read for. Zero for router and ERISC fields.
struct FieldIndex {
    uint32_t compact = 0;
    uint32_t fabric_position = 0;
    uint32_t vc = 0;
    uint32_t edge = 0;
    // The edge's rank among the router's edges, VC0's before VC1's, which its teardown semaphore is indexed by.
    uint32_t teardown = 0;
};

// When the builder emits a named argument.
enum class Emitted : uint8_t {
    ALWAYS,
    // Only when 2D routing is on. On 1D the field is left out.
    FABRIC_2D,
    // Only with the channel trimming capture on (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE).
    CHANNEL_TRIMMING_CAPTURE,
};

// A named compile-time argument, e.g. "EDM_STATUS_PTR_ADDR", or "SENDER_CH_{}_IS_INJECTION" for each channel.
struct NamedArg {
    template <std::size_t N>
    constexpr NamedArg(const char (&name)[N], ArgIndex index = ArgIndex::NONE, Emitted emitted = Emitted::ALWAYS) :
        name(name, N - 1), index(index), emitted(emitted) {}
    std::string_view name;
    ArgIndex index;
    Emitted emitted;
};

// A per-channel or per-edge builder member the router or the producer gets as a runtime argument. Runtime arguments
// are positional, so the field reads the member get_runtime_args pushes.
struct BuilderMember {
    uint32_t (*read)(const FabricEriscDatamoverBuilder& builder, const FieldIndex& index);
};

// What the kernel is fed that holds a field: the address, the stream id or the value.
using Source = std::variant<NamedArg, BuilderMember>;

template <typename E>
    requires std::is_enum_v<E>
constexpr content::Enum enum_of() {
    return {
        .type = ttsl::short_type_name<E>,
        .name_value = [](uint32_t value) -> std::optional<std::string_view> {
            using Underlying = std::underlying_type_t<E>;
            if (!std::in_range<Underlying>(value)) {
                return std::nullopt;
            }
            const auto enumerator = enchantum::cast<E>(static_cast<Underlying>(value));
            if (!enumerator.has_value()) {
                return std::nullopt;
            }
            return enchantum::to_string(*enumerator);
        },
    };
}

// What a table entry holds: a content (fabric_manifest_model.hpp), except that a stream is the register to read until
// the router pass reads the stream id.
using FieldContent = std::variant<content::L1, StreamRegField, content::Number, content::Flag, content::Enum>;

// A member every entry sets: it has no default, so an entry that leaves it out does not compile.
template <typename T>
struct Required {
    template <typename U>
        requires(!std::same_as<std::remove_cvref_t<U>, Required>)
    constexpr Required(U&& value) : value(std::forward<U>(value)) {}
    T value;
};

struct RouterField {
    // The field's key in the manifest.
    Required<std::string_view> key;
    Source source;
    Required<FieldCategory> category;
    Required<FieldContent> content;
    Required<std::string_view> description;
};

inline constexpr auto k_router_fields = [] {
    using enum FieldCategory;
    return std::array{
        // Lifecycle
        RouterField{
            .key = "edm_status",
            .source = "EDM_STATUS_PTR_ADDR",
            .category = LIFECYCLE,
            .content = register_l1_region<EDMStatus>(),
            .description = "How far the router has come through startup and teardown.",
        },
        RouterField{
            .key = "termination_signal",
            .source = "TERMINATION_SIGNAL_ADDR",
            .category = LIFECYCLE,
            .content = register_l1_region<TerminationSignal>(),
            .description = "Where the router is told to stop.",
        },
        RouterField{
            .key = "local_sync",
            .source = "EDM_LOCAL_SYNC_PTR_ADDR",
            .category = LIFECYCLE,
            .content = register_l1_region<uint32_t>(),
            .description = "The count the chip's routers sync through at startup: the others notify the master "
                           "router, which then notifies them all.",
        },
        RouterField{
            .key = "local_tensix_sync",
            .source = "EDM_LOCAL_TENSIX_SYNC_PTR_ADDR",
            .category = LIFECYCLE,
            .content = register_l1_region<uint32_t>(),
            .description = "The count the router waits on until its downstream tensix connections are ready.",
        },
        RouterField{
            .key = "handshake",
            .source = "HANDSHAKE_ADDR",
            .category = LIFECYCLE,
            .content = register_l1_region<erisc::datamover::handshake::handshake_info_t>(),
            .description = "The Ethernet handshake scratch. Each end of the link writes its scratch into the first "
                           "16 bytes of the other's.",
        },
        RouterField{
            .key = "handshake_sender",
            .source = "IS_HANDSHAKE_SENDER",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this end writes first: the sender repeats its write until the reply lands in its "
                           "own scratch, and the receiver waits for the sender's write and then replies once. The two "
                           "ends of a link differ.",
        },
        RouterField{
            .key = "erisc_sync",
            .source = "MULTI_RISC_TEARDOWN_SYNC_STREAM_ID",
            .category = LIFECYCLE,
            .content = register_stream_region<uint32_t>(StreamRegister::REMOTE_SRC),
            .description = "The register the router's ERISCs sync through at startup and teardown, led by ERISC0. "
                           "Unused when the router runs one ERISC.",
        },
        RouterField{
            .key = "retrain_sync",
            .source = "ETH_RETRAIN_LINK_SYNC_STREAM_ID",
            .category = LIFECYCLE,
            .content = register_stream_region<CoordinatedEriscContextSwitchState>(StreamRegister::REMOTE_SRC),
            .description = "The register the router's ERISCs sync through to retrain the link, led by the ERISC that "
                           "context switches.",
        },

        // Flow control
        RouterField{
            .key = "vc2_receiver_free_slots",
            .source = "VC2_RECEIVER_FREE_SLOTS_STREAM_ID",
            .category = FLOW_CONTROL,
            .content = register_stream_region<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The register reserved for the VC2 receiver's free slots. Null when the router has no VC2. "
                           "The router kernel declares it but does not read it.",
        },
        RouterField{
            .key = "notify_worker_src",
            .source = "NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR",
            .category = FLOW_CONTROL,
            .content = register_l1_region<uint32_t>(),
            .description = "Where the router stages the source of the inline write that updates a producer's read "
                           "counter. Allocated only on Blackhole.",
        },

        // Kernel parameters. Under DEBUG_PRINT_ENABLED the kernel ignores the context switch interval and the
        // handshake context switch timeout, and uses its own.
        RouterField{
            .key = "my_direction",
            .source = "MY_DIRECTION",
            .category = KERNEL_PARAMS,
            .content = enum_of<eth_chan_directions>(),
            .description = "The direction the router faces, which the kernel picks its downstream channels and "
                           "free-slot registers by. The router's key uses ControlPlane's direction.",
        },
        RouterField{
            .key = "wait_for_host_signal",
            .source = "WAIT_FOR_HOST_SIGNAL",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the router takes part in the chip's startup sync through local_sync. Without it, "
                           "the kernel ignores its local sync arguments.",
        },
        RouterField{
            .key = "idle_context_switching",
            .source = "IDLE_CONTEXT_SWITCHING",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the router context switches only once idle, rather than every interval.",
        },
        RouterField{
            .key = "context_switch_interval",
            .source = "SWITCH_INTERVAL",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "How many main loop iterations go by between context switches to base firmware; with "
                           "idle_context_switching, how many in a row without progress.",
        },
        RouterField{
            .key = "handshake_context_switch_timeout",
            .source = "DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "How many writes the handshake sender makes before it gives base firmware a turn to "
                           "route, and then starts counting again.",
        },
        RouterField{
            .key = "txq_spin_wait_send_data",
            .source = "ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether a data send spins until the Ethernet TX queue is free, instead of skipping the "
                           "send while it is busy.",
        },
        RouterField{
            .key = "txq_spin_wait_completion_ack",
            .source = "ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "The same, for a receiver's completion ack.",
        },
        RouterField{
            .key = "txq_accept_ahead",
            .source = "DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "How many packets each Ethernet TX queue accepts before the earlier ones are sent. "
                           "Written to both queues' ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD register at startup.",
        },
        RouterField{
            .key = "risc_cpu_data_cache",
            .source = "ENABLE_RISC_CPU_DATA_CACHE",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the kernel treats the RISC's data cache as on, and so invalidates it before "
                           "reading what other cores write to its L1.",
        },
        RouterField{
            .key = "vc0_bubble_flow_control",
            .source = "ENABLE_DEADLOCK_AVOIDANCE",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether VC0 runs bubble flow control for deadlock avoidance, which adds first-level acks: "
                           "VC0 senders then also get credits back when the peer receives a packet (acked).",
        },
        RouterField{
            .key = "speedy_vc0",
            .source = "ENABLE_SPEEDY_VC0",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether VC0 runs the speedy path: worker-only steps that pass credits in batches, so a "
                           "credit count can lag the packet count by up to a batch.",
        },
        RouterField{
            .key = "sender_credit_amortization",
            .source = "SENDER_CREDIT_AMORTIZATION_FREQUENCY",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "Speedy VC0 only: the sender reads completions once this many packets are outstanding, "
                           "and returns credits to its producer once this many have completed.",
        },
        RouterField{
            .key = "receiver_credit_amortization",
            .source = "RECEIVER_CREDIT_AMORTIZATION_FREQUENCY",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "Speedy VC0 only: the receiver acks completions to the peer's sender in batches of at "
                           "least this many packets.",
        },

        // Diagnostics
        RouterField{
            .key = "perf_telemetry",
            .source = "PERF_TELEMETRY_BUFFER_ADDR",
            .category = DIAGNOSTICS,
            .content = register_l1_region<std::byte>(
                0, static_cast<uint32_t>(FabricEriscDatamoverConfig::perf_telemetry_buffer_size), 1),
            .description = "Where the router records its bandwidth telemetry, when bandwidth telemetry is on. "
                           "Allocated only then, except on Blackhole, which always allocates it.",
        },
        RouterField{
            .key = "code_profiling",
            .source = "CODE_PROFILING_BUFFER_ADDR",
            .category = DIAGNOSTICS,
            .content = register_l1_region<CodeProfilingTimerResult>(0, get_max_code_profiling_timer_types()),
            .description = "Where the router accumulates each code profiling timer's cycles and captures, one result "
                           "per timer type. Allocated only with code profiling on.",
        },
        RouterField{
            .key = "channel_trimming_capture_enabled",
            .source = "ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE",
            .category = DIAGNOSTICS,
            .content = content::Flag{},
            .description = "Whether the router records which of its channels carry traffic, for a later run to trim "
                           "the unused ones.",
        },
        RouterField{
            .key = "channel_trimming_capture",
            .source =
                NamedArg{"RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS", ArgIndex::NONE, Emitted::CHANNEL_TRIMMING_CAPTURE},
            .category = DIAGNOSTICS,
            .content = register_l1_region<ChannelTrimmingOverrides>(),
            .description = "Where it records them: the packet sizes each sender channel saw, and which sender and "
                           "receiver channels carried traffic.",
        },

        // What differs between a router's ERISCs by design: each one's role, its handshake, and its context switching.
        RouterField{
            .key = "local_handshake_master",
            .source = "IS_LOCAL_HANDSHAKE_MASTER",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this ERISC leads the chip's startup sync (the chip's local_sync). The kernel reads "
                           "it only with wait_for_host_signal.",
        },
        RouterField{
            .key = "handshake_enabled",
            .source = "ENABLE_ETHERNET_HANDSHAKE",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether this ERISC runs the Ethernet handshake with the peer at startup.",
        },
        RouterField{
            .key = "context_switch_enabled",
            .source = "ENABLE_CONTEXT_SWITCH",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether this ERISC hands control back to base firmware.",
        },
        RouterField{
            .key = "services_sender_channels",
            .source = "RISC_SERVICES_SENDER_CHANNELS",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this ERISC has the sender role, so it sets up the sender channels' worker "
                           "interfaces at startup, and its sender-side TXQ1 state in multi-TXQ mode. On Blackhole with "
                           "two ERISCs only ERISC0 has it, and otherwise every ERISC does. Channel trimming doesn't "
                           "change it.",
        },
        RouterField{
            .key = "services_receiver_channels",
            .source = "RISC_SERVICES_RECEIVER_CHANNELS",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this ERISC has the receiver role, so it sets up its TXQ1 state at startup in "
                           "multi-TXQ mode. On Blackhole with two ERISCs only ERISC1 has it, and otherwise every ERISC "
                           "does. Channel trimming doesn't change it.",
        },
        RouterField{
            .key = "opens_vc0_downstream_connections",
            .source = "OPEN_DOWNSTREAM_VC0_CONNECTIONS",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this ERISC opens the router's VC0 connections to its downstream routers at "
                           "startup: set when it runs receiver 0's step.",
        },
        RouterField{
            .key = "opens_vc1_downstream_connections",
            .source = "OPEN_DOWNSTREAM_VC1_CONNECTIONS",
            .category = LIFECYCLE,
            .content = content::Flag{},
            .description = "Whether this ERISC opens the router's VC1 connections to its downstream routers at "
                           "startup: set when it runs receiver 0's step on a router that crosses VC0 traffic over to "
                           "VC1, or receiver 1's step on a router that services VC1. The kernel reads it only with "
                           "FABRIC_2D_VC1_ACTIVE.",
        },
        RouterField{
            .key = "interrupts_enabled",
            .source = "ENABLE_INTERRUPTS",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether this ERISC is configured with interrupts on. The router kernel declares it but "
                           "does not read it.",
        },
        RouterField{
            .key = "teardown_check_iterations",
            .source = "ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "How many passes over its channel steps the main loop makes between its checks for "
                           "termination and context switching.",
        },
        RouterField{
            .key = "telemetry_stats_mask",
            .source = "FABRIC_TELEMETRY_STATS_MASK",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The FabricTelemetryStatistic bits this ERISC collects; 0 when its telemetry is off.",
        },
        RouterField{
            .key = "telemetry_enabled",
            .source = "ENABLE_FABRIC_TELEMETRY",
            .category = DIAGNOSTICS,
            .content = content::Flag{},
            .description = "Whether this ERISC collects fabric telemetry. telemetry_stats_mask says which statistics.",
        },
        RouterField{
            .key = "perf_telemetry_mode",
            .source = "PERF_TELEMETRY_MODE",
            .category = DIAGNOSTICS,
            .content = content::Number{},
            .description = "The kernel's PerfTelemetryRecorderType: 0 for none, 1 for low-resolution bandwidth. The "
                           "enum is device-side only, so the value is written as a number.",
        },
        RouterField{
            .key = "code_profiling_timers",
            .source = "CODE_PROFILING_ENABLED_TIMERS",
            .category = DIAGNOSTICS,
            .content = content::Number{},
            .description = "The CodeProfilingTimerType bits the kernel times; 0 when code profiling is off.",
        },
        RouterField{
            .key = "sender_txq",
            .source = "SENDER_TXQ_ID",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The Ethernet TX queue the router sends packets on.",
        },
        RouterField{
            .key = "receiver_txq",
            .source = "RECEIVER_TXQ_ID",
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The Ethernet TX queue the router sends credits back on: 1 in 2-ERISC mode, otherwise the "
                           "sender's.",
        },
        RouterField{
            .key = "intermesh_router",
            .source = "IS_INTERMESH_ROUTER",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the router's link goes to another mesh.",
        },
        RouterField{
            .key = "intermesh_edge",
            .source = "IS_INTERMESH_ROUTER_ON_EDGE",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the router's channel is one of the chip's inter-mesh-facing channels "
                           "(ControlPlane::get_intermesh_facing_eth_chans).",
        },
        RouterField{
            .key = "intramesh_edge",
            .source = "IS_INTRAMESH_ROUTER_ON_EDGE",
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the router is an intra-mesh router on a chip with inter-mesh links, which checks "
                           "each packet for an inter-mesh exit.",
        },
        RouterField{
            .key = "mesh_x_size",
            .source = NamedArg{"MESH_X_SIZE", ArgIndex::NONE, Emitted::FABRIC_2D},
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The mesh's columns, which bound the router's 2D route table.",
        },
        RouterField{
            .key = "mesh_y_size",
            .source = NamedArg{"MESH_Y_SIZE", ArgIndex::NONE, Emitted::FABRIC_2D},
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The mesh's rows, which bound the router's 2D route table.",
        },
        RouterField{
            .key = "remote_worker_sender_channel",
            .source = "REMOTE_WORKER_SENDER_CHANNEL",
            .category = CONTROL_INFO,
            .content = content::Number{},
            .description = "The peer router's worker sender channel. In mux mode the router writes it as the source "
                           "channel of the packets it receives. 0 when the router has no peer.",
        },
    };
}();

inline constexpr auto k_sender_channel_fields = [] {
    using enum FieldCategory;
    return std::array{
        RouterField{
            .key = "is_injection_channel",
            .source = NamedArg{"SENDER_CH_{}_IS_INJECTION", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .content = content::Flag{},
            .description = "Whether the channel injects traffic into the fabric. With deadlock avoidance on, an "
                           "injection channel uses bubble flow control.",
        },
        RouterField{
            .key = "free_slots",
            .source = NamedArg{"SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", ArgIndex::FABRIC_POSITION},
            .category = FLOW_CONTROL,
            .content = register_stream_region<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The channel's slots not holding a packet waiting to be sent: the producer takes one per "
                           "packet it writes, and the router returns it as soon as it sends the packet over Ethernet. "
                           "The producer gets its credit back later, at the ack or the completion.",
        },
        RouterField{
            .key = "producer_credit_return_noc",
            .source = NamedArg{"SENDER_CH_{}_ACK_NOC_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .content = content::Number{},
            .description = "The NoC the router returns the producer's credits on.",
        },
        RouterField{
            .key = "producer_credit_return_cmd_buf",
            .source = NamedArg{"SENDER_CH_{}_ACK_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .content = enum_of<FabricEriscDatamoverConfig::NocCmdBuf>(),
            .description = "The NoC command buffer it returns them through.",
        },
        RouterField{
            .key = "connection",
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                return static_cast<uint32_t>(builder.sender_channels_connection_semaphore_id[index.compact]);
            }},
            .category = CONTROL_INFO,
            .content = register_l1_region<connection_interface::ConnectionValue>(),
            .description = "The state of the producer's connection: the producer writes open and close requested, and "
                           "the router writes unused once it has torn the connection down. In the router's runtime "
                           "args.",
        },
        RouterField{
            .key = "conn_info",
            .source = NamedArg{"LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", ArgIndex::COMPACT},
            .category = CONTROL_INFO,
            .content = register_l1_region<EDMChannelWorkerLocationInfo>(),
            .description = "Where the producer writes where it is, and the router its read counter.",
        },
        RouterField{
            .key = "buffer_index_sem",
            // What the builder gives the router to reset at startup (get_runtime_args): set only on the channels it
            // treats as worker channels. A sibling router's adapter never keeps a cursor, so the others are 0, which
            // leaves the field null.
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                return static_cast<uint32_t>(
                    builder.downstream_vcs_sender_channel_buffer_index_semaphore_id[index.compact]);
            }},
            .category = CONTROL_INFO,
            .content = register_l1_region<SenderChannelProducerCursor>(),
            .description = "Where the worker channel's producer keeps its write cursor across connections, for the "
                           "next producer to pick up. In the producer's runtime args, from the channel's connection "
                           "spec, and the router's, which zeroes it at startup. Null on the other channels, whose "
                           "producer keeps none.",
        },
        RouterField{
            .key = "static_connection",
            .source = NamedArg{"SENDER_CH_{}_WAIT_STATIC_CONNECTION", ArgIndex::COMPACT},
            .category = CONTROL_INFO,
            .content = content::Flag{},
            .description = "Whether a sibling router connects to the channel once, at startup: before its main loop, "
                           "each ERISC that runs the channel waits for that connection. The builder gives an ERISC "
                           "that doesn't run it 0.",
        },
        RouterField{
            .key = "live_check_skip",
            .source = NamedArg{"SENDER_CH_{}_LIVE_CHECK_SKIP", ArgIndex::COMPACT},
            .category = CONTROL_INFO,
            .content = content::Flag{},
            .description = "Whether the router skips the connection liveness check on the channel: it has a static "
                           "peer, or the builder forces the skip.",
        },
    };
}();

inline constexpr auto k_receiver_channel_fields = [] {
    using enum FieldCategory;
    return std::array{
        RouterField{
            .key = "forwarding_disabled",
            .source = NamedArg{"DISABLE_RX_CH{}_FORWARDING", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the receiver's step skips handing its packets on, to siblings or the local chip. "
                           "Set for the speedy VC0 and VC2 steps, which hand packets on themselves, and when a channel "
                           "trimming capture saw the channel do neither.",
        },
        RouterField{
            .key = "intermesh_ingress",
            .source = NamedArg{"IS_RECEIVER_CHANNEL_{}_INTERMESH_INGRESS", ArgIndex::COMPACT, Emitted::FABRIC_2D},
            .category = KERNEL_PARAMS,
            .content = content::Flag{},
            .description = "Whether the receiver's packets arrive from another mesh: the kernel re-encodes their "
                           "route from this mesh's route table before deciding where they go.",
        },
        // A NoC is a number, not an enum: NOC's RISCV_*_default aliases share its values, so they have no unique name.
        RouterField{
            .key = "forward_noc",
            .source = NamedArg{"RX_CH_{}_FWD_NOC_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The NoC the receiver forwards packets to sibling routers on.",
        },
        RouterField{
            .key = "forward_data_cmd_buf",
            .source = NamedArg{"RX_CH_{}_FWD_DATA_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = enum_of<FabricEriscDatamoverConfig::NocCmdBuf>(),
            .description = "The NoC command buffer it writes each forwarded packet through.",
        },
        RouterField{
            .key = "forward_sync_cmd_buf",
            .source = NamedArg{"RX_CH_{}_FWD_SYNC_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = enum_of<FabricEriscDatamoverConfig::NocCmdBuf>(),
            .description = "The NoC command buffer it sends the credit increment that announces a forwarded packet "
                           "through.",
        },
        RouterField{
            .key = "local_write_noc",
            .source = NamedArg{"RX_CH_{}_LOCAL_WRITE_NOC_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = content::Number{},
            .description = "The NoC the receiver delivers packets to the local chip on.",
        },
        RouterField{
            .key = "local_write_cmd_buf",
            .source = NamedArg{"RX_CH_{}_LOCAL_WRITE_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .content = enum_of<FabricEriscDatamoverConfig::NocCmdBuf>(),
            .description = "The NoC command buffer it delivers them through.",
        },
        RouterField{
            .key = "pkts_sent",
            .source = NamedArg{"TO_RECEIVER_{}_PKTS_SENT_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .content = register_stream_region<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The packets the peer's sender has sent the channel that the receiver has not taken yet: "
                           "the peer adds one per packet it sends over Ethernet, and the receiver takes one when it "
                           "acks the packet with first-level ack, or else once it has forwarded or delivered it.",
        },
    };
}();

inline constexpr auto k_downstream_edge_fields = [] {
    using enum FieldCategory;
    return std::array{
        RouterField{
            .key = "free_slots",
            .source = NamedArg{"VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", ArgIndex::VC_AND_EDGE},
            .category = FLOW_CONTROL,
            .content = register_stream_region<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The router's view of the free slots in the sender channel the edge lands on: the router "
                           "takes one per packet it forwards, and the sibling returns it when the packet is acked with "
                           "first-level ack, or else when it completes.",
        },
        RouterField{
            .key = "teardown_sem",
            // get_runtime_args pushes -1 for a semaphore the builder has not set.
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                const auto& ids = builder.receiver_channels_downstream_teardown_semaphore_id;
                return static_cast<uint32_t>(index.teardown < ids.size() ? ids[index.teardown].value_or(-1) : -1);
            }},
            .category = CONTROL_INFO,
            .content = register_l1_region<uint32_t>(),
            .description = "Where the sibling acks the router closing the edge's connection at teardown. In the "
                           "router's runtime args.",
        },
    };
}();

// The number of {} an argument index fills.
constexpr size_t num_placeholders(ArgIndex index) {
    switch (index) {
        case ArgIndex::NONE: return 0;
        case ArgIndex::COMPACT:
        case ArgIndex::FABRIC_POSITION: return 1;
        case ArgIndex::VC_AND_EDGE: return 2;
    }
    return 0;
}

constexpr size_t count_placeholders(std::string_view name) {
    size_t count = 0;
    for (size_t pos = name.find("{}"); pos != std::string_view::npos; pos = name.find("{}", pos + 2)) {
        ++count;
    }
    return count;
}

// Validate fields from a particular table. An argument's name has as many {} as its index fills, and only a table
// that is per channel or per edge has indexed arguments or builder members.
constexpr bool sources_name_channels(const auto& fields, bool per_channel) {
    for (const auto& field : fields) {
        if (const auto* arg = std::get_if<NamedArg>(&field.source)) {
            const size_t placeholders = count_placeholders(arg->name);
            if (placeholders != num_placeholders(arg->index) || (placeholders > 0) != per_channel) {
                return false;
            }
        } else if (!per_channel) {
            return false;
        }
    }
    return true;
}
static_assert(sources_name_channels(k_router_fields, false));
static_assert(sources_name_channels(k_sender_channel_fields, true));
static_assert(sources_name_channels(k_receiver_channel_fields, true));
static_assert(sources_name_channels(k_downstream_edge_fields, true));

// Every named argument and define the router kernel is fed is read by a field table above or by the router pass, or
// is listed as unrecorded with a reason; otherwise the router pass throws (check_kernel_inputs_accounted). Arguments
// are matched by family, so "SENDER_CH_{}_IS_INJECTION" covers every channel's; defines are matched exactly.

// Where the index at `pos` ends, or `pos` when no index starts there.
constexpr size_t skip_index(std::string_view name, size_t pos) {
    if (name.substr(pos, 2) == "{}") {
        return pos + 2;
    }
    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
        ++pos;
    }
    return pos;
}

// Whether two argument names are the same but for their indices: a run of digits or a {} matches any other.
constexpr bool same_arg_family(std::string_view a, std::string_view b) {
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
        const size_t next_i = skip_index(a, i);
        const size_t next_j = skip_index(b, j);
        if ((next_i != i) != (next_j != j)) {
            return false;
        }
        if (next_i != i) {
            i = next_i;
            j = next_j;
        } else if (a[i++] != b[j++]) {
            return false;
        }
    }
    return i == a.size() && j == b.size();
}

// The named arguments the router pass reads outside the field tables, to work out facts.
inline constexpr auto k_router_pass_args = std::to_array<std::string_view>({
    "NUM_ACTIVE_ERISCS",
    "ACTUAL_VC{}_SENDER_CHANNELS",
    "TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR",
    "TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR",
    "LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR",
    "LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR",
    "IS_2D_FABRIC",
    "ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE",
    "ENABLE_SPEEDY_VC0",
    "FABRIC_TENSIX_EXTENSION_MUX_MODE",
    "CHANNEL_BUFFER_SIZE",
    "VC{}_FABRIC_POSITION_START",
    "IS_SENDER_CHANNEL_{}_SERVICED",
    "IS_RECEIVER_CHANNEL_{}_SERVICED",
    "VC{}_USES_COUNTER_CREDITS",
    "TO_SENDER_{}_PKTS_COMPLETED_ID",
    "TO_SENDER_{}_PKTS_ACKED_ID",
    "ENABLE_FIRST_LEVEL_ACK_VC{}",
    "NUM_DOWNSTREAM_SENDERS_VC{}",
});

// The defines the router pass reads.
inline constexpr auto k_router_pass_defines = std::to_array<std::string_view>({
    "FABRIC_2D_VC1_SERVICED",
    "FABRIC_2D_VC2_SERVICED",
    "FABRIC_2D_VC0_CROSSOVER_TO_VC1",
});

// A kernel input the manifest leaves out, and why.
struct Unrecorded {
    std::string_view name;
    std::string_view reason;
};

inline constexpr std::string_view k_not_yet_described = "Not yet described.";
inline constexpr std::string_view k_link_sync =
    "The streaming profiler's link sync, fed on Blackhole when the profiler is on "
    "(link_sync::add_router_compile_args). Not yet described.";

inline constexpr auto k_unrecorded_args = std::to_array<Unrecorded>({
    // Recorded elsewhere
    {"LOCAL_HANDSHAKE_MASTER_ETH_CHAN", "The chip's local_sync master_eth_chan, from KernelCreationContext."},
    {"NUM_LOCAL_EDMS", "The chip's local_sync num_routers, from KernelCreationContext."},
    {"EDM_CHANNELS_MASK", "The chip's local_sync router_channels_mask, from KernelCreationContext."},
    {"MY_ERISC_ID", "The ERISC's index in the router's eriscs, which the manifest lists in RISC order."},
    {"NUM_RECEIVER_CHANNELS",
     "The router's receiver count. The manifest lists the receivers from the builder's VC shape."},

    // The same on every router
    {"MAX_NUM_SENDER_CHANNELS", "builder_config::num_max_sender_channels."},
    {"MAX_NUM_RECEIVER_CHANNELS", "builder_config::num_max_receiver_channels."},
    {"MAX_NUM_VCS", "builder_config::MAX_NUM_VCS."},
    {"NUM_ETH_PORTS", "The SoC's Ethernet channel count, which sizes the kernel's port_direction_table."},

    // Not read by the kernel, or carried by another input
    {"SKIP_SRC_CH_ID_UPDATE",
     "Not read: the kernel derives the flag from FABRIC_TENSIX_EXTENSION_MUX_MODE (fabric_erisc_router_ct_args.hpp)."},
    {"PACKED_DOWNSTREAM_VC{}_SENDER_CHANNEL_IDS",
     "The edges' landing channels, packed; fed only in 2D with channel trimming capture. Each edge records its own."},

    // Not yet described
    {"MY_ETH_CHANNEL", k_not_yet_described},
    {"NUM_SENDER_CHANNELS", k_not_yet_described},
    {"NUM_DOWNSTREAM_CHANNELS", k_not_yet_described},
    {"NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS", k_not_yet_described},
    {"VC{}_DOWNSTREAM_EDM_SIZE", k_not_yet_described},
    {"FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR", k_not_yet_described},
    {"EDM_NOC_VC", k_not_yet_described},
    {"FORCE_ALL_PATHS_TO_USE_SAME_NOC", k_not_yet_described},
    {"UDM_MODE", k_not_yet_described},
    {"LOCAL_RELAY_NUM_BUFFERS", k_not_yet_described},
    {"TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID", k_not_yet_described},

    // Fed by the streaming profiler
    {"LINK_SYNC_ROLE", k_link_sync},
    {"LINK_SYNC_L1_ADDR", k_link_sync},
    {"LINK_SYNC_CHECK", k_link_sync},
});

inline constexpr std::string_view k_fabric_wide_define =
    "Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines).";

inline constexpr auto k_unrecorded_defines = std::to_array<Unrecorded>({
    {"FABRIC_2D", "The same fact as IS_2D_FABRIC: both come from is_2D_routing_enabled."},
    {"FABRIC_2D_VC1_ACTIVE", "The same fact as ACTUAL_VC1_SENDER_CHANNELS > 0: set when the router has VC1 senders."},
    {"ROUTING_MODE", k_fabric_wide_define},
    {"UDM_MODE", k_fabric_wide_define},
    {"FABRIC_1D_PKT_HDR_EXTENSION_WORDS", k_fabric_wide_define},
    {"FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE", k_fabric_wide_define},
    {"FABRIC_EXPRESS_ENABLED", k_fabric_wide_define},
    {"FABRIC_EXPRESS_AXIS", k_fabric_wide_define},
    {"FORCE_WATCHER_OFF", "CreateEthernetKernel adds it when watcher is disabled on Ethernet cores."},
});

constexpr bool table_reads(const auto& fields, std::string_view name) {
    return std::ranges::any_of(fields, [&](const RouterField& field) {
        const auto* arg = std::get_if<NamedArg>(&field.source);
        return arg != nullptr && same_arg_family(arg->name, name);
    });
}

// Whether a field table or the router pass reads the named argument `name`.
constexpr bool manifest_reads_arg(std::string_view name) {
    return table_reads(k_router_fields, name) || table_reads(k_sender_channel_fields, name) ||
           table_reads(k_receiver_channel_fields, name) || table_reads(k_downstream_edge_fields, name) ||
           std::ranges::any_of(k_router_pass_args, [&](std::string_view arg) { return same_arg_family(arg, name); });
}

constexpr bool is_unrecorded_arg(std::string_view name) {
    return std::ranges::any_of(
        k_unrecorded_args, [&](const Unrecorded& arg) { return same_arg_family(arg.name, name); });
}

constexpr bool manifest_reads_define(std::string_view name) {
    return std::ranges::find(k_router_pass_defines, name) != k_router_pass_defines.end();
}

constexpr bool is_unrecorded_define(std::string_view name) {
    return std::ranges::find(k_unrecorded_defines, name, &Unrecorded::name) != k_unrecorded_defines.end();
}

// An input the manifest reads cannot also be listed as unrecorded.
static_assert(std::ranges::none_of(k_unrecorded_args, [](const Unrecorded& arg) {
    return manifest_reads_arg(arg.name);
}));
static_assert(std::ranges::none_of(k_unrecorded_defines, [](const Unrecorded& define) {
    return manifest_reads_define(define.name);
}));

}  // namespace tt::tt_fabric::manifest
