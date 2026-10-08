// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

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

// Kinds (fabric_manifest_model.hpp) whose element is a T.
template <typename T>
constexpr kind::L1Value l1_value() {
    return {element_type<T>()};
}

template <typename T>
constexpr kind::Stream stream(StreamRegister reg) {
    return {reg, element_type<T>()};
}

template <typename E>
    requires std::is_enum_v<E>
constexpr kind::Enum enum_kind() {
    return {
        element_type<E>(),
        [](uint32_t value) { return enchantum::cast<E>(static_cast<std::underlying_type_t<E>>(value)).has_value(); },
    };
}

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
    Required<Kind> kind;
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
            .kind = l1_value<EDMStatus>(),
            .description = "How far the router has come through startup and teardown.",
        },
        RouterField{
            .key = "termination_signal",
            .source = "TERMINATION_SIGNAL_ADDR",
            .category = LIFECYCLE,
            .kind = l1_value<TerminationSignal>(),
            .description = "Where the router is told to stop.",
        },
        RouterField{
            .key = "local_sync",
            .source = "EDM_LOCAL_SYNC_PTR_ADDR",
            .category = LIFECYCLE,
            .kind = l1_value<uint32_t>(),
            .description = "The count the chip's routers sync through at startup: the others notify the master "
                           "router, which then notifies them all.",
        },
        RouterField{
            .key = "local_tensix_sync",
            .source = "EDM_LOCAL_TENSIX_SYNC_PTR_ADDR",
            .category = LIFECYCLE,
            .kind = l1_value<uint32_t>(),
            .description = "The count the router waits on until its downstream tensix connections are ready.",
        },
        RouterField{
            .key = "handshake",
            .source = "HANDSHAKE_ADDR",
            .category = LIFECYCLE,
            .kind = l1_value<erisc::datamover::handshake::handshake_info_t>(),
            .description = "The Ethernet handshake scratch. Each end of the link writes its scratch into the first "
                           "16 bytes of the other's.",
        },
        RouterField{
            .key = "handshake_sender",
            .source = "IS_HANDSHAKE_SENDER",
            .category = LIFECYCLE,
            .kind = kind::Flag{},
            .description = "Whether this end writes first: the sender repeats its write until the reply lands in its "
                           "own scratch, and the receiver waits for the sender's write and then replies once. The two "
                           "ends of a link differ.",
        },
        RouterField{
            .key = "erisc_sync",
            .source = "MULTI_RISC_TEARDOWN_SYNC_STREAM_ID",
            .category = LIFECYCLE,
            .kind = stream<uint32_t>(StreamRegister::REMOTE_SRC),
            .description = "The register the router's ERISCs sync through at startup and teardown, led by ERISC0. "
                           "Unused when the router runs one ERISC.",
        },
        RouterField{
            .key = "retrain_sync",
            .source = "ETH_RETRAIN_LINK_SYNC_STREAM_ID",
            .category = LIFECYCLE,
            .kind = stream<CoordinatedEriscContextSwitchState>(StreamRegister::REMOTE_SRC),
            .description = "The register the router's ERISCs sync through to retrain the link, led by the ERISC that "
                           "context switches.",
        },

        // Flow control
        RouterField{
            .key = "vc2_receiver_free_slots",
            .source = "VC2_RECEIVER_FREE_SLOTS_STREAM_ID",
            .category = FLOW_CONTROL,
            .kind = stream<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The register reserved for the VC2 receiver's free slots, or the unused id when the router "
                           "has no VC2. The router kernel declares it but does not read it.",
        },

        // Kernel parameters. Under DEBUG_PRINT_ENABLED the kernel ignores the context switch interval and the
        // handshake context switch timeout, and uses its own.
        RouterField{
            .key = "my_direction",
            .source = "MY_DIRECTION",
            .category = KERNEL_PARAMS,
            .kind = enum_kind<eth_chan_directions>(),
            .description = "The direction the router faces, which the kernel picks its downstream channels and "
                           "free-slot registers by. The router's key uses ControlPlane's direction.",
        },
        RouterField{
            .key = "wait_for_host_signal",
            .source = "WAIT_FOR_HOST_SIGNAL",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether the router takes part in the chip's startup sync through local_sync. Without it, "
                           "the kernel ignores its local sync arguments.",
        },
        RouterField{
            .key = "idle_context_switching",
            .source = "IDLE_CONTEXT_SWITCHING",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether the router context switches only once idle, rather than every interval.",
        },
        RouterField{
            .key = "context_switch_interval",
            .source = "SWITCH_INTERVAL",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "How many main loop iterations go by between context switches to base firmware; with "
                           "idle_context_switching, how many in a row without progress.",
        },
        RouterField{
            .key = "handshake_context_switch_timeout",
            .source = "DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "How many writes the handshake sender makes before it gives base firmware a turn to "
                           "route, and then starts counting again.",
        },
        RouterField{
            .key = "txq_spin_wait_send_data",
            .source = "ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether a data send spins until the Ethernet TX queue is free, instead of skipping the "
                           "send while it is busy.",
        },
        RouterField{
            .key = "txq_spin_wait_completion_ack",
            .source = "ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "The same, for a receiver's completion ack.",
        },
        RouterField{
            .key = "txq_accept_ahead",
            .source = "DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "How many packets each Ethernet TX queue accepts before the earlier ones are sent. "
                           "Written to both queues' ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD register at startup.",
        },
        RouterField{
            .key = "risc_cpu_data_cache",
            .source = "ENABLE_RISC_CPU_DATA_CACHE",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether the kernel treats the RISC's data cache as on, and so invalidates it before "
                           "reading what other cores write to its L1.",
        },
        RouterField{
            .key = "speedy_vc0",
            .source = "ENABLE_SPEEDY_VC0",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether VC0 runs the speedy path: worker-only steps that pass credits in batches, so a "
                           "credit count can lag the packet count by up to a batch.",
        },
        RouterField{
            .key = "sender_credit_amortization",
            .source = "SENDER_CREDIT_AMORTIZATION_FREQUENCY",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "Speedy VC0 only: the sender reads completions once this many packets are outstanding, "
                           "and returns credits to its producer once this many have completed.",
        },
        RouterField{
            .key = "receiver_credit_amortization",
            .source = "RECEIVER_CREDIT_AMORTIZATION_FREQUENCY",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "Speedy VC0 only: the receiver acks completions to the peer's sender in batches of at "
                           "least this many packets.",
        },
    };
}();

inline constexpr auto k_erisc_fields = [] {
    using enum FieldCategory;
    return std::array{
        RouterField{
            .key = "local_handshake_master",
            .source = "IS_LOCAL_HANDSHAKE_MASTER",
            .category = LIFECYCLE,
            .kind = kind::Flag{},
            .description = "Whether this ERISC leads the chip's startup sync (the chip's local_sync). The kernel reads "
                           "it only with wait_for_host_signal.",
        },
        RouterField{
            .key = "handshake_enabled",
            .source = "ENABLE_ETHERNET_HANDSHAKE",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether this ERISC runs the Ethernet handshake with the peer at startup.",
        },
        RouterField{
            .key = "context_switch_enabled",
            .source = "ENABLE_CONTEXT_SWITCH",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether this ERISC hands control back to base firmware.",
        },
        RouterField{
            .key = "interrupts_enabled",
            .source = "ENABLE_INTERRUPTS",
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether this ERISC is configured with interrupts on. The router kernel declares it but "
                           "does not read it.",
        },
        RouterField{
            .key = "teardown_check_iterations",
            .source = "ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "How many passes over its channel steps the main loop makes between its checks for "
                           "termination and context switching.",
        },
        RouterField{
            .key = "telemetry_stats_mask",
            .source = "FABRIC_TELEMETRY_STATS_MASK",
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "The FabricTelemetryStatistic bits this ERISC collects; 0 when its telemetry is off.",
        },
    };
}();

// The chip pass reads this sender field to find the channels a sibling must have an edge into.
inline constexpr std::string_view k_static_connection_key = "static_connection";

inline constexpr auto k_sender_channel_fields = [] {
    using enum FieldCategory;
    return std::array{
        RouterField{
            .key = "is_injection_channel",
            .source = NamedArg{"SENDER_CH_{}_IS_INJECTION", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .kind = kind::Flag{},
            .description = "Whether the channel injects traffic into the fabric. With deadlock avoidance on, an "
                           "injection channel uses bubble flow control.",
        },
        RouterField{
            .key = "free_slots",
            .source = NamedArg{"SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", ArgIndex::FABRIC_POSITION},
            .category = FLOW_CONTROL,
            .kind = stream<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The channel's free slots: the producer takes one per packet it writes, and the router "
                           "returns it once the packet completes.",
        },
        RouterField{
            .key = "producer_credit_return_noc",
            .source = NamedArg{"SENDER_CH_{}_ACK_NOC_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .kind = kind::Number{},
            .description = "The NoC the router returns the producer's credits on.",
        },
        RouterField{
            .key = "producer_credit_return_cmd_buf",
            .source = NamedArg{"SENDER_CH_{}_ACK_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .kind = enum_kind<NocCmdBuf>(),
            .description = "The NoC command buffer it returns them through.",
        },
        RouterField{
            .key = "connection",
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                return static_cast<uint32_t>(builder.sender_channels_connection_semaphore_id[index.compact]);
            }},
            .category = CONTROL_INFO,
            .kind = l1_value<uint32_t>(),
            .description = "Where the producer opens and closes its connection. In the router's runtime args.",
        },
        RouterField{
            .key = "conn_info",
            .source = NamedArg{"LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", ArgIndex::COMPACT},
            .category = CONTROL_INFO,
            .kind = l1_value<EDMChannelWorkerLocationInfo>(),
            .description = "Where the producer writes where it is, and the router its read counter.",
        },
        RouterField{
            .key = "buffer_index_sem",
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                return static_cast<uint32_t>(builder.sender_channels_buffer_index_semaphore_id[index.compact]);
            }},
            .category = CONTROL_INFO,
            .kind = l1_value<SenderChannelProducerCursor>(),
            .description = "Where the producer keeps its write cursor across connections. In the producer's runtime "
                           "args, from the channel's connection spec.",
        },
        RouterField{
            .key = k_static_connection_key,
            .source = NamedArg{"SENDER_CH_{}_WAIT_STATIC_CONNECTION", ArgIndex::COMPACT},
            .category = CONTROL_INFO,
            .kind = kind::Flag{},
            .description = "Whether a sibling router connects to the channel once, at startup: before its main loop, "
                           "the router waits for that connection.",
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
            .kind = kind::Flag{},
            .description = "Whether the receiver's step skips handing its packets on, to siblings or the local chip. "
                           "Set for the speedy VC0 and VC2 steps, which hand packets on themselves, and when a channel "
                           "trimming capture saw the channel do neither.",
        },
        RouterField{
            .key = "intermesh_ingress",
            .source = NamedArg{"IS_RECEIVER_CHANNEL_{}_INTERMESH_INGRESS", ArgIndex::COMPACT, Emitted::FABRIC_2D},
            .category = KERNEL_PARAMS,
            .kind = kind::Flag{},
            .description = "Whether the receiver's packets arrive from another mesh: the kernel re-encodes their "
                           "route from this mesh's route table before deciding where they go.",
        },
        // A NoC is a number, not an enum: NOC's RISCV_*_default aliases share its values, so they have no unique name.
        RouterField{
            .key = "forward_noc",
            .source = NamedArg{"RX_CH_{}_FWD_NOC_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "The NoC the receiver forwards packets to sibling routers on.",
        },
        RouterField{
            .key = "forward_data_cmd_buf",
            .source = NamedArg{"RX_CH_{}_FWD_DATA_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .kind = enum_kind<NocCmdBuf>(),
            .description = "The NoC command buffer it writes each forwarded packet through.",
        },
        RouterField{
            .key = "forward_sync_cmd_buf",
            .source = NamedArg{"RX_CH_{}_FWD_SYNC_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .kind = enum_kind<NocCmdBuf>(),
            .description = "The NoC command buffer it sends the credit increment that announces a forwarded packet "
                           "through.",
        },
        RouterField{
            .key = "local_write_noc",
            .source = NamedArg{"RX_CH_{}_LOCAL_WRITE_NOC_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .kind = kind::Number{},
            .description = "The NoC the receiver delivers packets to the local chip on.",
        },
        RouterField{
            .key = "local_write_cmd_buf",
            .source = NamedArg{"RX_CH_{}_LOCAL_WRITE_CMD_BUF_ID", ArgIndex::COMPACT},
            .category = KERNEL_PARAMS,
            .kind = enum_kind<NocCmdBuf>(),
            .description = "The NoC command buffer it delivers them through.",
        },
        RouterField{
            .key = "pkts_sent",
            .source = NamedArg{"TO_RECEIVER_{}_PKTS_SENT_ID", ArgIndex::COMPACT},
            .category = FLOW_CONTROL,
            .kind = stream<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The packets the peer's sender has sent the channel that the receiver has not taken yet: "
                           "the peer adds one per packet it sends over Ethernet.",
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
            .kind = stream<uint32_t>(StreamRegister::BUF_SPACE_AVAILABLE),
            .description = "The router's view of the free slots in the sender channel the edge lands on: the router "
                           "takes one per packet it forwards, and the sibling returns it once the packet completes.",
        },
        RouterField{
            .key = "teardown_sem",
            // get_runtime_args pushes -1 for a semaphore the builder has not set.
            .source = BuilderMember{[](const FabricEriscDatamoverBuilder& builder, const FieldIndex& index) {
                const auto& ids = builder.receiver_channels_downstream_teardown_semaphore_id;
                return static_cast<uint32_t>(index.teardown < ids.size() ? ids[index.teardown].value_or(-1) : -1);
            }},
            .category = CONTROL_INFO,
            .kind = l1_value<uint32_t>(),
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
static_assert(sources_name_channels(k_erisc_fields, false));
static_assert(sources_name_channels(k_sender_channel_fields, true));
static_assert(sources_name_channels(k_receiver_channel_fields, true));
static_assert(sources_name_channels(k_downstream_edge_fields, true));

}  // namespace tt::tt_fabric::manifest
