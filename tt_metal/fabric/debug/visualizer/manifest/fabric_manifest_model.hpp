// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-umd/types/arch.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_build_identity.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

// The fabric manifest's model. The router pass fills in what the router builders know. The topology pass resolves each
// router's channels against ControlPlane, the cluster and the chip's other routers, and builds the meshes. The
// describe file fills the run, fabric and arch, and the writer turns the whole model into JSON.
namespace tt::tt_fabric::manifest {

// What a multi-sampled region's change across capture's rounds says about its router. Capture reads a multi-sampled
// region once per round, as well as once in its full read.
enum class MultiSampleRole : uint8_t {
    // Written every pass of the router's loop, stuck or not: it moving says the loop runs.
    LOOP_HEARTBEAT,
    // The fabric telemetry area.
    FABRIC_TELEMETRY,
};

// What a field or a typed member holds.
namespace content {

// A region of L1 memory.
struct L1 {
    // The first byte the region starts at.
    uint32_t address = 0;
    // The number of elements with type `type` in the region.
    uint32_t num_elements = 1;
    // The number of bytes each element takes up.
    uint32_t element_size = 0;
    // The type's registered name (fabric_manifest_type_registry.hpp).
    std::string type;
    // Null: capture reads the region once, in its full read. Otherwise it reads it in every round too.
    std::optional<MultiSampleRole> multi_sampled;
    // False when the run's options mean nothing writes the region, though it's allocated: its bytes are whatever the
    // L1 held.
    bool active = true;
};

// A stream register's word. Its value is the word's low `value_bits` bits, read as `type`; the bits above can hold
// other state.
struct StreamReg {
    uint32_t stream_id = 0;
    uint32_t address = 0;
    // The number of bits the word's value takes up (useful if the desired value in a stream register is only a certain
    // number of bits).
    uint32_t value_bits = 0;
    // The type's registered name (fabric_manifest_type_registry.hpp).
    std::string type;
    // Null: capture reads the register once, in its full read. Otherwise it reads it in every round too.
    std::optional<MultiSampleRole> multi_sampled;
};

struct Number {
    uint32_t value = 0;
};

struct Flag {
    bool value = false;
};

// A value of a configuration enum the kernel is given, such as a direction. No captured bytes hold it, so its type
// isn't registered: the manifest writes the value and the name of its enumerator.
struct Enum {
    uint32_t value = 0;
    // The enumerator `value` names, in lower case (lower_name), e.g. "east".
    std::string value_name;
};

}  // namespace content

using Content = std::variant<content::L1, content::StreamReg, content::Number, content::Flag, content::Enum>;

// Registers an L1 region of `num_elements` Ts, `element_size` bytes apart from `address`, under T's registered name.
template <typename T>
content::L1 register_l1_region(uint32_t address, uint32_t num_elements = 1, uint32_t element_size = sizeof(T)) {
    return {
        .address = address,
        .num_elements = num_elements,
        .element_size = element_size,
        .type = std::string(get_manifest_type_name<T>())};
}

// One of a router's L1 credit counter arrays (L1CreditCounters).
enum class CreditCounterArray : uint8_t {
    TO_SENDER_ACK,
    TO_SENDER_COMPLETION,
    RECEIVER_ACK,
    RECEIVER_COMPLETION,
};

// What carries a VC's sender credits on a mesh.
enum class CreditBacking : uint8_t {
    STREAM_REGISTER,
    L1_COUNTER,
};

// Whose sender channel each element of a credit counter array belongs to. Elements are indexed by sender channel,
// numbered densely across the router's VCs (its compact index).
enum class CreditCounterIndexedBy : uint8_t {
    // This router's sender channels (the to_sender arrays).
    OWN_SENDER_CHANNEL,
    // The sender channels of the router on the other end of the link (the receiver arrays).
    PEER_SENDER_CHANNEL,
};

// One element of a credit counter array on the same router.
struct CounterRef {
    CreditCounterArray array = CreditCounterArray::TO_SENDER_ACK;
    uint32_t index = 0;
};

// A router on the same chip and routing plane, named by the direction it faces.
struct SiblingRouterRef {
    eth_chan_directions direction = eth_chan_directions::EAST;
};

// What names a router on its chip, and makes its key in the chip's routers, such as "E0" (router_key).
struct RouterKey {
    // The direction the builder built the router facing.
    eth_chan_directions direction = eth_chan_directions::EAST;
    routing_plane_id_t routing_plane = 0;
};

// A router on any chip, named by its chip and its key there.
struct RouterRef {
    FabricNodeId node{MeshId{0}, 0};
    RouterKey key;
};

// The chip's local worker, as the producer of a sender channel.
struct LocalWorker {};

// The router's tensix mux, which in mux mode feeds the worker channel with the worker's and the siblings' VC0
// traffic.
struct LocalTensixMux {};

// Who writes into a sender channel.
using SenderChannelProducer = std::variant<LocalWorker, LocalTensixMux, SiblingRouterRef>;

// Why a channel's step runs or not.
enum class ChannelStatus : uint32_t {
    // An ERISC runs the channel's step.
    ACTIVE,
    // The kernel does not run the channel's VC on this router (no FABRIC_2D_VC<n>_SERVICED).
    VC_NOT_SERVICED,
    // A VC0 sender other than the worker channel, in mux mode. The tensix mux carries its traffic into the worker
    // channel, and the channel has no buffer slots.
    MUX,
    // The applied channel trimming profile turned the channel off.
    TRIMMED,
};

// A router's L1 credit counter arrays, shared by every VC that uses counter credits. k_credit_counter_arrays says whose
// sender channel indexes each. Active when any of the router's ERISCs is fed VC<n>_USES_COUNTER_CREDITS.
struct L1CreditCounters {
    content::L1 to_sender_ack;
    content::L1 to_sender_completion;
    content::L1 receiver_ack;
    content::L1 receiver_completion;
};

// One credit counter array: where L1CreditCounters holds it, and whose sender channel each element belongs to.
struct CreditCounterArrayLayout {
    CreditCounterArray array;
    content::L1 L1CreditCounters::* member;
    CreditCounterIndexedBy indexed_by;
};

// Every credit counter array, in CreditCounterArray's order. The to_sender arrays are indexed by this router's sender
// compact index, and the receiver arrays by the peer router's (the receiver counts credits for the peer's sender
// channels).
inline constexpr std::array<CreditCounterArrayLayout, 4> k_credit_counter_arrays{{
    {CreditCounterArray::TO_SENDER_ACK, &L1CreditCounters::to_sender_ack, CreditCounterIndexedBy::OWN_SENDER_CHANNEL},
    {CreditCounterArray::TO_SENDER_COMPLETION,
     &L1CreditCounters::to_sender_completion,
     CreditCounterIndexedBy::OWN_SENDER_CHANNEL},
    {CreditCounterArray::RECEIVER_ACK, &L1CreditCounters::receiver_ack, CreditCounterIndexedBy::PEER_SENDER_CHANNEL},
    {CreditCounterArray::RECEIVER_COMPLETION,
     &L1CreditCounters::receiver_completion,
     CreditCounterIndexedBy::PEER_SENDER_CHANNEL},
}};

static_assert(
    k_credit_counter_arrays.size() == enchantum::count<CreditCounterArray>,
    "k_credit_counter_arrays lists every CreditCounterArray");
static_assert(
    [] {
        for (size_t index = 0; index < k_credit_counter_arrays.size(); ++index) {
            if (k_credit_counter_arrays[index].array != static_cast<CreditCounterArray>(index)) {
                return false;
            }
        }
        return true;
    }(),
    "k_credit_counter_arrays lists each CreditCounterArray at its own value");

// The region `array` names in `counters`, such as the region a CounterRef's element is in.
inline const content::L1& credit_counter_region(const L1CreditCounters& counters, CreditCounterArray array) {
    return counters.*k_credit_counter_arrays.at(static_cast<size_t>(array)).member;
}

// Whose sender channel each element of `array` belongs to.
inline CreditCounterIndexedBy credit_counter_indexed_by(CreditCounterArray array) {
    return k_credit_counter_arrays.at(static_cast<size_t>(array)).indexed_by;
}

// High level information about a particular router.
struct RouterIdentity {
    uint32_t eth_chan = 0;
    RouterKey key;
    tt::tt_metal::CoreCoord logical_core;
    tt::tt_metal::CoreCoord virtual_core;
};

// Ethernet link information about a particular router.
struct EthLink {
    EdgeCapability edge_capability = EdgeCapability::INTRAMESH_CARDINAL;
    bool is_dispatch_link = false;
    // Null when ControlPlane connects the channel to nothing, or to a channel the peer's builder built no router on.
    std::optional<RouterRef> peer;
    bool cross_host = false;
    bool wrap = false;
};

// Information about the "shape" of a router, i.e. the number of VCs, senders, receivers, and active ERISCs.
struct RouterShape {
    uint32_t num_vcs = 0;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> senders_per_vc = {};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receivers_per_vc = {};
    uint32_t num_active_eriscs = 0;
    bool channel_trimming_overrides_applied = false;
    bool has_tensix_extension = false;
};

// The group a field belongs to. Mainly for logical grouping of information in decode / visualizer.
enum class FieldCategory : uint8_t {
    LIFECYCLE,
    KERNEL_PARAMS,
    FLOW_CONTROL,
    CONTROL_INFO,
    DIAGNOSTICS,
};

// A fact the router kernel is fed, read through the field tables (fabric_manifest_fields.hpp).
struct Field {
    std::string key;
    // Tag used for logical grouping of information in decode / visualizer.
    FieldCategory category = FieldCategory::LIFECYCLE;
    // Underlying content that the value the kernel is fed points to. Null when the field is a buffer or stream
    // register the builder did not allocate, which the kernel is fed as address 0 or k_unused_stream_id.
    std::optional<Content> content;
};

// The credits a sender channel receives back from the peer's receiver, as one ERISC is told to read them.
struct SenderChannelCredits {
    // Only on VC0, on an ERISC with bubble flow control.
    std::optional<CounterRef> acked;
    CounterRef completed;
};

// A field table's fields as each ERISC's arguments give them. Indexed by ERISC id.
using FieldsByErisc = std::vector<std::vector<Field>>;

// A sender channel that takes packets from its producer and sends them over Ethernet to the peer's receiver.
struct SenderChannel {
    ChannelStatus status = ChannelStatus::ACTIVE;
    // IDs of the ERISCs that run the channel's step.
    std::vector<uint32_t> serviced_by;
    // Null when nothing feeds the channel.
    std::optional<SenderChannelProducer> producer;
    // Null when the channel has no slots.
    std::optional<content::L1> ring_buffer;
    // Indexed by ERISC id.
    std::vector<SenderChannelCredits> credits_by_erisc;
    // Arguments the kernel is fed for the channel, read through the field table.
    FieldsByErisc fields_by_erisc;
};

// A receiver channel that takes packets from the peer's senders over Ethernet, delivers them locally, and forwards
// them to sibling routers' senders.
struct ReceiverChannel {
    ChannelStatus status = ChannelStatus::ACTIVE;
    // IDs of the ERISCs that run the channel's step.
    std::vector<uint32_t> serviced_by;
    // The VC whose downstream edges each ERISC gives the channel's step, indexed by ERISC id. Null on an ERISC that
    // doesn't run the step, or whose step forwards to no sibling.
    std::vector<std::optional<uint32_t>> forwards_on_by_erisc;
    // Null when the channel has no slots.
    std::optional<content::L1> ring_buffer;
    // Arguments the kernel is fed for the channel, read through the field table.
    FieldsByErisc fields_by_erisc;
};

// A router's channels.
struct RouterChannels {
    // Indexed [vc][channel], over RouterShape's counts.
    std::vector<std::vector<SenderChannel>> senders;
    std::vector<std::vector<ReceiverChannel>> receivers;
};

// The sender channel a downstream edge lands on: a sibling's, on the edge's VC.
struct DownstreamChannel {
    SiblingRouterRef target;
    uint32_t landing_channel = 0;
};

// A persistent connection on one VC from this router to a sender channel of a router on the same chip and routing
// plane. The receivers forwarding on the VC write through it. In mux mode a VC0 edge into a sibling ends at the
// sibling's tensix mux instead.
struct DownstreamEdge {
    // The kernel's EDGE_<n>: the sibling's compact index among this router's other directions, plus one.
    uint32_t edge = 0;
    DownstreamChannel downstream_channel;
    bool through_tensix_mux = false;
    FieldsByErisc fields_by_erisc;
};

// The VCs the kernel has a downstream array for, VC0 and VC1 (VC<v>_DOWNSTREAM_EDM_SIZE): the others never forward to
// a sibling.
inline constexpr uint32_t k_num_vcs_with_downstream_array = 2;

// One 2D router's live downstreams on each VC with a downstream array: the directions it forwards packets to, one
// Routing2DCodec action bit per direction. The router ignores every other bit of a packet's action.
using LiveDownstreamMasks = std::array<uint8_t, k_num_vcs_with_downstream_array>;

// One RISC the router's kernel runs on.
struct Erisc {
    // The processor the kernel is created on, among the ones the HAL defines for the ERISC mode.
    tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
    // The ERISC on the Ethernet core that the processor runs on, which indexes CoreRegisters::reset_masks.
    uint32_t physical_erisc = 0;
    // The level the builder compiled and linked the kernel at (get_kernel_opt_level).
    tt::tt_metal::KernelBuildOptLevel opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
};

// Information about a router.
struct Router {
    RouterIdentity identity;
    EthLink link;
    RouterShape shape;
    // Always reserved, although not always used. Whether a VC uses them is its mesh's credit_transport backing.
    L1CreditCounters credit_counters;
    RouterChannels channels;
    // Indexed [vc], by edge.
    std::vector<std::vector<DownstreamEdge>> intra_chip_downstream_edges;
    // Null in 1D, where a packet's route says only whether to forward, to the router's one downstream.
    std::optional<LiveDownstreamMasks> live_downstreams;
    // The router's fields passed to the kernel, per ERISC.
    FieldsByErisc fields_by_erisc;
    // Indexed by ERISC id.
    std::vector<Erisc> eriscs;
    // The L1 past the channel buffers. Null when the router has no leftover L1. Never active: nothing is placed there.
    std::optional<content::L1> leftover_l1;
};

// How the chip's routers sync at startup through their local_sync words, as the router kernels are fed it
// (KernelCreationContext).
struct LocalSync {
    // The router that leads the sync.
    std::optional<RouterRef> master;
    uint32_t num_routers = 0;
    // Bit N is set for the router on Ethernet channel N.
    uint32_t router_channels_mask = 0;
};

// Information about a chip. Every chip in the mesh graph is listed, local or not; a chip this rank can't map to a
// device is remote, and has only its identity.
struct Chip {
    // Mesh-local id for the chip.
    FabricNodeId fabric_node_id{MeshId{0}, 0};
    // The chip's coordinates in its mesh.
    std::vector<uint32_t> mesh_coord;
    // Which host rank of the chip's mesh owns it
    std::optional<uint32_t> host_rank;
    // Process-local id for the chip. Null for a remote chip.
    std::optional<ChipId> physical_chip_id;
    // Hardware id for the chip. Null for a remote chip.
    std::optional<uint64_t> asic_id;
    ZPortRole z_port_role = ZPortRole::NONE;
    // Null when the chip has no routers.
    std::optional<LocalSync> local_sync;
    std::vector<Router> routers;
};

// What carries one VC's sender credits on a mesh.
struct VcCreditTransport {
    uint32_t vc = 0;
    CreditBacking backing = CreditBacking::STREAM_REGISTER;
};

// Which axes of a 2D mesh wrap around.
struct MeshTorus {
    bool y = false;
    bool x = false;
};

// A mesh in the mesh graph, and its chips.
struct Mesh {
    MeshId mesh_id{0};
    std::vector<uint32_t> shape;
    // 2D meshes only.
    std::optional<MeshTorus> torus;
    bool express_routing = false;
    // The axis the express links run along (the router's FABRIC_EXPRESS_AXIS define). Null when express routing is off.
    std::optional<Fabric2DAxis> express_axis;
    // Each VC's credit transport, for meshes on this host only. VCs no router has senders on are left out.
    std::optional<std::vector<VcCreditTransport>> credit_transport;
    std::vector<Chip> chips;
};

// An L1 area on every router core that the architecture fixes, rather than the builder allocates. The manifest keys
// each in the arch block's l1_areas by its lower_enum_name, such as "routing_table". Adding one here adds it to the
// writer, the reader and for_each_arch_area; describe_arch_area's switch fails the build until it's described. Every
// router core writes each of them, so an arch area is always active: a region only some routers write is described on
// the router.
enum class ArchArea : uint8_t {
    // The word the router kernel writes its heartbeat to (HeartbeatProtocol). Multi-sampled as LOOP_HEARTBEAT.
    HEARTBEAT,
    // Multi-sampled as FABRIC_TELEMETRY.
    FABRIC_TELEMETRY,
    ROUTING_TABLE,
    // Only on architectures with an Ethernet firmware mailbox.
    ETH_FW_MAILBOX,
    // Base firmware's link status words, where the HAL gives an address for them (not on Wormhole).
    BASE_FW_PORT_STATUS,
    BASE_FW_TRAIN_STATUS,
    BASE_FW_RETRAIN_COUNT,
    BASE_FW_RX_LINK_UP,
    BASE_FW_POSTCODE,
};

static_assert(
    [] {
        for (size_t index = 0; index < enchantum::count<ArchArea>; ++index) {
            if (enchantum::values<ArchArea>[index] != static_cast<ArchArea>(index)) {
                return false;
            }
        }
        return true;
    }(),
    "ArchArea's enumerators are 0, 1, 2, ..., since ArchAreas indexes its regions by them");

// The region of each ArchArea on a router core. An area is null when the architecture doesn't have it.
struct ArchAreas {
    // Indexed by ArchArea.
    std::array<std::optional<content::L1>, enchantum::count<ArchArea>> regions;

    std::optional<content::L1>& get(ArchArea area) { return regions.at(static_cast<size_t>(area)); }
    const std::optional<content::L1>& get(ArchArea area) const { return regions.at(static_cast<size_t>(area)); }
};

// How the router kernel writes its heartbeat word (ArchArea::HEARTBEAT): magic | its 16-bit iteration count, every
// period_iters main-loop iterations.
struct HeartbeatProtocol {
    uint32_t magic = 0;
    uint32_t magic_mask = 0;
    uint32_t period_iters = 0;
};

// The addresses of a router core's soft reset and wall clock registers.
struct CoreRegisters {
    uint32_t soft_reset = 0;
    // Indexed by physical ERISC: the soft_reset bit that holds it in reset.
    std::vector<uint32_t> reset_masks;
    // Reading the low word latches the high word, so the low word is read first.
    uint32_t wall_clock_low = 0;
    uint32_t wall_clock_high = 0;
};

// One core register: its name in the manifest, and its member of CoreRegisters.
struct CoreRegisterLayout {
    std::string_view name;
    uint32_t CoreRegisters::* address;
};

// The core registers in the order they're read, which is the order the manifest lists them in.
inline constexpr std::array<CoreRegisterLayout, 3> k_core_registers{{
    {"soft_reset", &CoreRegisters::soft_reset},
    {"wall_clock_low", &CoreRegisters::wall_clock_low},
    {"wall_clock_high", &CoreRegisters::wall_clock_high},
}};

// The whole L1 of the Ethernet core a router runs on: size_bytes bytes from address.
struct EthCoreL1 {
    uint32_t address = 0;
    uint32_t size_bytes = 0;
};

// What the manifest describes of the architecture the routers run on.
struct Arch {
    tt::ARCH arch = tt::ARCH::Invalid;
    EthCoreL1 l1;
    ArchAreas areas;
    HeartbeatProtocol heartbeat_protocol;
    CoreRegisters core_registers;
    // FABRIC_TELEMETRY_VERSION: which layout of FabricTelemetry the fabric_telemetry area holds.
    uint32_t fabric_telemetry_version = 0;
};

// ============ The whole manifest ============

// A tt-run sub-context: one fabric of a job tt-run split into several, each with its own MPI communicator.
struct Subcontext {
    uint32_t id = 0;
    // This process's rank within the sub-context, and the sub-context's number of ranks.
    uint32_t rank = 0;
    uint32_t size = 1;
};

// This process and the build it runs.
struct RunInfo {
    // The fabric run this manifest belongs to: made by rank 0 at fabric init and shared by every rank of the fabric. It
    // names the run's folder, such as "20261105T221500Z-1a2b3c4d".
    std::string run_id;
    // This process's rank in the whole job, and the job's number of ranks.
    uint32_t mpi_rank = 0;
    uint32_t world_size = 1;
    // Null when the job isn't split into sub-contexts.
    std::optional<Subcontext> subcontext;
    // The host this process runs on, as the physical system descriptor names it.
    std::string host_name;
    // When the manifest was written, in UTC, as ISO 8601.
    std::string written_at;
    visualizer::BuildIdentity build;
};

// The settings fabric was set up with: SetFabricConfig's arguments.
struct FabricSettings {
    FabricConfig fabric_config = FabricConfig::DISABLED;
    FabricReliabilityMode reliability_mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    // The most routing planes fabric opens per link. Null when it opens all of them.
    std::optional<uint8_t> num_routing_planes;
    FabricTensixConfig tensix_config = FabricTensixConfig::DISABLED;
    FabricUDMMode udm_mode = FabricUDMMode::DISABLED;
    FabricManagerMode fabric_manager = FabricManagerMode::DEFAULT;
    FabricRouterConfig router_config;
};

// The fabric's inter-mesh VC setup (FabricBuilderContext::get_intermesh_vc_config()), which decides whether routers
// have VC1 and VC2.
struct IntermeshVc {
    IntermeshVCMode mode = IntermeshVCMode::DISABLED;
    bool requires_vc1 = false;
    bool requires_vc2 = false;
    // TT_METAL_FABRIC_DISABLE_INTERMESH_VC: inter-mesh traffic lands on VC0 instead of VC1.
    bool intermesh_vc_disabled = false;
};

// One range of router L1 that fabric init zeroes before launch.
struct HostClear {
    uint32_t address = 0;
    uint32_t size_bytes = 0;
};

// This run's fabric: the settings it was set up with, and what FabricContext derived from them.
struct FabricInfo {
    FabricSettings config;
    Topology topology = Topology::Linear;
    bool is_2d_routing = false;
    // The registered name of the packet header type the run's routers use (packet_header_types.hpp).
    std::string packet_header;
    // The bytes a packet's header takes: the type's size, plus the UDMControlFields that follow it in UDM mode.
    size_t packet_header_size_bytes = 0;
    size_t max_payload_size_bytes = 0;
    size_t channel_buffer_size_bytes = 0;
    // 2D routing only.
    std::optional<uint32_t> routing_2d_route_buffer_size;
    // 1D routing only.
    std::optional<uint32_t> routing_1d_extension_words;
    bool multi_txq = false;
    // What fabric init zeroes in every router's L1 before launch (configure_fabric_cores).
    std::vector<HostClear> host_clears;
    IntermeshVc intermesh_vc;
};

// Everything the sources read for one rank's manifest, one member per block they fill. The writer turns it into the
// JSON, and derives the types and vocabulary blocks from it.
struct ManifestFacts {
    RunInfo run;
    FabricInfo fabric;
    Arch arch;
    std::vector<Mesh> meshes;
};

// The facts' mesh, or null when the manifest has none by that id.
inline const Mesh* find_mesh(const ManifestFacts& facts, MeshId mesh_id) {
    const auto mesh = std::ranges::find(facts.meshes, mesh_id, &Mesh::mesh_id);
    return mesh == facts.meshes.end() ? nullptr : &*mesh;
}

// The facts' chip, or null when the manifest has none by that id.
inline const Chip* find_chip(const ManifestFacts& facts, FabricNodeId node) {
    const Mesh* mesh = find_mesh(facts, node.mesh_id);
    if (mesh == nullptr) {
        return nullptr;
    }
    const auto chip = std::ranges::find(mesh->chips, node, &Chip::fabric_node_id);
    return chip == mesh->chips.end() ? nullptr : &*chip;
}

// ============ Vocabulary ============

// One vocabulary list: its name, and the type whose names it holds (an enum, or Content for the field kinds).
// The vocabulary block consists of several lists, each named by name.
template <typename T>
struct VocabularyList {
    using type = T;
    std::string_view name;
};

// The lists in the manifest's vocabulary block. A label is a key whose value is one name from a fixed set, such as a
// channel's "status"; its list names every value it can hold (channel_status lists every ChannelStatus).
inline constexpr std::tuple k_vocabulary_lists{
    VocabularyList<tt::ARCH>{"arch"},
    VocabularyList<FabricConfig>{"fabric_config"},
    VocabularyList<FabricReliabilityMode>{"reliability_mode"},
    VocabularyList<FabricTensixConfig>{"tensix_config"},
    VocabularyList<FabricUDMMode>{"udm_mode"},
    VocabularyList<FabricManagerMode>{"fabric_manager_mode"},
    VocabularyList<Topology>{"fabric_topology"},
    VocabularyList<CreditBacking>{"credit_backing"},
    VocabularyList<ZPortRole>{"z_port_role"},
    VocabularyList<EdgeCapability>{"edge_capability"},
    VocabularyList<CreditCounterIndexedBy>{"credit_counter_indexed_by"},
    VocabularyList<FieldCategory>{"field_category"},
    VocabularyList<Content>{"field_kind"},
    VocabularyList<ChannelStatus>{"channel_status"},
    VocabularyList<tt::tt_metal::DataMovementProcessor>{"erisc_processor"},
    VocabularyList<tt::tt_metal::KernelBuildOptLevel>{"kernel_opt_level"},
    VocabularyList<IntermeshVCMode>{"intermesh_vc_mode"},
    VocabularyList<MultiSampleRole>{"multi_sample_role"},
};

}  // namespace tt::tt_fabric::manifest
