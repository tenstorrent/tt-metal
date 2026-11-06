// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt_stl/assert.hpp>
#include <tt-umd/arch/architecture_implementation.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/packet_header_types.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_build_identity.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_fabric::manifest {

namespace {

using tt::tt_metal::FWMailboxMsg;
using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalProgrammableCoreType;

// The time now in UTC, as ISO 8601.
std::string get_utc_now_iso8601() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

// The most routing planes fabric opens per link, or nullopt when it opens all the available ones.
std::optional<uint8_t> get_num_routing_planes() {
    const uint8_t num_routing_planes = tt_metal::MetalEnvAccessor(tt_metal::MetalContext::instance().get_env())
                                           .impl()
                                           .get_num_fabric_active_routing_planes();
    if (num_routing_planes == std::numeric_limits<uint8_t>::max()) {
        return std::nullopt;
    }
    return num_routing_planes;
}

uint32_t get_hal_l1_address(const tt::tt_metal::Hal& hal, HalL1MemAddrType area_id) {
    return static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, area_id));
}

uint32_t get_hal_l1_size(const tt::tt_metal::Hal& hal, HalL1MemAddrType area_id) {
    return hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, area_id);
}

// The whole L1 of the Ethernet core a router runs on: the HAL's BASE area on an active Ethernet core.
EthCoreL1 describe_eth_core_l1(const tt::tt_metal::Hal& hal) {
    return EthCoreL1{
        .address = get_hal_l1_address(hal, HalL1MemAddrType::BASE),
        .size_bytes = get_hal_l1_size(hal, HalL1MemAddrType::BASE),
    };
}

// A HAL L1 area holding a whole number of Ts, as many as the HAL's size holds.
template <typename T>
content::L1 describe_hal_l1_area(const tt::tt_metal::Hal& hal, HalL1MemAddrType area_id) {
    const uint32_t area_size = get_hal_l1_size(hal, area_id);
    TT_FATAL(
        area_size >= sizeof(T) && area_size % sizeof(T) == 0,
        "Fabric manifest: the HAL's {} area is {} bytes, which is not a whole number of its {}-byte {}",
        lower_enum_name(area_id),
        area_size,
        sizeof(T),
        get_manifest_type_name<T>());
    return register_l1_region<T>(get_hal_l1_address(hal, area_id), area_size / sizeof(T));
}

// A base firmware word, whose address the HAL keeps among its Ethernet firmware mailbox values; null when it is 0.
std::optional<content::L1> describe_base_fw_word(const tt::tt_metal::Hal& hal, tt::tt_metal::FWMailboxMsg word) {
    const uint32_t address = hal.get_eth_fw_mailbox_val(word);
    if (address == 0) {
        return std::nullopt;
    }
    return register_l1_region<uint32_t>(address);
}

// The kernel picks its heartbeat word's address by architecture: Blackhole's, or else Wormhole's.
Heartbeat describe_heartbeat(tt::ARCH arch) {
    return Heartbeat{
        .word = register_l1_region<uint32_t>(
            arch == tt::ARCH::BLACKHOLE ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE
                                        : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE),
        .magic = FABRIC_KERNEL_HEARTBEAT_MAGIC,
        .magic_mask = FABRIC_KERNEL_HEARTBEAT_MAGIC_MASK,
        .period_iters = FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS,
    };
}

CoreRegisters describe_core_registers(const tt::tt_metal::Hal& hal) {
    const auto arch = hal.get_arch();
    CoreRegisters out{
        .soft_reset = hal.get_risc_soft_reset_addr(),
        .wall_clock_low = hal.get_wall_clock_low_addr(),
        .wall_clock_high = hal.get_wall_clock_high_addr(),
    };
    TT_FATAL(
        out.soft_reset != 0 && out.wall_clock_low != 0 && out.wall_clock_high != 0,
        "Fabric manifest: the HAL gives no soft reset or wall clock address on {}",
        lower_enum_name(arch));
    const auto umd_arch = tt::umd::ArchitectureImplementation::create(arch);
    TT_FATAL(umd_arch != nullptr, "Fabric manifest: UMD has no soft reset bits for {}", lower_enum_name(arch));
    constexpr std::array k_eriscs = {tt::umd::RiscType::ERISC0, tt::umd::RiscType::ERISC1};
    const uint32_t num_eriscs = hal.get_num_physical_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    TT_FATAL(
        num_eriscs > 0 && num_eriscs <= k_eriscs.size(),
        "Fabric manifest: the HAL gives {} ERISCs on {}, but the manifest has soft reset bits for 1 to {}",
        num_eriscs,
        lower_enum_name(arch),
        k_eriscs.size());
    out.reset_masks.reserve(num_eriscs);
    for (uint32_t erisc = 0; erisc < num_eriscs; ++erisc) {
        out.reset_masks.push_back(umd_arch->get_soft_reset_reg_value(k_eriscs[erisc]));
    }
    return out;
}

// The registered name of the packet header type the run's routers use: the one the routing mode and its template
// argument pick.
std::string_view describe_packet_header(const FabricContext& fabric_context) {
    return visit_packet_header_type(
        fabric_context.get_packet_header_config(),
        []<typename Header>(std::type_identity<Header>) { return get_manifest_type_name<Header>(); });
}

// What fabric init zeroes in every router's L1 before launch.
std::vector<HostClear> describe_host_clears(const FabricBuilderContext& builder_context) {
    const auto size_bytes = static_cast<uint32_t>(
        builder_context.get_fabric_router_config().router_buffer_clear_size_words * sizeof(uint32_t));
    std::vector<HostClear> clears;
    for (const size_t address : builder_context.get_fabric_router_addresses_to_clear()) {
        clears.push_back({.address = static_cast<uint32_t>(address), .size_bytes = size_bytes});
    }
    return clears;
}

IntermeshVc describe_intermesh_vc(const FabricBuilderContext& builder_context) {
    const auto& config = builder_context.get_intermesh_vc_config();
    return {
        .mode = config.mode,
        .requires_vc1 = config.requires_vc1,
        .requires_vc2 = config.requires_vc2,
        .intermesh_vc_disabled = config.intermesh_vc_disabled,
    };
}

}  // namespace

RunInfo describe_run(const ControlPlane& control_plane, const std::string& run_id) {
    using tt_metal::distributed::multihost::DistributedContext;
    // The whole job's communicator, not the current world: tt-run can split a job into sub-contexts, and each
    // sub-context's ranks start again from 0.
    const auto world = DistributedContext::get_world_context();
    const auto& subcontext_world = DistributedContext::get_current_world();
    std::optional<Subcontext> subcontext;
    if (const auto subcontext_id = subcontext_world->subcontext_id(); subcontext_id.has_value()) {
        subcontext = Subcontext{
            .id = static_cast<uint32_t>(**subcontext_id),
            .rank = static_cast<uint32_t>(*subcontext_world->rank()),
            .size = static_cast<uint32_t>(*subcontext_world->size()),
        };
    }
    return RunInfo{
        .run_id = run_id,
        .mpi_rank = static_cast<uint32_t>(*world->rank()),
        .world_size = static_cast<uint32_t>(*world->size()),
        .subcontext = subcontext,
        .host_name = control_plane.get_physical_system_descriptor().my_host_name(),
        .written_at = get_utc_now_iso8601(),
        .build = visualizer::get_build_identity(),
    };
}

FabricInfo describe_fabric(const ControlPlane& control_plane) {
    const auto& fabric_context = control_plane.get_fabric_context();
    const bool is_2d_routing = fabric_context.is_2D_routing_enabled();
    return FabricInfo{
        .config =
            FabricSettings{
                .fabric_config = control_plane.get_fabric_config(),
                .reliability_mode = control_plane.get_fabric_reliability_mode(),
                .num_routing_planes = get_num_routing_planes(),
                .tensix_config = control_plane.get_fabric_tensix_config(),
                .udm_mode = control_plane.get_fabric_udm_mode(),
                .fabric_manager = control_plane.get_fabric_manager_mode(),
                .router_config = control_plane.get_fabric_router_config(),
            },
        .topology = fabric_context.get_fabric_topology(),
        .is_2d_routing = is_2d_routing,
        .packet_header = describe_packet_header(fabric_context),
        .packet_header_size_bytes = fabric_context.get_fabric_packet_header_size_bytes(),
        .max_payload_size_bytes = fabric_context.get_fabric_max_payload_size_bytes(),
        .channel_buffer_size_bytes = fabric_context.get_fabric_channel_buffer_size_bytes(),
        .routing_2d_route_buffer_size =
            is_2d_routing ? std::optional(fabric_context.get_2d_pkt_hdr_route_buffer_size()) : std::nullopt,
        .routing_1d_extension_words =
            is_2d_routing ? std::nullopt : std::optional(fabric_context.get_1d_pkt_hdr_extension_words()),
        .multi_txq = fabric_context.get_builder_context().get_fabric_router_config().multi_txq_enabled(),
        .host_clears = describe_host_clears(fabric_context.get_builder_context()),
        .intermesh_vc = describe_intermesh_vc(fabric_context.get_builder_context()),
    };
}

std::vector<VcCreditTransport> describe_credit_transport(const FabricBuilderContext& builder_context, MeshId mesh_id) {
    const auto& plan = builder_context.get_stream_assignment(mesh_id).plan();
    static_assert(builder_config::MAX_NUM_VCS == 3, "CreditTransportPlan has one bool per VC");
    const std::array<bool, builder_config::MAX_NUM_VCS> vc_uses_counters = {
        plan.vc0_uses_counters, plan.vc1_uses_counters, plan.vc2_uses_counters};
    std::vector<VcCreditTransport> transport;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (builder_context.get_max_sender_channels_per_vc()[vc] == 0) {
            continue;
        }
        transport.push_back(VcCreditTransport{
            .vc = vc,
            .backing = vc_uses_counters[vc] ? CreditBacking::L1_COUNTER : CreditBacking::STREAM_REGISTER,
        });
    }
    return transport;
}

Arch describe_arch(const tt::tt_metal::Hal& hal) {
    return Arch{
        .arch = hal.get_arch(),
        .l1 = describe_eth_core_l1(hal),
        .areas =
            ArchAreas{
                .heartbeat = describe_heartbeat(hal.get_arch()),
                .fabric_telemetry = describe_hal_l1_area<FabricTelemetry>(hal, HalL1MemAddrType::FABRIC_TELEMETRY),
                .routing_table = describe_hal_l1_area<routing_l1_info_t>(hal, HalL1MemAddrType::ROUTING_TABLE),
                .eth_fw_mailbox = hal.get_supports_eth_fw_mailbox() ? std::optional(describe_hal_l1_area<uint32_t>(
                                                                          hal, HalL1MemAddrType::ETH_FW_MAILBOX))
                                                                    : std::nullopt,
                .base_fw_port_status = describe_base_fw_word(hal, FWMailboxMsg::PORT_STATUS),
                .base_fw_train_status = describe_base_fw_word(hal, FWMailboxMsg::TRAIN_STATUS),
                .base_fw_retrain_count = describe_base_fw_word(hal, FWMailboxMsg::RETRAIN_COUNT),
                .base_fw_rx_link_up = describe_base_fw_word(hal, FWMailboxMsg::RX_LINK_UP),
                .base_fw_postcode = describe_base_fw_word(hal, FWMailboxMsg::POSTCODE),
            },
        .core_registers = describe_core_registers(hal),
        .fabric_telemetry_version = FABRIC_TELEMETRY_VERSION,
    };
}

}  // namespace tt::tt_fabric::manifest
