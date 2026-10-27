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
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/arch/architecture_implementation.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_build_identity.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_catalog.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
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

// A HAL L1 area holding a whole number of `element_type`s, described from its address as one of them or as an array of
// as many as the HAL's size holds.
content::L1 describe_hal_l1_area(
    const tt::tt_metal::Hal& hal, HalL1MemAddrType area_id, const layout::Type& element_type) {
    const uint32_t area_size = get_hal_l1_size(hal, area_id);
    TT_FATAL(
        element_type.size > 0 && area_size >= element_type.size && area_size % element_type.size == 0,
        "Fabric manifest: the HAL's {} area is {} bytes, which is not a whole number of its {}-byte {}",
        lower_enum_name(area_id),
        area_size,
        element_type.size,
        schema_name(element_type));
    const uint32_t num_elements = area_size / element_type.size;
    return {
        .address = get_hal_l1_address(hal, area_id),
        .type = num_elements == 1 ? element_type
                                  : layout::Type{element_type.element, num_elements * element_type.size, num_elements}};
}

// A base firmware word, whose address the HAL keeps among its Ethernet firmware mailbox values; null when it is 0.
std::optional<content::L1> describe_base_fw_word(const tt::tt_metal::Hal& hal, tt::tt_metal::FWMailboxMsg word) {
    const uint32_t address = hal.get_eth_fw_mailbox_val(word);
    if (address == 0) {
        return std::nullopt;
    }
    return content::L1{.address = address, .type = layout::type_of<uint32_t>()};
}

// The kernel picks its heartbeat word's address by architecture: Blackhole's, or else Wormhole's.
Heartbeat describe_heartbeat(tt::ARCH arch) {
    return Heartbeat{
        .word =
            {.address = arch == tt::ARCH::BLACKHOLE ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE
                                                    : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE,
             .type = layout::type_of<uint32_t>()},
        .magic = FABRIC_KERNEL_HEARTBEAT_MAGIC,
        .magic_mask = FABRIC_KERNEL_HEARTBEAT_MAGIC_MASK,
        .period_iters = FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS,
    };
}

// Where a stream register is in its stream's block, and how many of its low bits hold its value.
StreamRegisterLayout describe_stream_register_layout(const tt::tt_metal::Hal& hal, StreamRegister reg) {
    switch (reg) {
        case StreamRegister::BUF_SPACE_AVAILABLE:
            // The bits above the count can hold other state.
            return {
                .offset = static_cast<uint32_t>(
                    hal.get_noc_stream_remote_dest_buf_space_available_reg_index() * sizeof(uint32_t)),
                .bits = hal.get_noc_stream_remote_dest_words_free_width()};
        case StreamRegister::REMOTE_SRC:
            // The fabric's scratch register: four 6-bit fields on Wormhole and Blackhole (fabric_stream_regs.hpp).
            return {
                .offset = static_cast<uint32_t>(hal.get_noc_stream_remote_src_reg_index() * sizeof(uint32_t)),
                .bits = 24};
    }
    TT_THROW("Fabric manifest: unknown stream register {}", static_cast<int>(reg));
}

StreamRegisters describe_stream_registers(const tt::tt_metal::Hal& hal) {
    TT_FATAL(
        hal.has_stream_registers(),
        "Fabric manifest: the HAL gives no stream register addresses on {}",
        lower_enum_name(hal.get_arch()));
    StreamRegisters out{
        .base = hal.get_noc_overlay_start_addr(),
        .stride = hal.get_noc_stream_reg_space_size(),
        .count = StreamRegAssignments::num_eth_stream_registers,
    };
    for (const auto reg : enchantum::values<StreamRegister>) {
        out.registers.emplace_back(reg, describe_stream_register_layout(hal, reg));
    }
    return out;
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

}  // namespace

RunInfo describe_run(const ControlPlane& control_plane) {
    const auto& world = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    return RunInfo{
        .host_rank = static_cast<uint32_t>(*control_plane.get_local_host_rank_id_binding()),
        .mpi_rank = static_cast<uint32_t>(*world->rank()),
        .world_size = static_cast<uint32_t>(*world->size()),
        .written_at = get_utc_now_iso8601(),
        .build = get_build_identity(),
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
        .packet_header_size_bytes = fabric_context.get_fabric_packet_header_size_bytes(),
        .max_payload_size_bytes = fabric_context.get_fabric_max_payload_size_bytes(),
        .channel_buffer_size_bytes = fabric_context.get_fabric_channel_buffer_size_bytes(),
        .routing_2d_route_buffer_size =
            is_2d_routing ? std::optional(fabric_context.get_2d_pkt_hdr_route_buffer_size()) : std::nullopt,
        .routing_1d_extension_words =
            is_2d_routing ? std::nullopt : std::optional(fabric_context.get_1d_pkt_hdr_extension_words()),
        .multi_txq = fabric_context.get_builder_context().get_fabric_router_config().multi_txq_enabled(),
    };
}

std::vector<VcCreditTransport> describe_credit_transport(const FabricBuilderContext& builder_context, MeshId mesh_id) {
    const auto& plan = builder_context.get_stream_assignment(mesh_id).plan();
    std::vector<VcCreditTransport> transport;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (builder_context.get_max_sender_channels_per_vc()[vc] == 0) {
            continue;
        }
        transport.push_back(VcCreditTransport{
            .vc = vc,
            .backing = plan.vc_uses_counters(vc) ? CreditBacking::L1_COUNTER : CreditBacking::STREAM_REGISTER,
            .l1_counter_reasons = plan.get_l1_counters_reasons(vc),
        });
    }
    return transport;
}

Arch describe_arch(const tt::tt_metal::Hal& hal) {
    return Arch{
        .arch = hal.get_arch(),
        .areas =
            ArchAreas{
                .heartbeat = describe_heartbeat(hal.get_arch()),
                .fabric_telemetry =
                    describe_hal_l1_area(hal, HalL1MemAddrType::FABRIC_TELEMETRY, layout::type_of<FabricTelemetry>()),
                .routing_table =
                    describe_hal_l1_area(hal, HalL1MemAddrType::ROUTING_TABLE, layout::type_of<routing_l1_info_t>()),
                .eth_fw_mailbox = hal.get_supports_eth_fw_mailbox()
                                      ? std::optional(describe_hal_l1_area(
                                            hal, HalL1MemAddrType::ETH_FW_MAILBOX, layout::type_of<uint32_t>()))
                                      : std::nullopt,
                .base_fw_port_status = describe_base_fw_word(hal, FWMailboxMsg::PORT_STATUS),
                .base_fw_train_status = describe_base_fw_word(hal, FWMailboxMsg::TRAIN_STATUS),
                .base_fw_retrain_count = describe_base_fw_word(hal, FWMailboxMsg::RETRAIN_COUNT),
                .base_fw_rx_link_up = describe_base_fw_word(hal, FWMailboxMsg::RX_LINK_UP),
                .base_fw_postcode = describe_base_fw_word(hal, FWMailboxMsg::POSTCODE),
            },
        .stream_registers = describe_stream_registers(hal),
        .core_registers = describe_core_registers(hal),
    };
}

TypeCatalog describe_types() {
    return TypeCatalog{.structs = described_struct_types(), .enums = described_enum_types()};
}

}  // namespace tt::tt_fabric::manifest
