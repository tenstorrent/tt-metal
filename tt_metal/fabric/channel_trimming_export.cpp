// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exports per-router channel trimming capture data as YAML.
//
// Output schema:
//
//   channel_trimming_capture:
//     chip_<id>:
//       eth_channel_<id>:
//         direction: <EAST|WEST|NORTH|SOUTH|Z>
//         sender_channel_used_bitfield: 0x001F        # bit per sender channel, set if used
//         sender_channels:                             # details for each used sender channel
//           - id: 0
//             min_packet_size_bytes: 128
//             max_packet_size_bytes: 4096
//             forwarded_from_receiver_vcs: [0, 1]      # VCs that forwarded to this sender
//         sender_channel_forwarded_to_by_vc:           # per-VC bitfield of which sender channels received forwards
//           - vc: 0
//             bitfield: 0x001F
//           - vc: 1
//             bitfield: 0x0000
//         receiver_channel_data_forwarded_bitfield: 0x0003  # bit per receiver channel, set if forwarded data
//         noc_send_types_by_vc:                        # per-VC bitfield of NOC send types observed
//           - vc: 0
//             types: [NOC_UNICAST_WRITE, NOC_UNICAST_INLINE_WRITE]
//           - vc: 1
//             types: []

#include "channel_trimming_export.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <yaml-cpp/yaml.h>

#include "tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming_types.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "hostdevcommon/fabric_common.h"

#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_fabric {

namespace {

using CaptureResults =
    FabricDatapathUsageL1Results<true, builder_config::MAX_NUM_VCS, builder_config::num_max_sender_channels>;

const char* noc_send_type_to_string(uint8_t type) {
    switch (type) {
        case NocSendType::NOC_UNICAST_WRITE: return "NOC_UNICAST_WRITE";
        case NocSendType::NOC_UNICAST_INLINE_WRITE: return "NOC_UNICAST_INLINE_WRITE";
        case NocSendType::NOC_UNICAST_ATOMIC_INC: return "NOC_UNICAST_ATOMIC_INC";
        case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: return "NOC_FUSED_UNICAST_ATOMIC_INC";
        case NocSendType::NOC_UNICAST_SCATTER_WRITE: return "NOC_UNICAST_SCATTER_WRITE";
        case NocSendType::NOC_MULTICAST_WRITE: return "NOC_MULTICAST_WRITE";
        case NocSendType::NOC_MULTICAST_ATOMIC_INC: return "NOC_MULTICAST_ATOMIC_INC";
        case NocSendType::NOC_UNICAST_READ: return "NOC_UNICAST_READ";
        default: return "UNKNOWN";
    }
}

const char* direction_to_string(eth_chan_directions dir) {
    switch (dir) {
        case eth_chan_directions::EAST: return "EAST";
        case eth_chan_directions::WEST: return "WEST";
        case eth_chan_directions::NORTH: return "NORTH";
        case eth_chan_directions::SOUTH: return "SOUTH";
        case eth_chan_directions::Z: return "Z";
        default: return "UNKNOWN";
    }
}

void emit_sender_channels(
    YAML::Emitter& emitter,
    const CaptureResults& capture) {
    uint16_t used_bitfield = capture.sender_channel_used_bitfield_by_vc;
    emitter << YAML::Key << "sender_channel_used_bitfield" << YAML::Value
            << fmt::format("0x{:04X}", used_bitfield);

    emitter << YAML::Key << "sender_channels" << YAML::Value << YAML::BeginSeq;
    for (size_t ch = 0; ch < builder_config::num_max_sender_channels; ch++) {
        if (!(used_bitfield & (1u << ch))) {
            continue;
        }
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "id" << YAML::Value << ch;
        emitter << YAML::Key << "min_packet_size_bytes" << YAML::Value
                << capture.sender_channel_min_packet_size_seen_bytes_by_vc[ch];
        emitter << YAML::Key << "max_packet_size_bytes" << YAML::Value
                << capture.sender_channel_max_packet_size_seen_bytes_by_vc[ch];

        // Collect VCs that forward to this sender channel
        std::vector<size_t> forwarded_from_vcs;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; vc++) {
            if (capture.sender_channel_forwarded_to_bitfield_by_vc[vc] & (1u << ch)) {
                forwarded_from_vcs.push_back(vc);
            }
        }
        emitter << YAML::Key << "forwarded_from_receiver_vcs" << YAML::Value;
        emitter << YAML::Flow << YAML::BeginSeq;
        for (auto vc : forwarded_from_vcs) {
            emitter << vc;
        }
        emitter << YAML::EndSeq;

        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
}

void emit_sender_forwarded_to_bitfields(
    YAML::Emitter& emitter,
    const CaptureResults& capture) {
    emitter << YAML::Key << "sender_channel_forwarded_to_by_vc" << YAML::Value << YAML::BeginSeq;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; vc++) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "vc" << YAML::Value << vc;
        emitter << YAML::Key << "bitfield" << YAML::Value
                << fmt::format("0x{:04X}", capture.sender_channel_forwarded_to_bitfield_by_vc[vc]);
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
}

void emit_receiver_channels(
    YAML::Emitter& emitter,
    const CaptureResults& capture) {
    uint16_t fwd_bitfield = capture.receiver_channel_data_forwarded_bitfield_by_vc;
    emitter << YAML::Key << "receiver_channel_data_forwarded_bitfield" << YAML::Value
            << fmt::format("0x{:04X}", fwd_bitfield);
}

void emit_noc_send_types(
    YAML::Emitter& emitter,
    const CaptureResults& capture) {
    emitter << YAML::Key << "noc_send_types_by_vc" << YAML::Value << YAML::BeginSeq;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; vc++) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "vc" << YAML::Value << vc;

        uint16_t type_bitfield = capture.used_noc_send_type_by_vc_bitfield[vc];
        emitter << YAML::Key << "types" << YAML::Value;
        emitter << YAML::Flow << YAML::BeginSeq;
        for (uint8_t t = 0; t <= NocSendType::NOC_UNICAST_READ; t++) {
            if (type_bitfield & (1u << t)) {
                emitter << noc_send_type_to_string(t);
            }
        }
        emitter << YAML::EndSeq;

        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
}

void emit_capture_channel_yaml(YAML::Emitter& emitter, const CaptureResults& capture, const char* direction) {
    emitter << YAML::Key << "direction" << YAML::Value << direction;
    emit_sender_channels(emitter, capture);
    emit_sender_forwarded_to_bitfields(emitter, capture);
    emit_receiver_channels(emitter, capture);
    emit_noc_send_types(emitter, capture);
}

}  // namespace

void export_channel_trimming_capture() {
    auto& metal_ctx = tt::tt_metal::MetalContext::instance();
    const auto& rtoptions = metal_ctx.rtoptions();

    // Guard: capture must be enabled via runtime option
    if (!rtoptions.get_enable_channel_trimming_capture()) {
        return;
    }

    auto& cluster = metal_ctx.get_cluster();

    // Guard: skip on Mock devices
    if (cluster.get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }

    auto& control_plane = metal_ctx.get_control_plane();

    // Guard: capture buffer must be allocated
    const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
    auto buffer_map = builder_ctx.get_telemetry_and_metadata_buffer_map();
    if (!buffer_map.channel_trimming_capture.is_enabled()) {
        return;
    }

    size_t capture_addr = buffer_map.channel_trimming_capture.l1_address;
    size_t capture_size = buffer_map.channel_trimming_capture.size_bytes;

    auto all_chips = cluster.all_chip_ids();

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "channel_trimming_capture" << YAML::Value << YAML::BeginMap;

    auto& umd_cluster = const_cast<tt::umd::Cluster&>(*cluster.get_driver());

    for (ChipId chip_id : all_chips) {
        FabricNodeId fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(chip_id);

        const auto logical_cores = control_plane.get_active_ethernet_cores(chip_id);
        if (logical_cores.empty()) {
            continue;
        }

        const auto& chan_map = cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map;

        // Collect cores that have a valid channel mapping before emitting
        std::vector<std::pair<CoreCoord, chan_id_t>> mapped_cores;
        for (const auto& logical_core : logical_cores) {
            auto chan_it = chan_map.find(logical_core);
            if (chan_it != chan_map.end()) {
                mapped_cores.emplace_back(logical_core, static_cast<chan_id_t>(chan_it->second));
            }
        }
        if (mapped_cores.empty()) {
            continue;
        }

        cluster.l1_barrier(chip_id);

        emitter << YAML::Key << fmt::format("chip_{}", chip_id) << YAML::Value << YAML::BeginMap;

        for (const auto& [logical_core, channel_id] : mapped_cores) {
            eth_chan_directions direction = control_plane.get_eth_chan_direction(fabric_node_id, channel_id);

            const auto& soc_desc = umd_cluster.get_soc_descriptor(chip_id);
            tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(channel_id, tt::CoordSystem::LOGICAL);

            std::vector<std::byte> buffer(capture_size);
            umd_cluster.read_from_device(buffer.data(), chip_id, eth_core, capture_addr, capture_size);

            CaptureResults capture{};
            std::memcpy(&capture, buffer.data(), std::min(capture_size, sizeof(CaptureResults)));

            emitter << YAML::Key << fmt::format("eth_channel_{}", channel_id) << YAML::Value << YAML::BeginMap;
            emit_capture_channel_yaml(emitter, capture, direction_to_string(direction));
            emitter << YAML::EndMap;
        }

        emitter << YAML::EndMap;
    }

    emitter << YAML::EndMap;
    emitter << YAML::EndMap;

    // Write to file
    std::filesystem::path output_path =
        std::filesystem::path(rtoptions.get_logs_dir()) / "generated" / "reports" / "channel_trimming_capture.yaml";
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream out_file(output_path);
    if (!out_file.is_open()) {
        log_warning(tt::LogFabric, "Failed to open output file for channel trimming capture: {}", output_path.string());
        return;
    }
    out_file << emitter.c_str();
    out_file.close();

    log_info(tt::LogFabric, "Channel trimming capture data written to: {}", output_path.string());
}

}  // namespace tt::tt_fabric
