// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core_coord.hpp"
#include <common/TracyTTDeviceData.hpp>
#include <device.hpp>
#include <distributed.hpp>
#include "device_pool.hpp"
#include "llrt/hal.hpp"
#include "tools/profiler/event_metadata.hpp"
#include "distributed/fd_mesh_command_queue.hpp"
#include <host_api.hpp>
#include <enchantum/enchantum.hpp>
#include <nlohmann/json.hpp>
#include <stack>
#include <tracy/TracyTTDevice.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "assert.hpp"
#include "dispatch/hardware_command_queue.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "hal_types.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include "profiler.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "profiler_state_manager.hpp"
#include "tools/profiler/noc_event_profiler_utils.hpp"
#include "tracy/Tracy.hpp"
#include "tt-metalium/profiler_types.hpp"
#include "tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/arch/wormhole_implementation.hpp>
#include <tt-metalium/device_pool.hpp>
#include "tt_cluster.hpp"

namespace tt {

namespace tt_metal {

static kernel_profiler::PacketTypes get_packet_type(uint32_t timer_id) {
    return static_cast<kernel_profiler::PacketTypes>((timer_id >> 16) & 0x7);
}

tracy::TTDeviceMarkerType get_marker_type_from_packet_type(kernel_profiler::PacketTypes packet_type) {
    switch (packet_type) {
        case kernel_profiler::PacketTypes::ZONE_START: return tracy::TTDeviceMarkerType::ZONE_START;
        case kernel_profiler::PacketTypes::ZONE_END: return tracy::TTDeviceMarkerType::ZONE_END;
        case kernel_profiler::PacketTypes::ZONE_TOTAL: return tracy::TTDeviceMarkerType::ZONE_TOTAL;
        case kernel_profiler::PacketTypes::TS_DATA: return tracy::TTDeviceMarkerType::TS_DATA;
        case kernel_profiler::PacketTypes::TS_EVENT: return tracy::TTDeviceMarkerType::TS_EVENT;
        default: TT_THROW("Invalid packet type");
    }
}

uint32_t hash32CT(const char* str, size_t n, uint32_t basis) {
    return n == 0 ? basis : hash32CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

uint16_t hash16CT(const std::string& str) {
    uint32_t res = hash32CT(str.c_str(), str.length(), UINT32_C(2166136261));
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

void populateZoneSrcLocations(
    const std::string& new_log_name,
    const std::string& log_name,
    const bool push_new,
    std::unordered_map<uint16_t, tracy::MarkerDetails>& hash_to_zone_src_locations,
    std::unordered_set<std::string>& zone_src_locations) {
    std::ifstream log_file_read(new_log_name);
    std::string line;
    while (std::getline(log_file_read, line)) {
        std::string delimiter = "'#pragma message: ";
        int delimiter_index = line.find(delimiter) + delimiter.length();
        std::string zone_src_location = line.substr(delimiter_index, line.length() - delimiter_index - 1);

        uint16_t hash_16bit = hash16CT(zone_src_location);

        auto did_insert = zone_src_locations.insert(zone_src_location);
        if (did_insert.second && (hash_to_zone_src_locations.find(hash_16bit) != hash_to_zone_src_locations.end())) {
            TT_THROW("Source location hashes are colliding, two different locations are having the same hash");
        }

        std::stringstream ss(zone_src_location);
        std::string zone_name;
        std::string source_file;
        std::string line_num_str;
        std::getline(ss, zone_name, ',');
        std::getline(ss, source_file, ',');
        std::getline(ss, line_num_str, ',');

        tracy::MarkerDetails details(zone_name, source_file, std::stoull(line_num_str));

        auto ret = hash_to_zone_src_locations.emplace(hash_16bit, details);
        if (ret.second && push_new) {
            std::ofstream log_file_write(log_name, std::ios::app);
            log_file_write << line << std::endl;
            log_file_write.close();
        }
    }
    log_file_read.close();
}

std::unordered_map<uint16_t, tracy::MarkerDetails> generateZoneSourceLocationsHashes() {
    std::unordered_map<uint16_t, tracy::MarkerDetails> hash_to_zone_src_locations;
    std::unordered_set<std::string> zone_src_locations;

    // Load existing zones from previous runs
    populateZoneSrcLocations(
        tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG, "", false, hash_to_zone_src_locations, zone_src_locations);

    // Load new zones from the current run
    populateZoneSrcLocations(
        tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG,
        tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG,
        true,
        hash_to_zone_src_locations,
        zone_src_locations);

    return hash_to_zone_src_locations;
}

void mergeSortedDeviceMarkerChunks(
    std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    const std::vector<uint32_t>& device_markers_chunk_offsets,
    ThreadPool& thread_pool) {
    const uint32_t num_chunks = device_markers_chunk_offsets.size() - 1;

    uint32_t num_chunks_to_merge_together = 2;
    while (num_chunks_to_merge_together <= num_chunks) {
        uint32_t i = 0;
        while (i <= num_chunks - num_chunks_to_merge_together) {
            thread_pool.enqueue([&device_markers, &device_markers_chunk_offsets, i, num_chunks_to_merge_together]() {
                TT_ASSERT(std::is_sorted(
                    device_markers.begin() + device_markers_chunk_offsets[i],
                    device_markers.begin() + device_markers_chunk_offsets[i + (num_chunks_to_merge_together / 2)],
                    [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
                       std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); }));
                TT_ASSERT(std::is_sorted(
                    device_markers.begin() + device_markers_chunk_offsets[i + (num_chunks_to_merge_together / 2)],
                    device_markers.begin() + device_markers_chunk_offsets[i + num_chunks_to_merge_together],
                    [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
                       std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); }));

                std::inplace_merge(
                    device_markers.begin() + device_markers_chunk_offsets[i],
                    device_markers.begin() + device_markers_chunk_offsets[i + (num_chunks_to_merge_together / 2)],
                    device_markers.begin() + device_markers_chunk_offsets[i + num_chunks_to_merge_together],
                    [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
                       std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); });
            });
            i += num_chunks_to_merge_together;
        }

        thread_pool.wait();

        TT_ASSERT(std::is_sorted(
            device_markers.begin() + device_markers_chunk_offsets[i - num_chunks_to_merge_together],
            device_markers.begin() + device_markers_chunk_offsets[i],
            [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
               std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); }));
        TT_ASSERT(std::is_sorted(
            device_markers.begin() + device_markers_chunk_offsets[i],
            device_markers.end(),
            [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
               std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); }));

        std::inplace_merge(
            device_markers.begin() + device_markers_chunk_offsets[i - num_chunks_to_merge_together],
            device_markers.begin() + device_markers_chunk_offsets[i],
            device_markers.end(),
            [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
               std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); });

        num_chunks_to_merge_together *= 2;
    }

    TT_ASSERT(std::is_sorted(
        device_markers.begin(),
        device_markers.end(),
        [](std::reference_wrapper<const tracy::TTDeviceMarker> a,
           std::reference_wrapper<const tracy::TTDeviceMarker> b) { return a.get() < b.get(); }));
}

// Merges markers from each (physical core, risc type) group into a single sorted vector. The markers in each group
// should already be sorted.
//
// IMPORTANT: This function creates a vector of references to the TTDeviceMarker objects stored in
// device_markers_per_core_risc_map. These are direct references to the original objects, not copies of the data.
// Thread safety warning: device_markers_per_core_risc_map MUST NOT be modified (no insertions, deletions, or rehashing)
// while these references are in use, as this could invalidate the references and cause undefined behavior.
std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>> getSortedDeviceMarkersVector(
    const std::map<CoreCoord, std::map<tracy::RiscType, std::set<tracy::TTDeviceMarker>>>&
        device_markers_per_core_risc_map,
    ThreadPool& thread_pool) {
    uint32_t total_num_markers = 0;
    auto middle = device_markers_per_core_risc_map.begin();
    std::advance(middle, device_markers_per_core_risc_map.size() / 2);
    uint32_t middle_index = 0;
    std::vector<uint32_t> device_markers_chunk_offsets;
    for (const auto& [core, risc_map] : device_markers_per_core_risc_map) {
        if (core == middle->first) {
            middle_index = total_num_markers;
        }
        for (const auto& [_, markers] : risc_map) {
            device_markers_chunk_offsets.push_back(total_num_markers);
            total_num_markers += markers.size();
        }
    }

    device_markers_chunk_offsets.push_back(total_num_markers);

    tracy::TTDeviceMarker dummy_marker;
    std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>> device_markers_vec(
        total_num_markers, std::cref(dummy_marker));

    thread_pool.enqueue([&device_markers_vec, &device_markers_per_core_risc_map, middle, middle_index]() {
        uint32_t i = middle_index;
        for (auto it = middle; it != device_markers_per_core_risc_map.end(); ++it) {
            for (const auto& [_, markers] : it->second) {
                for (const tracy::TTDeviceMarker& marker : markers) {
                    device_markers_vec[i] = std::cref(marker);
                    ++i;
                }
            }
        }
    });

    uint32_t i = 0;
    for (auto it = device_markers_per_core_risc_map.begin(); it != middle; ++it) {
        for (const auto& [_, markers] : it->second) {
            for (const tracy::TTDeviceMarker& marker : markers) {
                device_markers_vec[i] = std::cref(marker);
                ++i;
            }
        }
    }

    thread_pool.wait();

    mergeSortedDeviceMarkerChunks(device_markers_vec, device_markers_chunk_offsets, thread_pool);

    return device_markers_vec;
}

bool doAllDispatchCoresComeAfterNonDispatchCores(const IDevice* device, const std::vector<CoreCoord>& virtual_cores) {
    const auto& dispatch_core_config = get_dispatch_core_config();
    const std::vector<CoreCoord> logical_dispatch_cores =
        get_logical_dispatch_cores(device->id(), device->num_hw_cqs(), dispatch_core_config);

    std::vector<CoreCoord> virtual_dispatch_cores;
    for (const CoreCoord& core : logical_dispatch_cores) {
        const CoreCoord virtual_dispatch_core =
            device->virtual_core_from_logical_core(core, dispatch_core_config.get_core_type());
        virtual_dispatch_cores.push_back(virtual_dispatch_core);
    }

    bool has_dispatch_core_been_found = false;
    for (const CoreCoord& core : virtual_cores) {
        if (std::find(virtual_dispatch_cores.begin(), virtual_dispatch_cores.end(), core) !=
            virtual_dispatch_cores.end()) {
            has_dispatch_core_been_found = true;
        } else if (has_dispatch_core_been_found) {
            return false;
        }
    }
    return true;
}

CoreCoord getPhysicalAddressFromVirtual(chip_id_t device_id, const CoreCoord& c) {
    bool coord_is_translated = MetalContext::instance().get_cluster().arch() != tt::ARCH::WORMHOLE_B0 ||
                               c.x >= tt::umd::wormhole::tensix_translated_coordinate_start_x ||
                               c.y >= tt::umd::wormhole::tensix_translated_coordinate_start_y ||
                               c.x >= tt::umd::wormhole::eth_translated_coordinate_start_x ||
                               c.y >= tt::umd::wormhole::eth_translated_coordinate_start_y;

    try {
        if (MetalContext::instance().hal().is_coordinate_virtualization_enabled() && coord_is_translated) {
            const metal_SocDescriptor& soc_desc =
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
            // disable linting here; slicing is __intended__
            // NOLINTBEGIN
            return soc_desc.translate_coord_to(c, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            // NOLINTEND
        } else {
            return c;
        }
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "Failed to translate virtual coordinate {},{} to physical", c.x, c.y);
        return c;
    }
    return c;
}

bool isMarkerAZoneEndpoint(const tracy::TTDeviceMarker& marker) {
    return marker.marker_type == tracy::TTDeviceMarkerType::ZONE_START ||
           marker.marker_type == tracy::TTDeviceMarkerType::ZONE_END;
}

bool isMarkerATimestampedDatapoint(const tracy::TTDeviceMarker& marker) {
    return marker.marker_type == tracy::TTDeviceMarkerType::TS_DATA;
}

void addFabricMuxEvents(
    std::vector<tracy::TTDeviceMarker>& markers,
    std::unordered_map<CoreCoord, std::queue<tracy::TTDeviceMarker>>& fabric_mux_markers,
    const CoreCoord& fabric_mux_core) {
    using EMD = KernelProfilerNocEventMetadata;
    for (int i = 0; i < markers.size(); i++) {
        if (isMarkerATimestampedDatapoint(markers[i]) &&
            CoreCoord(markers[i].core_x, markers[i].core_y) == fabric_mux_core &&
            std::get<EMD::LocalNocEvent>(EMD(markers[i].data).getContents()).noc_xfer_type ==
                EMD::NocEventType::WRITE_) {
            fabric_mux_markers[fabric_mux_core].push(markers[i]);
        }
    }
}

void removeFabricMuxEvents(
    std::vector<std::variant<FabricEventMarkers, tracy::TTDeviceMarker>>& coalesced_events,
    const CoreCoord& fabric_mux_core) {
    std::vector<std::variant<FabricEventMarkers, tracy::TTDeviceMarker>> filtered_events;
    for (int i = 0; i < coalesced_events.size(); i++) {
        if (std::holds_alternative<tracy::TTDeviceMarker>(coalesced_events[i])) {
            auto event = std::get<tracy::TTDeviceMarker>(coalesced_events[i]);

            if (isMarkerAZoneEndpoint(event) || CoreCoord(event.core_x, event.core_y) != fabric_mux_core) {
                filtered_events.push_back(coalesced_events[i]);
            }
        } else {
            filtered_events.push_back(coalesced_events[i]);
        }
    }
    coalesced_events = std::move(filtered_events);
}

std::unordered_map<RuntimeID, nlohmann::json::array_t> convertNocTracePacketsToJson(
    const std::unordered_set<tracy::TTDeviceMarker>& device_markers,
    chip_id_t device_id,
    const FabricRoutingLookup& routing_lookup) {
    if (!MetalContext::instance().rtoptions().get_profiler_noc_events_enabled()) {
        return std::unordered_map<RuntimeID, nlohmann::json::array_t>();
    }

    using EMD = KernelProfilerNocEventMetadata;
    std::unordered_map<RuntimeID, std::vector<tracy::TTDeviceMarker>> markers_by_opname;
    // Pass 1: separate out start/end zones and noc events from markers and group by runtime id
    for (const tracy::TTDeviceMarker& marker : device_markers) {
        if (isMarkerAZoneEndpoint(marker)) {
            if ((marker.risc == tracy::RiscType::BRISC || marker.risc == tracy::RiscType::NCRISC) &&
                (marker.marker_name.starts_with("TRUE-KERNEL-END") || marker.marker_name.ends_with("-KERNEL"))) {
                markers_by_opname[marker.runtime_host_id].push_back(marker);
            }
        } else if (isMarkerATimestampedDatapoint(marker)) {
            markers_by_opname[marker.runtime_host_id].push_back(marker);
        }
    }

    // Pass 2: sort noc events in each opname group by x, y, proc, and then timestamp
    for (auto& [runtime_id, markers] : markers_by_opname) {
        std::sort(markers.begin(), markers.end(), [](const auto& a, const auto& b) {
            return std::tie(a.core_x, a.core_y, a.risc, a.timestamp) <
                   std::tie(b.core_x, b.core_y, b.risc, b.timestamp);
        });
    }

    // Pass 3: for each opname in events_by_opname, adjust timestamps to be relative to the smallest timestamp within
    // the group with identical sx,sy,proc
    for (auto& [runtime_id, markers] : markers_by_opname) {
        std::tuple<int, int, std::string> reference_event_loc;
        uint64_t reference_timestamp = 0;
        for (tracy::TTDeviceMarker& marker : markers) {
            // if -KERNEL::begin event is found, reset the reference timestamp
            std::string zone = marker.marker_name;
            if (zone.ends_with("-KERNEL") && marker.marker_type == tracy::TTDeviceMarkerType::ZONE_START) {
                reference_timestamp = marker.timestamp;
            }

            // fix timestamp to be relative to reference_timestamp
            marker.timestamp = marker.timestamp - reference_timestamp;
        }
    }

    // Pass 4: group fabric event markers into a single struct to process later
    std::unordered_map<RuntimeID, std::vector<std::variant<FabricEventMarkers, tracy::TTDeviceMarker>>>
        coalesced_events_by_opname;
    for (auto& [runtime_id, markers] : markers_by_opname) {
        // temporary queue to store events on fabric muxes
        std::unordered_map<CoreCoord, std::queue<tracy::TTDeviceMarker>> fabric_mux_markers;

        for (size_t i = 0; i < markers.size(); /* manual increment */) {
            // If it is a zone, simply copy existing event as-is
            if (isMarkerAZoneEndpoint(markers[i])) {
                coalesced_events_by_opname[runtime_id].push_back(markers[i]);
                i += 1;
                continue;
            }

            auto current_event = EMD(markers[i].data).getContents();
            if (std::holds_alternative<EMD::FabricNoCScatterEvent>(current_event) ||
                std::holds_alternative<EMD::FabricNoCEvent>(current_event)) {
                FabricEventMarkers fabric_event_markers;
                if (std::holds_alternative<EMD::FabricNoCScatterEvent>(current_event)) {
                    auto fabric_noc_scatter_event = std::get<EMD::FabricNoCScatterEvent>(current_event);

                    if (i + fabric_noc_scatter_event.num_chunks - 1 >= markers.size()) {
                        log_warning(
                            tt::LogMetal,
                            "[profiler noc tracing] Failed to coalesce fabric noc trace events in op '{}': "
                            "missing remaining fabric scatter write chunks.",
                            markers[i].op_name);
                        i += 1;
                        continue;
                    }

                    for (int j = 0; j < fabric_noc_scatter_event.num_chunks; j++) {
                        fabric_event_markers.fabric_write_markers.push_back(markers[i + j]);
                    }

                    i += fabric_noc_scatter_event.num_chunks - 1;
                } else {
                    fabric_event_markers.fabric_write_markers.push_back(markers[i]);
                }

                if (i + 2 >= markers.size() ||
                    !std::holds_alternative<EMD::FabricRoutingFields1D>(EMD(markers[i + 1].data).getContents()) &&
                        !std::holds_alternative<EMD::FabricRoutingFields2D>(EMD(markers[i + 1].data).getContents()) ||
                    !std::holds_alternative<EMD::LocalNocEvent>(EMD(markers[i + 2].data).getContents()) ||
                    std::get<EMD::LocalNocEvent>(EMD(markers[i + 2].data).getContents()).noc_xfer_type !=
                        EMD::NocEventType::WRITE_) {
                    log_warning(
                        tt::LogMetal,
                        "[profiler noc tracing] Failed to coalesce fabric noc trace events in op '{}': "
                        "missing routing fields event and/or local write.",
                        markers[i].op_name);
                    i += 1;
                    continue;
                }

                fabric_event_markers.fabric_routing_fields_marker = markers[i + 1];
                fabric_event_markers.local_noc_write_marker = markers[i + 2];

                // if local noc write is to a fabric mux (i.e. worker core), add marker for fabric mux
                // if it is to a fabric router (i.e. active ethernet core), do nothing
                auto local_noc_write = std::get<EMD::LocalNocEvent>(EMD(markers[i + 2].data).getContents());
                CoreCoord local_noc_write_dst_virt = {local_noc_write.dst_x, local_noc_write.dst_y};
                const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device_id, local_noc_write_dst_virt);
                if (core_type == HalProgrammableCoreType::TENSIX) {
                    CoreCoord local_noc_write_dst_phys =
                        getPhysicalAddressFromVirtual(device_id, local_noc_write_dst_virt);
                    if (fabric_mux_markers.find(local_noc_write_dst_phys) == fabric_mux_markers.end()) {
                        addFabricMuxEvents(markers, fabric_mux_markers, local_noc_write_dst_phys);
                    }
                    if (fabric_mux_markers[local_noc_write_dst_phys].size() == 0) {
                        log_warning(
                            tt::LogMetal,
                            "[profiler noc tracing] Failed to coalesce fabric noc trace events in op '{}': "
                            "local write is to a worker core but corresponding event from worker (fabric mux) to eth "
                            "router not found.",
                            markers[i].op_name);
                        i += 3;
                        continue;
                    }
                    fabric_event_markers.fabric_mux_marker = fabric_mux_markers[local_noc_write_dst_phys].front();
                    fabric_mux_markers[local_noc_write_dst_phys].pop();
                } else if (core_type != HalProgrammableCoreType::ACTIVE_ETH) {
                    log_warning(
                        tt::LogMetal,
                        "[profiler noc tracing] Failed to coalesce fabric noc trace events in op '{}': "
                        "local noc write is to an invalid core (neither worker nor active eth).",
                        markers[i].op_name);
                    i += 3;
                    continue;
                }

                // Check if timestamps are close enough;
                // otherwise advance past all fabric event markers
                double ts_diff = markers[i + 2].timestamp - markers[i].timestamp;
                if (ts_diff > 1000) {
                    log_warning(
                        tt::LogMetal,
                        "[profiler noc tracing] Failed to coalesce fabric noc trace events because timestamps are "
                        "implausibly "
                        "far apart.");
                    i += 3;
                    continue;
                }

                // Advance past all fabric event markers (fabric_event, fabric_routing_fields,
                // local_noc_write_event)
                i += 3;
                coalesced_events_by_opname[runtime_id].push_back(fabric_event_markers);
            } else {
                // If not a fabric event group, simply copy existing event as-is
                coalesced_events_by_opname[runtime_id].push_back(markers[i]);
                i += 1;
            }
        }

        // remove fabric mux events since they are now part of the coalesced fabric events
        for (auto& [fabric_mux_core, _] : fabric_mux_markers) {
            removeFabricMuxEvents(coalesced_events_by_opname[runtime_id], fabric_mux_core);
        }
    }

    // Pass 5: convert to json
    std::unordered_map<RuntimeID, nlohmann::json::array_t> json_events_by_opname;
    for (auto& [runtime_id, markers] : coalesced_events_by_opname) {
        for (auto marker : markers) {
            if (std::holds_alternative<tracy::TTDeviceMarker>(marker)) {
                auto device_marker = std::get<tracy::TTDeviceMarker>(marker);

                if (isMarkerAZoneEndpoint(device_marker)) {
                    tracy::TTDeviceMarkerType zone_phase =
                        (device_marker.marker_type == tracy::TTDeviceMarkerType::ZONE_END)
                            ? tracy::TTDeviceMarkerType::ZONE_END
                            : tracy::TTDeviceMarkerType::ZONE_START;
                    json_events_by_opname[runtime_id].push_back(nlohmann::ordered_json{
                        {"run_host_id", device_marker.runtime_host_id},
                        {"op_name", device_marker.op_name},
                        {"proc", enchantum::to_string(device_marker.risc)},
                        {"zone", device_marker.marker_name},
                        {"zone_phase", enchantum::to_string(zone_phase)},
                        {"sx", device_marker.core_x},
                        {"sy", device_marker.core_y},
                        {"timestamp", device_marker.timestamp},
                    });
                } else {
                    auto local_noc_event = std::get<EMD::LocalNocEvent>(EMD(device_marker.data).getContents());

                    nlohmann::ordered_json data = {
                        {"run_host_id", device_marker.runtime_host_id},
                        {"op_name", device_marker.op_name},
                        {"proc", enchantum::to_string(device_marker.risc)},
                        {"noc", enchantum::to_string(local_noc_event.noc_type)},
                        {"vc", int(local_noc_event.noc_vc)},
                        {"src_device_id", device_marker.chip_id},
                        {"sx", device_marker.core_x},
                        {"sy", device_marker.core_y},
                        {"num_bytes", local_noc_event.getNumBytes()},
                        {"type", enchantum::to_string(local_noc_event.noc_xfer_type)},
                        {"timestamp", device_marker.timestamp},
                    };

                    // handle dst coordinates correctly for different NocEventType
                    if (local_noc_event.dst_x == -1 || local_noc_event.dst_y == -1 ||
                        local_noc_event.noc_xfer_type == EMD::NocEventType::READ_WITH_STATE ||
                        local_noc_event.noc_xfer_type == EMD::NocEventType::WRITE_WITH_STATE) {
                        // DO NOT emit destination coord; it isn't meaningful

                    } else if (local_noc_event.noc_xfer_type == EMD::NocEventType::WRITE_MULTICAST) {
                        auto phys_start_coord = getPhysicalAddressFromVirtual(
                            device_marker.chip_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                        data["mcast_start_x"] = phys_start_coord.x;
                        data["mcast_start_y"] = phys_start_coord.y;
                        auto phys_end_coord = getPhysicalAddressFromVirtual(
                            device_marker.chip_id, {local_noc_event.mcast_end_dst_x, local_noc_event.mcast_end_dst_y});
                        data["mcast_end_x"] = phys_end_coord.x;
                        data["mcast_end_y"] = phys_end_coord.y;
                    } else {
                        auto phys_coord = getPhysicalAddressFromVirtual(
                            device_marker.chip_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                        data["dx"] = phys_coord.x;
                        data["dy"] = phys_coord.y;
                    }

                    json_events_by_opname[runtime_id].push_back(data);
                }
            } else if (std::holds_alternative<FabricEventMarkers>(marker)) {
                // coalesce fabric event markers into a single logical trace event with extra 'fabric_send' metadata
                auto fabric_event_markers = std::get<FabricEventMarkers>(marker);

                auto first_fabric_write_marker = fabric_event_markers.fabric_write_markers[0];
                auto fabric_routing_fields_marker = fabric_event_markers.fabric_routing_fields_marker;
                auto local_noc_write_marker = fabric_event_markers.local_noc_write_marker;

                EMD::FabricPacketType routing_fields_type;
                EMD::NocEventType noc_xfer_type;
                if (std::holds_alternative<EMD::FabricNoCEvent>(EMD(first_fabric_write_marker.data).getContents())) {
                    auto fabric_write_event =
                        std::get<EMD::FabricNoCEvent>(EMD(first_fabric_write_marker.data).getContents());
                    routing_fields_type = fabric_write_event.routing_fields_type;
                    noc_xfer_type = fabric_write_event.noc_xfer_type;
                } else {
                    auto first_fabric_scatter_write_event =
                        std::get<EMD::FabricNoCScatterEvent>(EMD(first_fabric_write_marker.data).getContents());
                    routing_fields_type = first_fabric_scatter_write_event.routing_fields_type;
                    noc_xfer_type = first_fabric_scatter_write_event.noc_xfer_type;
                }

                auto local_noc_write_event =
                    std::get<EMD::LocalNocEvent>(EMD(local_noc_write_marker.data).getContents());

                nlohmann::ordered_json fabric_event_json = {
                    {"run_host_id", local_noc_write_marker.runtime_host_id},
                    {"op_name", local_noc_write_marker.op_name},
                    {"proc", enchantum::to_string(local_noc_write_marker.risc)},
                    {"noc", enchantum::to_string(local_noc_write_event.noc_type)},
                    {"vc", int(local_noc_write_event.noc_vc)},
                    {"src_device_id", local_noc_write_marker.chip_id},
                    {"sx", local_noc_write_marker.core_x},
                    {"sy", local_noc_write_marker.core_y},
                    {"num_bytes", local_noc_write_event.getNumBytes()},
                    {"type", enchantum::to_string(noc_xfer_type)},  // replace the type with fabric event type
                    {"timestamp", local_noc_write_marker.timestamp},
                };

                // extract routing metadata from routing fields event
                switch (routing_fields_type) {
                    case EMD::FabricPacketType::REGULAR: {
                        auto fabric_routing_fields_event =
                            std::get<EMD::FabricRoutingFields1D>(EMD(fabric_routing_fields_marker.data).getContents());
                        auto [start_distance, range] =
                            get_routing_start_distance_and_range(fabric_routing_fields_event.routing_fields_value);
                        fabric_event_json["fabric_send"] = {{"start_distance", start_distance}, {"range", range}};
                        break;
                    }
                    case EMD::FabricPacketType::LOW_LATENCY: {
                        auto fabric_routing_fields_event =
                            std::get<EMD::FabricRoutingFields1D>(EMD(fabric_routing_fields_marker.data).getContents());
                        auto [start_distance, range] = get_low_latency_routing_start_distance_and_range(
                            fabric_routing_fields_event.routing_fields_value);
                        fabric_event_json["fabric_send"] = {{"start_distance", start_distance}, {"range", range}};
                        break;
                    }
                    case KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY_MESH: {
                        auto fabric_routing_fields_event =
                            std::get<EMD::FabricRoutingFields2D>(EMD(fabric_routing_fields_marker.data).getContents());
                        fabric_event_json["fabric_send"] = {
                            {"ns_hops", fabric_routing_fields_event.ns_hops},
                            {"e_hops", fabric_routing_fields_event.e_hops},
                            {"w_hops", fabric_routing_fields_event.w_hops},
                            {"is_mcast", fabric_routing_fields_event.is_mcast}};
                        break;
                    }
                    case KernelProfilerNocEventMetadata::FabricPacketType::DYNAMIC_MESH: {
                        log_error(
                            tt::LogMetal, "[profiler noc tracing] noc tracing does not support DYNAMIC_MESH packets!");
                        continue;
                    }
                }

                // if fabric mux is used, add fabric mux coords and noc into "fabric_send" metadata
                // and use corresponding write on fabric mux to get eth channel used on src device for the transfer
                if (fabric_event_markers.fabric_mux_marker.has_value()) {
                    // mux core location is derived from the local noc write event
                    auto mux_phys_coord = getPhysicalAddressFromVirtual(
                        local_noc_write_marker.chip_id, {local_noc_write_event.dst_x, local_noc_write_event.dst_y});

                    auto fabric_mux_marker = fabric_event_markers.fabric_mux_marker.value();
                    auto fabric_mux_event = std::get<EMD::LocalNocEvent>(EMD(fabric_mux_marker.data).getContents());

                    fabric_event_json["fabric_send"]["fabric_mux"] = {
                        {"x", mux_phys_coord.x},
                        {"y", mux_phys_coord.y},
                        {"noc", enchantum::to_string(fabric_mux_event.noc_type)}};

                    CoreCoord eth_router_phys_coord = getPhysicalAddressFromVirtual(
                        fabric_mux_marker.chip_id, {fabric_mux_event.dst_x, fabric_mux_event.dst_y});
                    auto eth_chan_opt =
                        routing_lookup.getRouterEthCoreToChannelLookup(device_id, eth_router_phys_coord);
                    if (!eth_chan_opt) {
                        log_error(
                            tt::LogMetal,
                            "[profiler noc tracing] Fabric edm_location->channel lookup failed for event in op '{}' at "
                            "ts {}: "
                            "src_dev={}, "
                            "eth_core=({}, {}). Skipping.",
                            first_fabric_write_marker.op_name,
                            first_fabric_write_marker.timestamp,
                            device_id,
                            eth_router_phys_coord.x,
                            eth_router_phys_coord.y);
                        continue;
                    }
                    tt::tt_fabric::chan_id_t eth_chan = *eth_chan_opt;
                    fabric_event_json["fabric_send"]["eth_chan"] = eth_chan;
                } else {
                    // router eth core location is derived from the local noc write event
                    auto eth_router_phys_coord = getPhysicalAddressFromVirtual(
                        local_noc_write_marker.chip_id, {local_noc_write_event.dst_x, local_noc_write_event.dst_y});
                    auto eth_chan_opt =
                        routing_lookup.getRouterEthCoreToChannelLookup(device_id, eth_router_phys_coord);
                    if (!eth_chan_opt) {
                        log_error(
                            tt::LogMetal,
                            "[profiler noc tracing] Fabric edm_location->channel lookup failed for event in op '{}' at "
                            "ts {}: "
                            "src_dev={}, "
                            "eth_core=({}, {}). Skipping.",
                            first_fabric_write_marker.op_name,
                            first_fabric_write_marker.timestamp,
                            device_id,
                            eth_router_phys_coord.x,
                            eth_router_phys_coord.y);
                        continue;
                    }
                    tt::tt_fabric::chan_id_t eth_chan = *eth_chan_opt;
                    fabric_event_json["fabric_send"]["eth_chan"] = eth_chan;
                }

                // Add true destination coord(s) from fabric unicast/scatter event
                if (KernelProfilerNocEventMetadata::isFabricUnicastEventType(noc_xfer_type)) {
                    auto fabric_write_marker = fabric_event_markers.fabric_write_markers[0];
                    auto fabric_write_event =
                        std::get<EMD::FabricNoCEvent>(EMD(fabric_write_marker.data).getContents());
                    auto phys_coord = getPhysicalAddressFromVirtual(
                        fabric_write_marker.chip_id, {fabric_write_event.dst_x, fabric_write_event.dst_y});
                    fabric_event_json["dst"] = {
                        {{"dx", phys_coord.x},
                         {"dy", phys_coord.y},
                         {"num_bytes", local_noc_write_event.getNumBytes()}}};
                } else if (KernelProfilerNocEventMetadata::isFabricScatterEventType(noc_xfer_type)) {
                    // add all chunks for scatter write and compute last chunk size
                    fabric_event_json["dst"] = nlohmann::json::array();
                    int last_chunk_size = local_noc_write_event.getNumBytes();
                    for (int j = 0; j < fabric_event_markers.fabric_write_markers.size(); j++) {
                        auto fabric_scatter_write_marker = fabric_event_markers.fabric_write_markers[j];
                        auto fabric_scatter_write =
                            std::get<EMD::FabricNoCScatterEvent>(EMD(fabric_scatter_write_marker.data).getContents());
                        auto phys_coord = getPhysicalAddressFromVirtual(
                            fabric_scatter_write_marker.chip_id,
                            {fabric_scatter_write.dst_x, fabric_scatter_write.dst_y});
                        fabric_event_json["dst"].push_back({
                            {"dx", phys_coord.x},
                            {"dy", phys_coord.y},
                            {"num_bytes", fabric_scatter_write.chunk_size},
                        });
                        last_chunk_size -= fabric_scatter_write.chunk_size;
                    }

                    fabric_event_json["dst"].back()["num_bytes"] = last_chunk_size;
                } else {
                    log_error(
                        tt::LogMetal, "[profiler noc tracing] Noc multicasts in fabric events are not supported!");
                    continue;
                }

                json_events_by_opname[runtime_id].push_back(fabric_event_json);
            }
        }
    }

    return json_events_by_opname;
}

void dumpJsonNocTraces(
    const std::vector<std::unordered_map<RuntimeID, nlohmann::json::array_t>>& noc_trace_data,
    chip_id_t device_id,
    const std::filesystem::path& output_dir) {
    // create output directory if it does not exist
    std::filesystem::create_directories(output_dir);
    if (!std::filesystem::is_directory(output_dir)) {
        log_error(
            tt::LogMetal,
            "Could not write profiler noc traces to '{}' because the directory path could not be created!",
            output_dir);
        return;
    }

    for (const auto& processed_events_by_opname : noc_trace_data) {
        for (auto& [runtime_id, events] : processed_events_by_opname) {
            // dump events to a json file inside directory output_dir named after the op_name
            std::filesystem::path rpt_path = output_dir;
            const std::string op_name = events.front().value("op_name", "UnknownOP");
            if (!op_name.empty()) {
                rpt_path /= fmt::format("noc_trace_dev{}_{}_ID{}.json", device_id, op_name, runtime_id);
            } else {
                rpt_path /= fmt::format("noc_trace_dev{}_ID{}.json", device_id, runtime_id);
            }
            std::ofstream file(rpt_path);
            if (file.is_open()) {
                // Write the final processed events for this op
                file << nlohmann::json(events).dump(2);
            } else {
                log_error(tt::LogMetal, "Could not write profiler noc json trace to '{}'", rpt_path);
            }
        }
    }
}

void writeCSVHeader(std::ofstream& log_file_ofs, tt::ARCH device_architecture, int device_core_frequency) {
    log_file_ofs << "ARCH: " << get_string_lowercase(device_architecture)
                 << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
    log_file_ofs << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, "
                    "run host ID,  zone name, type, source line, source file, meta data"
                 << std::endl;
}

void dumpDeviceResultsToCSV(
    const std::map<CoreCoord, std::map<tracy::RiscType, std::set<tracy::TTDeviceMarker>>>&
        device_markers_per_core_risc_map,
    tt::ARCH device_arch,
    int device_core_frequency,
    const std::filesystem::path& log_path) {
    // open CSV log file
    std::ofstream log_file_ofs;

    // append to existing CSV log file if it already exists
    if (std::filesystem::exists(log_path)) {
        log_file_ofs.open(log_path, std::ios_base::app);
    } else {
        log_file_ofs.open(log_path);
        writeCSVHeader(log_file_ofs, device_arch, device_core_frequency);
    }

    if (!log_file_ofs) {
        log_error(tt::LogMetal, "Could not open kernel profiler dump file '{}'", log_path);
        return;
    }

    for (const auto& [core, device_markers_per_risc_map] : device_markers_per_core_risc_map) {
        for (const auto& [risc, device_markers] : device_markers_per_risc_map) {
            for (const tracy::TTDeviceMarker& marker : device_markers) {
                std::string meta_data_str = "";
                if (!marker.meta_data.is_null()) {
                    meta_data_str = marker.meta_data.dump();
                    std::replace(meta_data_str.begin(), meta_data_str.end(), ',', ';');
                }

                log_file_ofs << fmt::format(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    marker.chip_id,
                    marker.core_x,
                    marker.core_y,
                    enchantum::to_string(marker.risc),
                    marker.marker_id,
                    marker.timestamp,
                    marker.data,
                    marker.runtime_host_id,
                    marker.marker_name,
                    enchantum::to_string(marker.marker_type),
                    marker.line,
                    marker.file,
                    meta_data_str);
            }
        }
    }

    log_file_ofs.close();
}

bool isGalaxyMMIODevice(IDevice* device) {
    // This is wrapped in a try-catch block because get_mesh_device() can throw a std::bad_weak_ptr if profiler read is
    // called during MeshDevice::close()
    try {
        if (auto mesh_device = device->get_mesh_device()) {
            return false;
        } else {
            return tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() &&
                   device->is_mmio_capable();
        }
    } catch (const std::bad_weak_ptr& e) {
        return false;
    }
}

bool useFastDispatch(IDevice* device) {
    return tt::DevicePool::instance().is_dispatch_firmware_active() && !isGalaxyMMIODevice(device);
}

void writeToCoreControlBuffer(IDevice* device, const CoreCoord& virtual_core, const std::vector<uint32_t>& data) {
    ZoneScoped;

    const auto& hal = MetalContext::instance().hal();
    const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device->id(), virtual_core);
    DeviceAddr profiler_msg_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::PROFILER);
    DeviceAddr control_vector_addr =
        profiler_msg_addr + hal.get_dev_msgs_factory(core_type).offset_of<dev_msgs::profiler_msg_t>(
                                dev_msgs::profiler_msg_t::Field::control_vector);
    if (useFastDispatch(device)) {
        if (auto mesh_device = device->get_mesh_device()) {
            distributed::FDMeshCommandQueue& mesh_cq =
                dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
            const distributed::DeviceMemoryAddress address = {device_coord, virtual_core, control_vector_addr};
            mesh_cq.enqueue_write_shard_to_core(
                address, data.data(), kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, true);
        } else {
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core(
                    virtual_core,
                    data.data(),
                    control_vector_addr,
                    kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE,
                    true);
        }
    } else {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            device->id(), virtual_core, data, control_vector_addr);
    }
}

void DeviceProfiler::issueFastDispatchReadFromProfilerBuffer(IDevice* device) {
    ZoneScoped;
    TT_ASSERT(tt::DevicePool::instance().is_dispatch_firmware_active());
    const DeviceAddr profiler_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::PROFILER);
    uint32_t profile_buffer_idx = 0;

    const CoreCoord dram_grid_size = device->dram_grid_size();
    for (uint32_t x = 0; x < dram_grid_size.x; ++x) {
        for (uint32_t y = 0; y < dram_grid_size.y; ++y) {
            const CoreCoord dram_core = device->virtual_core_from_logical_core({x, y}, CoreType::DRAM);
            if (auto mesh_device = device->get_mesh_device()) {
                const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device_id);
                dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue())
                    .enqueue_read_shard_from_core(
                        distributed::DeviceMemoryAddress{device_coord, dram_core, profiler_addr},
                        &(profile_buffer[profile_buffer_idx]),
                        profile_buffer_bank_size_bytes,
                        true);
            } else {
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_read_from_core(
                        dram_core,
                        &(profile_buffer[profile_buffer_idx]),
                        profiler_addr,
                        profile_buffer_bank_size_bytes,
                        true);
            }
            profile_buffer_idx += profile_buffer_bank_size_bytes / sizeof(uint32_t);
        }
    }
}

void DeviceProfiler::issueSlowDispatchReadFromProfilerBuffer(IDevice* device) {
    ZoneScoped;
    const DeviceAddr profiler_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::PROFILER);
    uint32_t profile_buffer_idx = 0;

    const int num_dram_channels = device->num_dram_channels();
    for (int dram_channel = 0; dram_channel < num_dram_channels; ++dram_channel) {
        std::vector<uint32_t> profile_buffer_bank_data(profile_buffer_bank_size_bytes / sizeof(uint32_t), 0);
        tt::tt_metal::MetalContext::instance().get_cluster().read_dram_vec(
            profile_buffer_bank_data.data(), profile_buffer_bank_size_bytes, device_id, dram_channel, profiler_addr);

        std::copy(
            profile_buffer_bank_data.begin(),
            profile_buffer_bank_data.end(),
            profile_buffer.begin() + profile_buffer_idx);
        profile_buffer_idx += profile_buffer_bank_size_bytes / sizeof(uint32_t);
    }
}

void DeviceProfiler::issueFastDispatchReadFromL1DataBuffer(
    IDevice* device, const CoreCoord& worker_core, std::vector<uint32_t>& core_l1_data_buffer) {
    ZoneScoped;

    TT_ASSERT(tt::DevicePool::instance().is_dispatch_firmware_active());

    const Hal& hal = MetalContext::instance().hal();
    const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device_id, worker_core);
    DeviceAddr profiler_msg_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::PROFILER);
    DeviceAddr buffer_addr =
        profiler_msg_addr + hal.get_dev_msgs_factory(core_type).offset_of<dev_msgs::profiler_msg_t>(
                                dev_msgs::profiler_msg_t::Field::buffer);
    const uint32_t num_risc_processors = hal.get_num_risc_processors(core_type);
    core_l1_data_buffer.resize(kernel_profiler::PROFILER_L1_VECTOR_SIZE * num_risc_processors);
    if (auto mesh_device = device->get_mesh_device()) {
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device_id);
        dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue())
            .enqueue_read_shard_from_core(
                distributed::DeviceMemoryAddress{device_coord, worker_core, buffer_addr},
                core_l1_data_buffer.data(),
                kernel_profiler::PROFILER_L1_BUFFER_SIZE * num_risc_processors,
                true);
    } else {
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(
                worker_core,
                core_l1_data_buffer.data(),
                buffer_addr,
                kernel_profiler::PROFILER_L1_BUFFER_SIZE * num_risc_processors,
                true);
    }
}

void DeviceProfiler::issueSlowDispatchReadFromL1DataBuffer(
    IDevice* device, const CoreCoord& worker_core, std::vector<uint32_t>& core_l1_data_buffer) {
    ZoneScoped;

    const Hal& hal = MetalContext::instance().hal();
    const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device_id, worker_core);
    DeviceAddr profiler_msg_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::PROFILER);
    DeviceAddr buffer_addr =
        profiler_msg_addr + hal.get_dev_msgs_factory(core_type).offset_of<dev_msgs::profiler_msg_t>(
                                dev_msgs::profiler_msg_t::Field::buffer);
    core_l1_data_buffer = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id,
        worker_core,
        buffer_addr,
        kernel_profiler::PROFILER_L1_BUFFER_SIZE * hal.get_num_risc_processors(core_type));
}

void DeviceProfiler::readL1DataBufferForCore(
    IDevice* device, const CoreCoord& virtual_core, std::vector<uint32_t>& core_l1_data_buffer) {
    ZoneScoped;
    if (useFastDispatch(device)) {
        issueFastDispatchReadFromL1DataBuffer(device, virtual_core, core_l1_data_buffer);
    } else {
        issueSlowDispatchReadFromL1DataBuffer(device, virtual_core, core_l1_data_buffer);
    }
}

void DeviceProfiler::readL1DataBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores) {
    ZoneScoped;

    for (const CoreCoord& virtual_core : virtual_cores) {
        std::vector<uint32_t>& core_l1_data_buffer = core_l1_data_buffers[virtual_core];
        readL1DataBufferForCore(device, virtual_core, core_l1_data_buffer);
    }
}

void DeviceProfiler::readControlBufferForCore(IDevice* device, const CoreCoord& virtual_core) {
    ZoneScoped;
    const auto& hal = MetalContext::instance().hal();
    const HalProgrammableCoreType core_type = tt::llrt::get_core_type(device_id, virtual_core);
    DeviceAddr profiler_msg = hal.get_dev_addr(core_type, HalL1MemAddrType::PROFILER);
    DeviceAddr control_vector_addr =
        profiler_msg + hal.get_dev_msgs_factory(core_type).offset_of<dev_msgs::profiler_msg_t>(
                           dev_msgs::profiler_msg_t::Field::control_vector);
    if (useFastDispatch(device)) {
        if (auto mesh_device = device->get_mesh_device()) {
            distributed::FDMeshCommandQueue& mesh_cq =
                dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device_id);
            const distributed::DeviceMemoryAddress address = {device_coord, virtual_core, control_vector_addr};
            core_control_buffers[virtual_core].resize(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE);
            mesh_cq.enqueue_read_shard_from_core(
                address,
                core_control_buffers[virtual_core].data(),
                kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE,
                true);
        } else {
            core_control_buffers[virtual_core].resize(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE);
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(
                    virtual_core,
                    core_control_buffers[virtual_core].data(),
                    control_vector_addr,
                    kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE,
                    true);
        }
    } else {
        core_control_buffers[virtual_core] = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            device_id, virtual_core, control_vector_addr, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
    }
}

void DeviceProfiler::readControlBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores) {
    ZoneScoped;
    for (const CoreCoord& virtual_core : virtual_cores) {
        readControlBufferForCore(device, virtual_core);
    }
}

void DeviceProfiler::resetControlBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores) {
    ZoneScoped;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_control_buffer_resets;
    for (const CoreCoord& virtual_core : virtual_cores) {
        const std::vector<uint32_t>& control_buffer = core_control_buffers.at(virtual_core);

        std::vector<uint32_t>& core_control_buffer_reset = core_control_buffer_resets[virtual_core];
        core_control_buffer_reset.resize(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE);
        core_control_buffer_reset[kernel_profiler::DRAM_PROFILER_ADDRESS] =
            control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS];
        core_control_buffer_reset[kernel_profiler::FLAT_ID] = control_buffer[kernel_profiler::FLAT_ID];
        core_control_buffer_reset[kernel_profiler::CORE_COUNT_PER_DRAM] =
            control_buffer[kernel_profiler::CORE_COUNT_PER_DRAM];
    }

    for (const auto& [virtual_core, control_buffer_reset] : core_control_buffer_resets) {
        writeToCoreControlBuffer(device, virtual_core, control_buffer_reset);
    }
}

void DeviceProfiler::readProfilerBuffer(IDevice* device) {
    ZoneScoped;
    if (useFastDispatch(device)) {
        issueFastDispatchReadFromProfilerBuffer(device);
    } else {
        issueSlowDispatchReadFromProfilerBuffer(device);
    }
}

void DeviceProfiler::readRiscProfilerResults(
    IDevice* device,
    const CoreCoord& worker_core,
    const ProfilerDataBufferSource data_source,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
    ZoneScoped;

    if (data_source == ProfilerDataBufferSource::DRAM_AND_L1) {
        readRiscProfilerResults(device, worker_core, ProfilerDataBufferSource::DRAM, metadata);
        readRiscProfilerResults(device, worker_core, ProfilerDataBufferSource::L1, metadata);
        return;
    }

    const std::vector<uint32_t>& control_buffer = core_control_buffers.at(worker_core);

    const std::vector<uint32_t>& data_buffer =
        (data_source == ProfilerDataBufferSource::DRAM) ? profile_buffer : core_l1_data_buffers.at(worker_core);

    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();

    if (!rtoptions.get_profiler_trace_only()) {
        if ((control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR_ER] == 0) &&
            (control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_NC] == 0)) {
            return;
        }
    }

    const uint32_t coreFlatID =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_routing_to_profiler_flat_id(device_id).at(
            worker_core);
    const uint32_t startIndex = coreFlatID * MetalContext::instance().hal().get_max_processors_per_core() *
                                PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

    // translate worker core virtual coord to phys coordinates
    const CoreCoord phys_coord = getPhysicalAddressFromVirtual(device_id, worker_core);

    // helper function to lookup opname from runtime id if metadata is available
    auto getOpNameIfAvailable = [&metadata](auto device_id, auto runtime_id) {
        return (metadata.has_value()) ? metadata->get_op_name(device_id, runtime_id) : "";
    };

    HalProgrammableCoreType CoreType = tt::llrt::get_core_type(device_id, worker_core);
    int riscCount = 1;

    if (!rtoptions.get_profiler_trace_only() && CoreType == HalProgrammableCoreType::TENSIX) {
        riscCount = 5;
    }

    std::map<tracy::RiscType, std::set<tracy::TTDeviceMarker>>& device_markers_for_core =
        device_markers_per_core_risc_map[phys_coord];

    for (int riscEndIndex = 0; riscEndIndex < riscCount; riscEndIndex++) {
        uint32_t bufferEndIndex = control_buffer[riscEndIndex];
        if (data_source == ProfilerDataBufferSource::L1) {
            // Just grab the device end index
            bufferEndIndex = control_buffer[riscEndIndex + kernel_profiler::DEVICE_BUFFER_END_INDEX_BR_ER];
        }
        tracy::RiscType riscType;
        if (rtoptions.get_profiler_trace_only() && CoreType == HalProgrammableCoreType::TENSIX) {
            riscType = tracy::RiscType::CORE_AGG;
        } else if (CoreType == HalProgrammableCoreType::TENSIX) {
            riscType = static_cast<tracy::RiscType>(riscEndIndex);
        } else {
            riscType = tracy::RiscType::ERISC;
        }
        if (bufferEndIndex > 0) {
            uint32_t bufferRiscShift = riscEndIndex * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if (data_source == ProfilerDataBufferSource::L1) {
                // Shift by L1 buffer size only
                bufferRiscShift = riscEndIndex * kernel_profiler::PROFILER_L1_VECTOR_SIZE;
            }
            if ((control_buffer[kernel_profiler::DROPPED_ZONES] >> riscEndIndex) & 1) {
                std::string warningMsg = fmt::format(
                    "Profiler DRAM buffers were full, markers were dropped! device {}, worker core {}, {}, Risc "
                    "{},  "
                    "bufferEndIndex = {}. "
                    "Please either decrease the number of ops being profiled or run read device profiler more often",
                    device_id,
                    worker_core.x,
                    worker_core.y,
                    enchantum::to_string(static_cast<tracy::RiscType>(riscEndIndex)),
                    bufferEndIndex);
                TracyMessageC(warningMsg.c_str(), warningMsg.size(), tracy::Color::Tomato3);
                log_warning(tt::LogMetal, "{}", warningMsg);
            }

            uint32_t riscNumRead = 0;
            uint32_t coreFlatIDRead = 0;
            uint32_t runHostCounterRead = 0;

            bool newRunStart = false;
            bool oneStartFound = false;

            uint32_t opTime_H = 0;
            uint32_t opTime_L = 0;
            std::string opname;

            std::set<tracy::TTDeviceMarker>& device_markers_for_core_risc = device_markers_for_core[riscType];

            for (int index = bufferRiscShift; index < (bufferRiscShift + bufferEndIndex);
                 index += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE) {
                if (!newRunStart && data_buffer.at(index) == 0 && data_buffer.at(index + 1) == 0) {
                    newRunStart = true;
                    oneStartFound = true;
                    opTime_H = 0;
                    opTime_L = 0;
                } else if (newRunStart) {
                    newRunStart = false;

                    // TODO(MO): Cleanup magic numbers
                    riscNumRead = data_buffer.at(index) & 0x7;
                    coreFlatIDRead = (data_buffer.at(index) >> 3) & 0xFF;
                    runHostCounterRead = data_buffer.at(index + 1);
                    uint32_t base_program_id =
                        tt::tt_metal::detail::DecodePerDeviceProgramID(runHostCounterRead).base_program_id;

                    opname = getOpNameIfAvailable(device_id, base_program_id);

                } else if (oneStartFound) {
                    uint32_t timer_id = (data_buffer.at(index) >> 12) & 0x7FFFF;
                    kernel_profiler::PacketTypes packet_type = get_packet_type(timer_id);

                    switch (packet_type) {
                        case kernel_profiler::ZONE_START:
                        case kernel_profiler::ZONE_END: {
                            uint32_t time_H = data_buffer.at(index) & 0xFFF;
                            if (timer_id || time_H) {
                                uint32_t time_L = data_buffer.at(index + 1);

                                if (opTime_H == 0) {
                                    opTime_H = time_H;
                                }
                                if (opTime_L == 0) {
                                    opTime_L = time_L;
                                }

                                TT_ASSERT(
                                    riscNumRead == riscEndIndex,
                                    "Unexpected risc id, expected {}, read {}. In core {},{} {} at run {}, index {}",
                                    riscEndIndex,
                                    riscNumRead,
                                    worker_core.x,
                                    worker_core.y,
                                    enchantum::to_string(CoreType),
                                    runHostCounterRead,
                                    index);
                                TT_ASSERT(
                                    coreFlatIDRead == coreFlatID,
                                    "Unexpected core id, expected {}, read {}. In core {},{} {} at run {}, index {}",
                                    coreFlatID,
                                    coreFlatIDRead,
                                    worker_core.x,
                                    worker_core.y,
                                    enchantum::to_string(CoreType),
                                    runHostCounterRead,
                                    index);

                                readDeviceMarkerData(
                                    device_markers_for_core_risc,
                                    runHostCounterRead,
                                    opname,
                                    device_id,
                                    phys_coord,
                                    riscType,
                                    0,
                                    timer_id,
                                    (uint64_t(time_H) << 32) | time_L);
                            }
                        } break;
                        case kernel_profiler::ZONE_TOTAL: {
                            uint32_t sum = data_buffer.at(index + 1);

                            uint32_t time_H = opTime_H;
                            uint32_t time_L = opTime_L;
                            readDeviceMarkerData(
                                device_markers_for_core_risc,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                riscType,
                                sum,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);
                            break;
                        }
                        case kernel_profiler::TS_DATA: {
                            uint32_t time_H = data_buffer.at(index) & 0xFFF;
                            uint32_t time_L = data_buffer.at(index + 1);
                            index += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
                            uint32_t data_H = data_buffer.at(index);
                            uint32_t data_L = data_buffer.at(index + 1);
                            readDeviceMarkerData(
                                device_markers_for_core_risc,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                riscType,
                                (uint64_t(data_H) << 32) | data_L,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);
                            continue;
                        }
                        case kernel_profiler::TS_EVENT: {
                            uint32_t time_H = data_buffer.at(index) & 0xFFF;
                            uint32_t time_L = data_buffer.at(index + 1);
                            readDeviceMarkerData(
                                device_markers_for_core_risc,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                riscType,
                                0,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);
                        }
                    }
                }
            }
        }
    }
}

void DeviceProfiler::updateFirstTimestamp(uint64_t timestamp) {
    if (timestamp < smallest_timestamp) {
        smallest_timestamp = timestamp;
    }
}

tracy::MarkerDetails DeviceProfiler::getMarkerDetails(uint16_t timer_id) const {
    auto marker_details_iter = hash_to_zone_src_locations.find(timer_id);
    if (marker_details_iter != hash_to_zone_src_locations.end()) {
        return marker_details_iter->second;
    } else {
        return tracy::UnidentifiedMarkerDetails;
    }
}

void DeviceProfiler::readDeviceMarkerData(
    std::set<tracy::TTDeviceMarker>& device_markers,
    uint32_t run_host_id,
    const std::string& op_name,
    chip_id_t device_id,
    const CoreCoord& physical_core,
    tracy::RiscType risc_type,
    uint64_t data,
    uint32_t timer_id,
    uint64_t timestamp) {
    ZoneScoped;

    nlohmann::json meta_data;
    const tracy::MarkerDetails marker_details = getMarkerDetails(timer_id);
    const kernel_profiler::PacketTypes packet_type = get_packet_type(timer_id);

    const auto& [_, new_marker_inserted] = device_markers.emplace(
        run_host_id,
        device_id,
        physical_core.x,
        physical_core.y,
        risc_type,
        timer_id,
        timestamp,
        data,
        op_name,
        marker_details.source_line_num,
        marker_details.source_file,
        marker_details.marker_name,
        get_marker_type_from_packet_type(packet_type),
        marker_details.marker_name_keyword_flags,
        meta_data);

    if (!new_marker_inserted) {
        return;
    }

    device_tracy_contexts.try_emplace({device_id, physical_core}, nullptr);

    updateFirstTimestamp(timestamp);
}

struct DispatchMetaData {
    // Dispatch command queue command type
    std::string cmd_type = "";

    // Worker's runtime id
    uint32_t worker_runtime_id = 0;

    // dispatch command subtype.
    std::string cmd_subtype = "";
};

void DeviceProfiler::processDeviceMarkerData(std::set<tracy::TTDeviceMarker>& device_markers) {
    DispatchMetaData current_dispatch_meta_data;
    std::stack<std::set<tracy::TTDeviceMarker>::iterator> start_marker_stack;

    auto updateDeviceMarker = [&](const tracy::TTDeviceMarker& updated_marker,
                                  const std::set<tracy::TTDeviceMarker>::iterator& original_marker_it)
        -> std::pair<std::set<tracy::TTDeviceMarker>::iterator, std::set<tracy::TTDeviceMarker>::iterator> {
        const auto& next_device_marker_it = device_markers.erase(original_marker_it);
        const auto& [device_marker_it, _] = device_markers.insert(updated_marker);
        TT_ASSERT(std::next(device_marker_it) == next_device_marker_it);
        return {device_marker_it, next_device_marker_it};
    };

    auto device_marker_it = device_markers.begin();
    while (device_marker_it != device_markers.end()) {
        tracy::TTDeviceMarker marker = *device_marker_it;
        tracy::MarkerDetails marker_details = this->getMarkerDetails(marker.marker_id);

        auto next_device_marker_it = std::next(device_marker_it);

        if (isMarkerAZoneEndpoint(marker)) {
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_trace_only() &&
                marker.risc == tracy::RiscType::CORE_AGG) {
                if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::BRISC_FW)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::NCRISC_FW)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::TRISC_FW)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::ERISC_FW)]) {
                    marker.marker_name = "TRACE-FW";
                    const auto& ret = updateDeviceMarker(marker, device_marker_it);
                    device_marker_it = ret.first;
                    next_device_marker_it = ret.second;
                }
                if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::BRISC_KERNEL)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::NCRISC_KERNEL)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::TRISC_KERNEL)] ||
                    marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::ERISC_KERNEL)]) {
                    marker.marker_name = "TRACE-KERNEL";
                    const auto& ret = updateDeviceMarker(marker, device_marker_it);
                    device_marker_it = ret.first;
                    next_device_marker_it = ret.second;
                }
            }

            // Reset the command subtype, in case it isn't set during the command.
            current_dispatch_meta_data.cmd_subtype = "";

            if (marker.marker_type == tracy::TTDeviceMarkerType::ZONE_START) {
                start_marker_stack.push(device_marker_it);
            } else if (marker.marker_type == tracy::TTDeviceMarkerType::ZONE_END) {
                TT_FATAL(
                    !start_marker_stack.empty(),
                    "End marker {} found without a corresponding start marker",
                    marker.marker_id);

                const auto& start_marker_it = start_marker_stack.top();

                if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_trace_only()) {
                    TT_FATAL(
                        start_marker_it->marker_id == marker.marker_id,
                        "Start {} and end {} markers do not match",
                        start_marker_it->marker_id,
                        marker.marker_id);

                    if (start_marker_it->marker_name != marker.marker_name) {
                        marker.marker_name = start_marker_it->marker_name;
                        const auto& ret = updateDeviceMarker(marker, device_marker_it);
                        device_marker_it = ret.first;
                        next_device_marker_it = ret.second;
                    }
                } else {
                    TT_FATAL(
                        start_marker_it->marker_name == marker.marker_name,
                        "Start {} and end {} marker names do not match",
                        start_marker_it->marker_name,
                        marker.marker_name);
                }
                start_marker_stack.pop();
            }
        } else if (isMarkerATimestampedDatapoint(marker)) {
            if (!start_marker_stack.empty()) {
                auto curr_zone_start_marker_it = start_marker_stack.top();
                TT_ASSERT(curr_zone_start_marker_it->marker_type == tracy::TTDeviceMarkerType::ZONE_START);

                // Check if we are in a Tensix Dispatch zone. If so, we could have gotten dispatch meta data packets
                // These packets can amend parent zone's info
                const tracy::MarkerDetails curr_zone_start_marker_details =
                    getMarkerDetails(curr_zone_start_marker_it->marker_id);
                if ((marker.risc == tracy::RiscType::BRISC || marker.risc == tracy::RiscType::NCRISC) &&
                    curr_zone_start_marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                        tracy::MarkerDetails::MarkerNameKeyword::DISPATCH)]) {
                    if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                            tracy::MarkerDetails::MarkerNameKeyword::PROCESS_CMD)]) {
                        current_dispatch_meta_data.cmd_type =
                            fmt::format("{}", enchantum::to_string((CQDispatchCmdId)marker.data));
                        marker.meta_data["dispatch_command_type"] = current_dispatch_meta_data.cmd_type;
                    } else if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                                   tracy::MarkerDetails::MarkerNameKeyword::RUNTIME_HOST_ID_DISPATCH)]) {
                        current_dispatch_meta_data.worker_runtime_id = (uint32_t)marker.data;
                        marker.meta_data["workers_runtime_id"] = current_dispatch_meta_data.worker_runtime_id;
                    } else if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                                   tracy::MarkerDetails::MarkerNameKeyword::PACKED_DATA_DISPATCH)]) {
                        current_dispatch_meta_data.cmd_subtype = fmt::format(
                            "{}{}",
                            marker.data & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST ? "MCAST;" : "",
                            enchantum::to_string(static_cast<CQDispatchCmdPackedWriteType>(
                                (marker.data >> 1) << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT)));
                        marker.meta_data["dispatch_command_subtype"] = current_dispatch_meta_data.cmd_subtype;
                    } else if (marker_details.marker_name_keyword_flags[static_cast<uint16_t>(
                                   tracy::MarkerDetails::MarkerNameKeyword::PACKED_LARGE_DATA_DISPATCH)]) {
                        current_dispatch_meta_data.cmd_subtype = fmt::format(
                            "{}", enchantum::to_string(static_cast<CQDispatchCmdPackedWriteLargeType>(marker.data)));
                        marker.meta_data["dispatch_command_subtype"] = current_dispatch_meta_data.cmd_subtype;
                    }

                    std::string new_marker_name = current_dispatch_meta_data.cmd_type;
                    if (marker.risc == tracy::RiscType::BRISC) {
                        if (current_dispatch_meta_data.cmd_subtype != "") {
                            new_marker_name = fmt::format(
                                "{}:{}",
                                current_dispatch_meta_data.worker_runtime_id,
                                current_dispatch_meta_data.cmd_subtype);
                        } else {
                            new_marker_name = fmt::format(
                                "{}:{}",
                                current_dispatch_meta_data.worker_runtime_id,
                                current_dispatch_meta_data.cmd_type);
                        }
                    }

                    const auto& marker_ret = updateDeviceMarker(marker, device_marker_it);
                    device_marker_it = marker_ret.first;
                    next_device_marker_it = marker_ret.second;

                    tracy::TTDeviceMarker curr_zone_start_marker = *curr_zone_start_marker_it;
                    curr_zone_start_marker.runtime_host_id = current_dispatch_meta_data.worker_runtime_id;
                    curr_zone_start_marker.marker_name = curr_zone_start_marker.marker_name + ":" + new_marker_name;
                    const auto& curr_zone_start_marker_ret =
                        updateDeviceMarker(curr_zone_start_marker, curr_zone_start_marker_it);
                    curr_zone_start_marker_it = curr_zone_start_marker_ret.first;

                    start_marker_stack.pop();
                    start_marker_stack.push(curr_zone_start_marker_it);
                }
            }
        }

        device_marker_it = next_device_marker_it;
    }

    TT_FATAL(
        start_marker_stack.empty(),
        "{} start markers detected without corresponding end markers",
        start_marker_stack.size());
}

void DeviceProfiler::setLastFDReadAsNotDone() { this->is_last_fd_read_done = false; }

void DeviceProfiler::setLastFDReadAsDone() { this->is_last_fd_read_done = true; }

bool DeviceProfiler::isLastFDReadDone() const { return this->is_last_fd_read_done; }

DeviceProfiler::DeviceProfiler(const IDevice* device, const bool new_logs) {
#if defined(TRACY_ENABLE)
    ZoneScopedC(tracy::Color::Green);
    this->device_id = device->id();
    this->device_arch = device->arch();
    this->device_core_frequency =
        tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(this->device_id);
    this->output_dir = std::filesystem::path(get_profiler_logs_dir());
    std::filesystem::create_directories(this->output_dir);
    std::filesystem::path log_path = this->output_dir / DEVICE_SIDE_LOG;

    if (new_logs) {
        std::filesystem::remove(log_path);
    }

    const std::string noc_events_report_path =
        tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_noc_events_report_path();
    if (!noc_events_report_path.empty()) {
        this->noc_trace_data_output_dir = std::filesystem::path(noc_events_report_path);
    } else {
        this->noc_trace_data_output_dir = this->output_dir;
    }

    this->is_last_fd_read_done = false;
    this->device_tracy_contexts.reserve(
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);
#endif
}

void DeviceProfiler::dumpDeviceResults(bool is_mid_run_dump) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    if (!this->thread_pool) {
        this->thread_pool =
            create_device_bound_thread_pool(tt::tt_metal::MetalContext::instance()
                                                .profiler_state_manager()
                                                ->calculate_optimal_num_threads_for_device_profiler_thread_pool());
    }

    initializeMissingTracyContexts(/*blocking=*/is_mid_run_dump);

    if (!is_mid_run_dump) {
        for (auto& [core, _] : this->device_markers_per_core_risc_map) {
            this->thread_pool->enqueue([this, core]() {
                for (auto& [risc_num, device_markers] : this->device_markers_per_core_risc_map[core]) {
                    processDeviceMarkerData(device_markers);
                }
            });
        }

        this->thread_pool->wait();
    }

    std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>> device_markers_vec =
        getSortedDeviceMarkersVector(this->device_markers_per_core_risc_map, *this->thread_pool);

    this->thread_pool->enqueue([this]() { writeDeviceResultsToFiles(); });
    pushTracyDeviceResults(device_markers_vec);

    this->thread_pool->wait();

    this->device_markers_per_core_risc_map.clear();
#endif
}

void DeviceProfiler::freshDeviceLog() {
#if defined(TRACY_ENABLE)
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::filesystem::remove(log_path);
#endif
}

void DeviceProfiler::setOutputDir(const std::string& new_output_dir) {
#if defined(TRACY_ENABLE)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void DeviceProfiler::readResults(
    IDevice* device,
    const std::vector<CoreCoord>& virtual_cores,
    const ProfilerReadState state,
    const ProfilerDataBufferSource data_source,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    const std::string zone_name = fmt::format(
        "{}-{}-{}-{}", "readResults", device_id, enchantum::to_string(state), enchantum::to_string(data_source));
    ZoneName(zone_name.c_str(), zone_name.size());

    hash_to_zone_src_locations = generateZoneSourceLocationsHashes();

    TT_ASSERT(doAllDispatchCoresComeAfterNonDispatchCores(device, virtual_cores));

    if (data_source == ProfilerDataBufferSource::DRAM) {
        readControlBuffers(device, virtual_cores);

        readProfilerBuffer(device);

        resetControlBuffers(device, virtual_cores);
    } else if (data_source == ProfilerDataBufferSource::L1) {
        readControlBuffers(device, virtual_cores);

        resetControlBuffers(device, virtual_cores);

        readL1DataBuffers(device, virtual_cores);
    } else {
        TT_ASSERT(data_source == ProfilerDataBufferSource::DRAM_AND_L1);
        readControlBuffers(device, virtual_cores);

        readProfilerBuffer(device);

        readL1DataBuffers(device, virtual_cores);

        resetControlBuffers(device, virtual_cores);
    }
#endif
}

void DeviceProfiler::processResults(
    IDevice* device,
    const std::vector<CoreCoord>& virtual_cores,
    const ProfilerReadState state,
    const ProfilerDataBufferSource data_source,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    const std::string zone_name = fmt::format(
        "{}-{}-{}-{}", "processResults", device_id, enchantum::to_string(state), enchantum::to_string(data_source));
    ZoneName(zone_name.c_str(), zone_name.size());

    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();

    if (rtoptions.get_profiler_noc_events_enabled()) {
        log_warning(
            tt::LogAlways, "Profiler NoC events are enabled; this can add 1-15% cycle overhead to typical operations!");
    }

    for (const auto& virtual_core : virtual_cores) {
        readRiscProfilerResults(device, virtual_core, data_source, metadata);
    }

    if (rtoptions.get_profiler_noc_events_enabled() &&
        (state == ProfilerReadState::NORMAL || state == ProfilerReadState::LAST_FD_READ)) {
        FabricRoutingLookup routing_lookup(device);

        std::unordered_set<tracy::TTDeviceMarker> new_device_markers;
        for (const auto& [core, risc_map] : device_markers_per_core_risc_map) {
            for (const auto& [risc, device_markers] : risc_map) {
                for (const tracy::TTDeviceMarker& marker : device_markers) {
                    if (noc_trace_markers_processed.find(marker) == noc_trace_markers_processed.end()) {
                        new_device_markers.insert(marker);
                        noc_trace_markers_processed.insert(marker);
                    }
                }
            }
        }

        const std::unordered_map<RuntimeID, nlohmann::json::array_t> processed_markers_by_op_name =
            convertNocTracePacketsToJson(new_device_markers, device_id, routing_lookup);
        noc_trace_data.push_back(std::move(processed_markers_by_op_name));
    }
#endif
}

void DeviceProfiler::dumpRoutingInfo() const {
    std::filesystem::create_directories(noc_trace_data_output_dir);
    if (!std::filesystem::is_directory(noc_trace_data_output_dir)) {
        log_error(
            tt::LogMetal,
            "Could not dump topology to '{}' because the directory path could not be created!",
            noc_trace_data_output_dir);
        return;
    }

    tt::tt_metal::dumpRoutingInfo(noc_trace_data_output_dir / "topology.json");
}

void DeviceProfiler::dumpClusterCoordinates() const {
    std::filesystem::create_directories(noc_trace_data_output_dir);
    if (!std::filesystem::is_directory(noc_trace_data_output_dir)) {
        log_error(
            tt::LogMetal,
            "Could not dump cluster coordinates to '{}' because the directory path could not be created!",
            noc_trace_data_output_dir);
        return;
    }

    tt::tt_metal::dumpClusterCoordinatesAsJson(noc_trace_data_output_dir / "cluster_coordinates.json");
}

bool isSyncInfoNewer(const SyncInfo& old_info, const SyncInfo& new_info) {
    return (
        (old_info.frequency == 0 && new_info.frequency != 0) ||
        (old_info.cpu_time < new_info.cpu_time &&
         ((old_info.device_time / old_info.frequency) < (new_info.device_time / new_info.frequency))));
}

void DeviceProfiler::writeDeviceResultsToFiles() const {
#if defined(TRACY_ENABLE)
    std::scoped_lock lock(tt::tt_metal::MetalContext::instance().profiler_state_manager()->file_write_mutex);

    const std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    dumpDeviceResultsToCSV(device_markers_per_core_risc_map, device_arch, device_core_frequency, log_path);

    if (!noc_trace_data.empty()) {
        dumpJsonNocTraces(noc_trace_data, device_id, noc_trace_data_output_dir);
    }
#endif
}

void DeviceProfiler::pushTracyDeviceResults(
    std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers_vec) {
#if defined(TRACY_ENABLE)
    // If this device is root, it may have new sync info updated with syncDeviceHost
    for (auto& [core, info] : device_core_sync_info) {
        if (isSyncInfoNewer(device_sync_info, info)) {
            setSyncInfo(info);
        }
    }

    updateTracyContexts(device_markers_vec);

    for (auto& marker_ref : device_markers_vec) {
        std::reference_wrapper<const tracy::TTDeviceMarker>& marker_to_push_ref = marker_ref;

        const tracy::TTDeviceMarker& orig_marker = marker_ref.get();
        tracy::TTDeviceMarker marker_with_adjusted_timestamp;
        const uint64_t adjusted_timestamp = orig_marker.timestamp * this->freq_scale + this->shift;
        if (adjusted_timestamp != orig_marker.timestamp) {
            marker_with_adjusted_timestamp = tracy::TTDeviceMarker(
                orig_marker.runtime_host_id,
                orig_marker.chip_id,
                orig_marker.core_x,
                orig_marker.core_y,
                orig_marker.risc,
                orig_marker.marker_id,
                adjusted_timestamp,
                orig_marker.data,
                orig_marker.op_name,
                orig_marker.line,
                orig_marker.file,
                orig_marker.marker_name,
                orig_marker.marker_type,
                orig_marker.marker_name_keyword_flags,
                orig_marker.meta_data);
            marker_to_push_ref = std::cref(marker_with_adjusted_timestamp);
        }

        const tracy::TTDeviceMarker& marker_to_push = marker_to_push_ref.get();
        std::pair<chip_id_t, CoreCoord> device_core = {
            marker_to_push.chip_id, (CoreCoord){marker_to_push.core_x, marker_to_push.core_y}};
        if (marker_to_push.marker_type == tracy::TTDeviceMarkerType::ZONE_START) {
            TracyTTPushStartMarker(device_tracy_contexts[device_core], marker_to_push);
        } else if (marker_to_push.marker_type == tracy::TTDeviceMarkerType::ZONE_END) {
            TracyTTPushEndMarker(device_tracy_contexts[device_core], marker_to_push);
        }
    }
#endif
}

void DeviceProfiler::setSyncInfo(const SyncInfo& sync_info) { device_sync_info = sync_info; }

void DeviceProfiler::initializeMissingTracyContexts(bool blocking) {
#if defined(TRACY_ENABLE)
    TT_ASSERT(this->thread_pool != nullptr);

    for (const auto& [device_core, _] : device_tracy_contexts) {
        if (device_tracy_contexts.at(device_core) == nullptr) {
            this->thread_pool->enqueue(
                [this, device_core]() { device_tracy_contexts.at(device_core) = TracyTTContext(); });
        }
    }

    if (blocking) {
        this->thread_pool->wait();
    }
#endif
}

void DeviceProfiler::updateTracyContexts(
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers_vec) {
#if defined(TRACY_ENABLE)
    std::unordered_set<std::pair<chip_id_t, CoreCoord>, pair_hash<chip_id_t, CoreCoord>> device_cores_to_update;
    device_cores_to_update.reserve(device_tracy_contexts.size());

    for (const auto& [device_core, _] : device_tracy_contexts) {
        TT_ASSERT(device_tracy_contexts.at(device_core) != nullptr);
        device_cores_to_update.insert(device_core);
    }

    // Tracy contexts must be updated in order of their first timestamps
    for (const auto& marker_ref : device_markers_vec) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        auto device_core_it = device_cores_to_update.find({marker.chip_id, {marker.core_x, marker.core_y}});
        if (device_core_it != device_cores_to_update.end()) {
            updateTracyContext(*device_core_it);
            device_cores_to_update.erase(device_core_it);
        }

        if (device_cores_to_update.empty()) {
            break;
        }
    }
#endif
}

void DeviceProfiler::updateTracyContext(const std::pair<chip_id_t, CoreCoord>& device_core) {
#if defined(TRACY_ENABLE)
    const chip_id_t device_id = device_core.first;
    const CoreCoord worker_core = device_core.second;

    if (core_sync_info.find(worker_core) == core_sync_info.end()) {
        const std::string tracyTTCtxName =
            fmt::format("Device: {}, Core ({},{})", device_id, worker_core.x, worker_core.y);

        double cpu_time = device_sync_info.cpu_time;
        double device_time = device_sync_info.device_time;
        double frequency = device_sync_info.frequency;

        if (frequency == 0) {
            cpu_time = TracyGetCpuTime();
            device_time = smallest_timestamp;
            frequency = device_core_frequency / 1000.0;
            device_sync_info = SyncInfo(cpu_time, device_time, frequency);
            log_debug(
                tt::LogMetal,
                "For device {}, core {},{} default frequency was used and its zones will be out of sync",
                device_id,
                worker_core.x,
                worker_core.y);
        } else {
            log_debug(
                tt::LogMetal,
                "Device {}, core {},{} sync info are, frequency {} GHz,  delay {} cycles and, sync point {} seconds",
                device_id,
                worker_core.x,
                worker_core.y,
                frequency,
                device_time,
                cpu_time);
        }

        TracyTTCtx tracyCtx = device_tracy_contexts.at(device_core);
        TT_ASSERT(tracyCtx != nullptr);

        TracyTTContextPopulate(tracyCtx, cpu_time, device_time, frequency);
        TracyTTContextName(tracyCtx, tracyTTCtxName.c_str(), tracyTTCtxName.size());

        core_sync_info.emplace(worker_core, SyncInfo(cpu_time, device_time, frequency));
    } else {
        // Update the existing tracy context for this device core
        if (isSyncInfoNewer(core_sync_info[worker_core], device_sync_info)) {
            core_sync_info[worker_core] = device_sync_info;
            double cpu_time = device_sync_info.cpu_time;
            double device_time = device_sync_info.device_time;
            double frequency = device_sync_info.frequency;
            TracyTTCtx tracyCtx = device_tracy_contexts.at(device_core);
            TT_ASSERT(tracyCtx != nullptr);
            TracyTTContextCalibrate(tracyCtx, cpu_time, device_time, frequency);
            log_debug(
                tt::LogMetal,
                "Device {}, core {},{} calibration info are, frequency {} GHz,  delay {} cycles and, sync point {} "
                "seconds",
                device_id,
                worker_core.x,
                worker_core.y,
                frequency,
                device_time,
                cpu_time);
        }
    }
#endif
}

void DeviceProfiler::destroyTracyContexts() {
#if defined(TRACY_ENABLE)
    TT_ASSERT(this->thread_pool != nullptr);

    for (const auto& [device_core, _] : device_tracy_contexts) {
        TT_ASSERT(device_tracy_contexts.at(device_core) != nullptr);
        this->thread_pool->enqueue([this, device_core]() { TracyTTDestroy(device_tracy_contexts.at(device_core)); });
    }

    this->thread_pool->wait();
#endif
}

bool getDeviceProfilerState() { return tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled(); }

}  // namespace tt_metal

}  // namespace tt
