// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/experimental/disaggregation/kv_chunk_address_table_protobuf.hpp"

#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <google/protobuf/text_format.h>

#include "protobuf/kv_chunk_address_table.pb.h"

namespace tt::tt_metal::experimental::disaggregation {

namespace detail {

::tt::disaggregation::proto::KvChunkAddressTable to_proto_message(const KvChunkAddressTable& table) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;

    // All configs ("groups") — authoritative. Mirror config 0 into the legacy
    // scalar fields so old single-config readers still work.
    for (uint32_t c = 0; c < table.num_configs(); c++) {
        const auto& cfg = table.config(c);
        auto* pb_cfg = pb.add_configs();
        pb_cfg->set_name(table.config_name(c));
        pb_cfg->set_num_layers(cfg.num_layers);
        pb_cfg->set_max_sequence_length(cfg.max_sequence_length);
        pb_cfg->set_num_slots(cfg.num_slots);
        pb_cfg->set_chunk_n_tokens(cfg.chunk_n_tokens);
        pb_cfg->set_chunk_size_bytes(cfg.chunk_size_bytes);
    }
    const auto& cfg0 = table.config(0);
    pb.set_num_layers(cfg0.num_layers);
    pb.set_max_sequence_length(cfg0.max_sequence_length);
    pb.set_num_slots(cfg0.num_slots);
    pb.set_chunk_n_tokens(cfg0.chunk_n_tokens);
    pb.set_chunk_size_bytes(cfg0.chunk_size_bytes);

    for (size_t i = 0; i < table.num_device_groups(); i++) {
        const auto& group = table.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)});
        auto* pb_group = pb.add_device_groups();
        for (const auto& fnid : group.fabric_node_ids) {
            auto* pb_fnid = pb_group->add_fabric_node_ids();
            pb_fnid->set_mesh_id(*fnid.mesh_id);
            pb_fnid->set_chip_id(fnid.chip_id);
        }
    }

    // Export host mappings, deduplicating across device groups.
    // Only hosts for nodes that appear in at least one device group are exported.
    std::unordered_set<tt::tt_fabric::FabricNodeId> exported_hosts;
    for (size_t i = 0; i < table.num_device_groups(); i++) {
        const auto& group = table.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)});
        for (const auto& fnid : group.fabric_node_ids) {
            if (table.has_host(fnid) && exported_hosts.insert(fnid).second) {
                auto* pb_host = pb.add_fabric_node_hosts();
                pb_host->set_mesh_id(*fnid.mesh_id);
                pb_host->set_chip_id(fnid.chip_id);
                pb_host->set_host_name(table.get_host(fnid));
            }
        }
    }

    for (uint32_t c = 0; c < table.num_configs(); c++) {
        const auto& cfg = table.config(c);
        for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                    const auto& loc = table.lookup(layer, pos, slot, c);
                    if (loc.noc_addr == 0 && loc.size_bytes == 0 && *loc.device_group_index == 0) {
                        continue;
                    }
                    auto* entry = pb.add_entries();
                    entry->set_slot(slot);
                    entry->set_layer(layer);
                    entry->set_position(pos);
                    entry->set_noc_addr(loc.noc_addr);
                    entry->set_size_bytes(loc.size_bytes);
                    entry->set_device_group_index(*loc.device_group_index);
                    entry->set_config_idx(c);
                }
            }
        }
    }

    return pb;
}

KvChunkAddressTable from_proto_message(const ::tt::disaggregation::proto::KvChunkAddressTable& pb) {
    // Reconstruct configs. `configs` (field 9) is authoritative when present;
    // otherwise fall back to the legacy single-config scalar fields. Entries are
    // placed by config NAME (idx_to_name) so they land correctly even if the map
    // constructor reassigns ids by sorted-key order.
    std::map<std::string, KvChunkAddressTableConfig> configs;
    std::vector<std::string> idx_to_name;
    if (pb.configs_size() > 0) {
        idx_to_name.reserve(pb.configs_size());
        for (const auto& pb_cfg : pb.configs()) {
            KvChunkAddressTableConfig cfg{
                .num_layers = pb_cfg.num_layers(),
                .max_sequence_length = pb_cfg.max_sequence_length(),
                .num_slots = pb_cfg.num_slots(),
                .chunk_n_tokens = pb_cfg.chunk_n_tokens(),
                .chunk_size_bytes = pb_cfg.chunk_size_bytes(),
            };
            if (!configs.emplace(pb_cfg.name(), cfg).second) {
                throw std::runtime_error("duplicate config name '" + pb_cfg.name() + "' in KvChunkAddressTable proto");
            }
            idx_to_name.push_back(pb_cfg.name());
        }
    } else {
        configs.emplace(
            "0",
            KvChunkAddressTableConfig{
                .num_layers = pb.num_layers(),
                .max_sequence_length = pb.max_sequence_length(),
                .num_slots = pb.num_slots(),
                .chunk_n_tokens = pb.chunk_n_tokens(),
                .chunk_size_bytes = pb.chunk_size_bytes(),
            });
        idx_to_name.push_back("0");
    }
    KvChunkAddressTable table(configs);

    for (const auto& pb_group : pb.device_groups()) {
        std::vector<tt::tt_fabric::FabricNodeId> fnids;
        fnids.reserve(pb_group.fabric_node_ids_size());
        for (const auto& pb_fnid : pb_group.fabric_node_ids()) {
            fnids.emplace_back(tt::tt_fabric::MeshId{pb_fnid.mesh_id()}, pb_fnid.chip_id());
        }
        table.add_device_group(std::move(fnids));
    }

    for (const auto& pb_host : pb.fabric_node_hosts()) {
        tt::tt_fabric::FabricNodeId fnid(tt::tt_fabric::MeshId{pb_host.mesh_id()}, pb_host.chip_id());
        table.set_fabric_node_host(fnid, pb_host.host_name());
    }

    for (const auto& entry : pb.entries()) {
        if (entry.config_idx() >= idx_to_name.size()) {
            throw std::runtime_error("entry config_idx out of range in KvChunkAddressTable proto");
        }
        KvCacheLocation loc{
            .noc_addr = entry.noc_addr(),
            .size_bytes = entry.size_bytes(),
            .device_group_index = DeviceGroupIndex{entry.device_group_index()},
        };
        table.set(entry.layer(), entry.position(), entry.slot(), loc, idx_to_name[entry.config_idx()]);
    }

    return table;
}

}  // namespace detail

// --- Binary wire format ---

std::string export_to_protobuf(const KvChunkAddressTable& table) {
    auto pb = detail::to_proto_message(table);
    std::string out;
    if (!pb.SerializeToString(&out)) {
        throw std::runtime_error("Failed to serialize KvChunkAddressTable to protobuf");
    }
    return out;
}

void export_to_protobuf_file(const KvChunkAddressTable& table, const std::string& path) {
    std::string data = export_to_protobuf(table);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

KvChunkAddressTable import_from_protobuf(const std::string& data) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    if (!pb.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse protobuf data as KvChunkAddressTable");
    }
    return detail::from_proto_message(pb);
}

KvChunkAddressTable import_from_protobuf_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return import_from_protobuf(data);
}

// --- Text format (debug only) ---

std::string export_to_protobuf_text(const KvChunkAddressTable& table) {
    auto pb = detail::to_proto_message(table);
    std::string out;
    if (!google::protobuf::TextFormat::PrintToString(pb, &out)) {
        throw std::runtime_error("Failed to serialize KvChunkAddressTable to protobuf text format");
    }
    return out;
}

void export_to_protobuf_text_file(const KvChunkAddressTable& table, const std::string& path) {
    std::string text = export_to_protobuf_text(table);
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    out << text;
}

KvChunkAddressTable import_from_protobuf_text(const std::string& text) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    if (!google::protobuf::TextFormat::ParseFromString(text, &pb)) {
        throw std::runtime_error("Failed to parse protobuf text format as KvChunkAddressTable");
    }
    return detail::from_proto_message(pb);
}

KvChunkAddressTable import_from_protobuf_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return import_from_protobuf_text(text);
}

}  // namespace tt::tt_metal::experimental::disaggregation
