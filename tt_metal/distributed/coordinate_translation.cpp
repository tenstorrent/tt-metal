// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/coordinate_translation.hpp"

#include <nlohmann/json.hpp>

namespace tt::tt_metal::distributed {

namespace {

std::string get_config_path(const std::string& filename) {
    std::string root_path = getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : "./";
    return root_path + "/tt_metal/distributed/config/" + filename;
}

CoordinateTranslationMap load_translation_map(const std::string& filename, const std::string& key) {
    std::ifstream file(filename);
    TT_FATAL(file.is_open(), "Unable to open file: {}", filename);

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        TT_THROW("JSON parsing error in file {}: {}", filename, e.what());
    }

    TT_FATAL(j.contains(key), "Key '{}' not found in JSON file: {}", key, filename);

    CoordinateTranslationMap result;
    for (const auto& mapping : j[key]) {
        if (mapping.size() != 2 || mapping[0].size() != 2 || mapping[1].size() != 5) {
            TT_THROW("Invalid coordinate format in JSON file: {}", filename);
        }
        result.emplace(
            Coordinate{mapping[0][0], mapping[0][1]},
            PhysicalCoordinate{
                mapping[1][0],  // cluster_id
                mapping[1][2],  // x
                mapping[1][1],  // y
                mapping[1][3],  // rack
                mapping[1][4]   // shelf
            });
    }

    return result;
}

MeshShape get_system_mesh_shape(size_t system_num_devices) {
    static const std::unordered_map<size_t, MeshShape> system_mesh_to_shape = {
        {1, MeshShape{1, 1}},   // single-device
        {2, MeshShape{1, 2}},   // N300
        {8, MeshShape{2, 4}},   // T3000; as ring to match existing tests
        {32, MeshShape{8, 4}},  // TG, QG
        {64, MeshShape{8, 8}},  // TGG
    };
    TT_FATAL(
        system_mesh_to_shape.contains(system_num_devices), "Unsupported number of devices: {}", system_num_devices);
    auto shape = system_mesh_to_shape.at(system_num_devices);
    log_debug(LogMetal, "Logical SystemMesh Shape: {}x{}", shape.num_rows, shape.num_cols);
    return shape;
}

}  // namespace

std::pair<CoordinateTranslationMap, MeshShape> get_system_mesh_coordinate_translation_map() {
    static const auto* cached_translation_map = new std::pair<CoordinateTranslationMap, MeshShape>([] {
        auto system_num_devices = tt::Cluster::instance().number_of_user_devices();

        std::string galaxy_mesh_descriptor = "TG.json";
        if (tt::Cluster::instance().number_of_pci_devices() == system_num_devices) {
            galaxy_mesh_descriptor = "QG.json";
        }

        const std::unordered_map<size_t, std::string> system_mesh_translation_map = {
            {1, "device.json"},
            {2, "N300.json"},
            {8, "T3000.json"},
            {32, galaxy_mesh_descriptor},
            {64, "TGG.json"},
        };

        TT_FATAL(
            system_mesh_translation_map.contains(system_num_devices),
            "Unsupported number of devices: {}",
            system_num_devices);

        auto translation_config_file = get_config_path(system_mesh_translation_map.at(system_num_devices));
        return std::pair<CoordinateTranslationMap, MeshShape>{
            load_translation_map(translation_config_file, "logical_to_physical_coordinates"),
            get_system_mesh_shape(system_num_devices)};
    }());

    return *cached_translation_map;
}

}  // namespace tt::tt_metal::distributed
