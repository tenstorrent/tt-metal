// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/coordinate_translation.hpp"

#include "indestructible.hpp"
#include "tt_cluster.hpp"

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
            MeshCoordinate(mapping[0][0], mapping[0][1]),
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

}  // namespace

const std::pair<CoordinateTranslationMap, MeshShape>& get_system_mesh_coordinate_translation_map() {
    static tt::stl::Indestructible<std::pair<CoordinateTranslationMap, MeshShape>> kTranslationMap([]() {
        const auto system_num_devices = tt::Cluster::instance().number_of_user_devices();

        const bool is_qg = tt::Cluster::instance().number_of_pci_devices() == system_num_devices;

        // TODO: #17477 - This assumes shapes and coordinates are in 2D. This will be extended for 3D.
        // Consider if 1D can be used for single device and N300.
        const std::unordered_map<size_t, std::pair<std::string, MeshShape>> system_mesh_translation_map = {
            {1, std::make_pair("device.json", MeshShape(1, 1))},
            {2, std::make_pair("N300.json", MeshShape(1, 2))},
            {8, std::make_pair("T3000.json", MeshShape(2, 4))},
            {32, std::make_pair(is_qg ? "QG.json" : "TG.json", MeshShape(8, 4))},
            {64, std::make_pair("TGG.json", MeshShape(8, 8))},
        };
        TT_FATAL(
            system_mesh_translation_map.contains(system_num_devices),
            "Unsupported number of devices: {}",
            system_num_devices);

        const auto [translation_config_file, shape] = system_mesh_translation_map.at(system_num_devices);
        TT_FATAL(
            system_num_devices == shape.mesh_size(),
            "Mismatch between number of devices and the mesh shape: {} != {}",
            system_num_devices,
            shape.mesh_size());
        log_debug(LogMetal, "Logical SystemMesh Shape: {}", shape);

        return std::pair<CoordinateTranslationMap, MeshShape>{
            load_translation_map(get_config_path(translation_config_file), /*key=*/"logical_to_physical_coordinates"),
            shape};
    }());

    return kTranslationMap.get();
}

}  // namespace tt::tt_metal::distributed
