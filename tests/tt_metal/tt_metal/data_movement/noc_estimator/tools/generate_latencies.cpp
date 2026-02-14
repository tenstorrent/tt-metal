// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../offline/csv_reader.hpp"
#include "../offline/data_processor.hpp"
#include "../offline/data_extractor.hpp"
#include "../offline/yaml_serializer.hpp"
#include "../common/types.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::string> find_all_csvs(const std::string& data_dir) {
    std::vector<std::string> csv_paths;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(data_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csv_paths.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }

    return csv_paths;
}

bool generate_yaml_from_csvs(const std::vector<std::string>& csv_paths, const std::string& output_path) {
    std::vector<tt::noc_estimator::offline::DataPoint> all_points;

    for (const auto& path : csv_paths) {
        tt::noc_estimator::offline::CsvReader reader;
        if (!reader.load_csv(path)) {
            std::cerr << "Failed to load " << path << std::endl;
            continue;
        }
        std::cout << "Loaded: " << path << " (" << reader.get_data_points().size() << " points)\n";

        const auto& points = reader.get_data_points();
        all_points.insert(all_points.end(), points.begin(), points.end());
    }

    if (all_points.empty()) {
        std::cerr << "No data points loaded" << std::endl;
        return false;
    }

    auto groups = tt::noc_estimator::offline::group_by_parameters(all_points);
    std::cout << "\nExtracting latencies for " << groups.size() << " groups...\n";

    std::map<tt::noc_estimator::common::GroupKey, tt::noc_estimator::common::LatencyData> entries;
    for (const auto& [key, points] : groups) {
        auto latency_data = tt::noc_estimator::offline::extract_latencies(points);
        entries[key] = latency_data;
    }

    if (!tt::noc_estimator::offline::save_latency_data_to_yaml(entries, output_path)) {
        return false;
    }

    std::cout << "\nGenerated YAML: " << output_path << "\n";
    std::cout << "Total entries: " << entries.size() << "\n";

    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_directory> <output_yaml>\n";
        std::cerr << "Example: " << argv[0]
                  << " tests/tt_metal/tt_metal/data_movement/data "
                     "tests/tt_metal/tt_metal/data_movement/noc_estimator/noc_latencies.yaml\n";
        return 1;
    }

    std::string data_dir = argv[1];
    std::string output_path = argv[2];

    std::vector<std::string> csv_paths = find_all_csvs(data_dir);

    if (csv_paths.empty()) {
        std::cerr << "No CSV files found in: " << data_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << csv_paths.size() << " CSV files" << std::endl;

    if (!generate_yaml_from_csvs(csv_paths, output_path)) {
        return 1;
    }

    std::cout << "\nSuccess" << std::endl;
    return 0;
}
