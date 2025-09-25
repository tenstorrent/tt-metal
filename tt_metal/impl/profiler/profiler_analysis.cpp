// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <tt-logger/tt-logger.hpp>
#include <type_traits>
#include <fstream>

#include "assert.hpp"
#include "core_coord.hpp"
#include "impl/context/metal_context.hpp"
#include "profiler_analysis.hpp"
#include "profiler_state_manager.hpp"

namespace tracy {
NLOHMANN_JSON_SERIALIZE_ENUM(
    TTDeviceMarkerType,
    {{TTDeviceMarkerType::ZONE_START, "ZONE_START"},
     {TTDeviceMarkerType::ZONE_END, "ZONE_END"},
     {TTDeviceMarkerType::ZONE_TOTAL, "ZONE_TOTAL"},
     {TTDeviceMarkerType::TS_DATA, "TS_DATA"},
     {TTDeviceMarkerType::TS_EVENT, "TS_EVENT"}});

NLOHMANN_JSON_SERIALIZE_ENUM(
    RiscType,
    {{RiscType::BRISC, "BRISC"},
     {RiscType::NCRISC, "NCRISC"},
     {RiscType::TRISC_0, "TRISC_0"},
     {RiscType::TRISC_1, "TRISC_1"},
     {RiscType::TRISC_2, "TRISC_2"},
     {RiscType::TENSIX_RISC_AGG, "TENSIX_RISC_AGG"},
     {RiscType::ERISC, "ERISC"}});
}  // namespace tracy

namespace tt {

namespace tt_metal {

bool matches_start_end_risc(AnalysisRisc marker_risc, AnalysisRisc config_risc) {
    return static_cast<std::underlying_type_t<AnalysisRisc>>(config_risc & marker_risc) != 0;
}

bool matches_start_end_marker_type(AnalysisMarkerType marker_type, AnalysisMarkerType config_marker_type) {
    return static_cast<std::underlying_type_t<AnalysisMarkerType>>(config_marker_type & marker_type) != 0;
}

bool matches_start_end_marker_name_keywords(
    const std::array<
        bool,
        static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
            tracy::MarkerDetails::MarkerNameKeyword::COUNT)>& marker_name_keywords,
    const AnalysisMarkerNameKeywords& config_marker_name_keywords) {
    bool match = false;
    for (tracy::MarkerDetails::MarkerNameKeyword keyword : config_marker_name_keywords) {
        match |=
            marker_name_keywords[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(keyword)];
    }
    return match;
}

bool matches_start_end_config(const tracy::TTDeviceMarker& marker, const AnalysisStartEndConfig& start_end_config) {
    return matches_start_end_risc(marker.risc, start_end_config.risc) &&
           matches_start_end_marker_type(marker.marker_type, start_end_config.marker_type) &&
           matches_start_end_marker_name_keywords(
               marker.marker_name_keyword_flags, start_end_config.marker_name_keywords);
}

DurationAnalysisResults parse_duration(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    TT_FATAL(analysis_config.type == AnalysisType::OP_FIRST_TO_LAST_MARKER, "Unsupported analysis type");

    std::unordered_map<uint64_t, DurationAnalysisResults::SingleResult> results_per_runtime_id;
    std::unordered_map<uint64_t, std::unordered_set<CoreCoord>> fw_cores_per_runtime_id;
    std::unordered_map<uint64_t, bool> start_end_timestamp_pair_found_per_runtime_id;

    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        if (matches_start_end_config(marker, analysis_config.start_config)) {
            if (results_per_runtime_id.find(marker.runtime_host_id) == results_per_runtime_id.end()) {
                results_per_runtime_id[marker.runtime_host_id].start_timestamp = marker.timestamp;
                results_per_runtime_id[marker.runtime_host_id].start_marker = marker_ref.get();
                start_end_timestamp_pair_found_per_runtime_id[marker.runtime_host_id] = false;
            } else if (start_end_timestamp_pair_found_per_runtime_id[marker.runtime_host_id]) {
                // log_warning(tt::LogMetal, "");
            }
        }
        if (matches_start_end_config(marker, analysis_config.end_config)) {
            if (results_per_runtime_id.find(marker.runtime_host_id) != results_per_runtime_id.end()) {
                results_per_runtime_id[marker.runtime_host_id].end_timestamp = marker.timestamp;
                results_per_runtime_id[marker.runtime_host_id].end_marker = marker_ref.get();
                start_end_timestamp_pair_found_per_runtime_id[marker.runtime_host_id] = true;
            }
        }

        if (marker
                .marker_name_keyword_flags[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
                    tracy::MarkerDetails::MarkerNameKeyword::_FW)]) {
            fw_cores_per_runtime_id[marker.runtime_host_id].emplace(marker.core_x, marker.core_y);
        }
    }

    DurationAnalysisResults duration_analysis_results;
    for (auto& [runtime_id, result] : results_per_runtime_id) {
        TT_ASSERT(result.start_timestamp <= result.end_timestamp);

        result.duration = result.end_timestamp - result.start_timestamp;
        duration_analysis_results.addResultsForRuntimeId(runtime_id, result);

        TT_ASSERT(result.start_marker.chip_id == result.end_marker.chip_id);
        TT_ASSERT(result.start_marker.op_name == result.end_marker.op_name);
        TT_ASSERT(fw_cores_per_runtime_id.find(runtime_id) != fw_cores_per_runtime_id.end());

        const Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const tt_ClusterDescriptor* cluster_desc = cluster.get_cluster_desc();
        const ARCH device_arch = cluster_desc->get_arch(result.start_marker.chip_id);

        const uint8_t num_hw_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
        const DispatchCoreConfig& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
        const CoreCoord compute_grid_size =
            tt::get_compute_grid_size(result.start_marker.chip_id, num_hw_cqs, dispatch_core_config);
        const uint32_t num_available_worker_cores = compute_grid_size.x * compute_grid_size.y;

        duration_analysis_results.addMetaDataForRuntimeId(
            runtime_id,
            {.device_id = result.start_marker.chip_id,
             .device_arch = device_arch,
             .op_name = result.start_marker.op_name,
             .num_fw_cores = fw_cores_per_runtime_id[runtime_id].size(),
             .num_available_worker_cores = num_available_worker_cores});
    }

    duration_analysis_results.results_config = analysis_config.results_config;

    return duration_analysis_results;
}

std::unique_ptr<AnalysisResults> generateAnalysisForDeviceMarkers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers) {
    TT_ASSERT(std::is_sorted(
        device_markers.begin(), device_markers.end(), [](const auto& a, const auto& b) { return a.get() < b.get(); }));
    TT_FATAL(analysis_config.dimension == AnalysisDimension::OP, "Analysis config dimension must be across ops");

    switch (analysis_config.type) {
        case AnalysisType::OP_FIRST_TO_LAST_MARKER:
            return std::make_unique<DurationAnalysisResults>(parse_duration(analysis_config, device_markers));
        default: TT_THROW("Invalid analysis type");
    }
}

void writeAnalysisResultsToCSV(
    const std::vector<std::unique_ptr<const AnalysisResults>>& analysis_results,
    const std::filesystem::path& report_path) {
    std::scoped_lock lock(tt::tt_metal::MetalContext::instance().profiler_state_manager()->perf_ops_report_write_mutex);

    std::string header_string =
        "GLOBAL CALL COUNT,DEVICE ID,DEVICE ARCH,OP NAME,CORE COUNT,AVAILABLE WORKER CORE COUNT";

    for (const auto& analysis_result : analysis_results) {
        header_string += "," + analysis_result->getStringifiedHeaders();
    }

    std::map<uint64_t, std::string> results_string_per_runtime_id;

    for (const auto& analysis_result : analysis_results) {
        for (const uint64_t runtime_id : analysis_result->getRuntimeIds()) {
            if (results_string_per_runtime_id.find(runtime_id) == results_string_per_runtime_id.end()) {
                const AnalysisResults::RuntimeIdMetaData meta_data =
                    analysis_result->getMetaDataForRuntimeId(runtime_id);
                results_string_per_runtime_id[runtime_id] =
                    std::to_string(runtime_id) + "," + std::to_string(meta_data.device_id) + "," +
                    arch_to_str(meta_data.device_arch) + "," + meta_data.op_name + "," +
                    std::to_string(meta_data.num_fw_cores) + "," + std::to_string(meta_data.num_available_worker_cores);
            }
        }
    }

    for (const auto& analysis_result : analysis_results) {
        for (const auto& [runtime_id, results_string] : results_string_per_runtime_id) {
            results_string_per_runtime_id[runtime_id] +=
                "," + analysis_result->getStringifiedResultsForRuntimeId(runtime_id);
        }
    }

    TT_ASSERT(std::filesystem::exists(report_path.parent_path()));
    TT_ASSERT(report_path.extension() == ".csv");

    std::ofstream log_file_ofs;
    if (std::filesystem::exists(report_path)) {
        log_file_ofs.open(report_path, std::ios_base::app);
    } else {
        log_file_ofs.open(report_path);
        log_file_ofs << header_string << std::endl;
    }

    for (const auto& [_, results_string] : results_string_per_runtime_id) {
        log_file_ofs << results_string << std::endl;
    }

    log_file_ofs.close();
}

NLOHMANN_JSON_SERIALIZE_ENUM(AnalysisType, {{AnalysisType::OP_FIRST_TO_LAST_MARKER, "OP_FIRST_TO_LAST_MARKER"}});
NLOHMANN_JSON_SERIALIZE_ENUM(AnalysisDimension, {{AnalysisDimension::OP, "OP"}});

void from_json(const nlohmann::json& j, AnalysisResultsConfig& config) {
    j.at("analysis_name").get_to(config.analysis_name);
    config.display_start_and_end_timestamps = j.value("display_start_and_end_timestamps", false);
    config.start_timestamp_header = j.contains("start_timestamp_header")
                                        ? std::make_optional(j.at("start_timestamp_header").get<std::string>())
                                        : std::nullopt;
    config.end_timestamp_header = j.contains("end_timestamp_header")
                                      ? std::make_optional(j.at("end_timestamp_header").get<std::string>())
                                      : std::nullopt;
}

void from_json(const nlohmann::json& j, AnalysisStartEndConfig& config) {
    if (j.contains("risc")) {
        if (j["risc"].is_array()) {
            uint8_t risc_val = 0;
            for (const auto& risc : j["risc"]) {
                risc_val |= static_cast<std::underlying_type_t<AnalysisRisc>>(risc.get<AnalysisRisc>());
            }
            config.risc = static_cast<AnalysisRisc>(risc_val);
        } else {
            j.at("risc").get_to(config.risc);
        }
    } else {
        config.risc = AnalysisRiscAny;
    }

    if (j.contains("marker_type")) {
        if (j["marker_type"].is_array()) {
            uint8_t marker_type_val = 0;
            for (const auto& marker_type : j["marker_type"]) {
                marker_type_val |=
                    static_cast<std::underlying_type_t<AnalysisMarkerType>>(marker_type.get<AnalysisMarkerType>());
            }
            config.marker_type = static_cast<AnalysisMarkerType>(marker_type_val);
        } else {
            j.at("marker_type").get_to(config.marker_type);
        }
    } else {
        config.marker_type = AnalysisMarkerTypeAny;
    }

    if (j.contains("marker_name_keywords")) {
        for (const auto& marker_name_keyword : j["marker_name_keywords"]) {
            config.marker_name_keywords.insert(
                tracy::MarkerDetails::marker_name_keywords_map.at(marker_name_keyword.get<std::string>()));
        }
    } else {
        config.marker_name_keywords = AnalysisMarkerNameKeywords();
    }
}

void from_json(const nlohmann::json& j, AnalysisConfig& config) {
    j.at("type").get_to(config.type);
    j.at("dimension").get_to(config.dimension);
    j.at("results_config").get_to(config.results_config);
    j.at("start_config").get_to(config.start_config);
    j.at("end_config").get_to(config.end_config);
}

std::vector<AnalysisConfig> loadAnalysisConfigsFromJSON(const std::filesystem::path& json_path) {
    TT_ASSERT(std::filesystem::exists(json_path));
    std::ifstream json_ifs(json_path);
    const nlohmann::json configs_json = nlohmann::json::parse(json_ifs);

    std::vector<AnalysisConfig> configs;
    for (const auto& config_json : configs_json) {
        configs.push_back(config_json.get<AnalysisConfig>());
    }
    return configs;
}
}  // namespace tt_metal

}  // namespace tt
