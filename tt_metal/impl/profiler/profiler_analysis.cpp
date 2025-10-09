// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <type_traits>
#include <fstream>

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

bool matches_start_end_risc(tracy::RiscType risc_type, AnalysisRiscTypes config_risc_types) {
    return config_risc_types.find(risc_type) != config_risc_types.end();
}

bool matches_start_end_marker_type(tracy::TTDeviceMarkerType marker_type, AnalysisMarkerTypes config_marker_types) {
    return config_marker_types.find(marker_type) != config_marker_types.end();
}

bool matches_start_end_marker_name_keywords(
    const std::array<
        bool,
        static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
            tracy::MarkerDetails::MarkerNameKeyword::COUNT)>& marker_name_keywords,
    const AnalysisMarkerNameKeywords& config_marker_name_keywords) {
    if (config_marker_name_keywords.empty()) {
        return true;
    }

    bool match = false;
    for (tracy::MarkerDetails::MarkerNameKeyword keyword : config_marker_name_keywords) {
        match |=
            marker_name_keywords[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(keyword)];
    }
    return match;
}

bool matches_start_end_config(const tracy::TTDeviceMarker& marker, const AnalysisStartEndConfig& start_end_config) {
    return matches_start_end_risc(marker.risc, start_end_config.risc_types) &&
           matches_start_end_marker_type(marker.marker_type, start_end_config.marker_types) &&
           matches_start_end_marker_name_keywords(
               marker.marker_name_keyword_flags, start_end_config.marker_name_keywords);
}

AnalysisResults parse_duration(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    TT_FATAL(analysis_config.type == AnalysisType::OP_FIRST_TO_LAST_MARKER, "Unsupported analysis type");

    AnalysisResults analysis_results;
    std::unordered_map<OpId, AnalysisResults::SingleResult, OpIdHasher>& results_per_op_id =
        analysis_results.results_per_op_id;

    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        const OpId op_id = {marker.runtime_host_id, marker.trace_id, marker.trace_id_counter};
        auto [op_id_results_it, _] = results_per_op_id.try_emplace(op_id, AnalysisResults::INVALID_SINGLE_RESULT);

        if (matches_start_end_config(marker, analysis_config.start_config)) {
            if (op_id_results_it->second == AnalysisResults::INVALID_SINGLE_RESULT) {
                op_id_results_it->second.start_timestamp = marker.timestamp;
                op_id_results_it->second.start_marker = marker_ref.get();
            }
        }
        if (matches_start_end_config(marker, analysis_config.end_config)) {
            if (op_id_results_it->second != AnalysisResults::INVALID_SINGLE_RESULT) {
                op_id_results_it->second.end_timestamp = marker.timestamp;
                op_id_results_it->second.end_marker = marker_ref.get();
            }
        }
    }

    for (auto& [_, result] : results_per_op_id) {
        if (result != AnalysisResults::INVALID_SINGLE_RESULT) {
            TT_ASSERT(result.start_timestamp <= result.end_timestamp);
            result.duration = result.end_timestamp - result.start_timestamp;
        }
    }

    analysis_results.results_config = analysis_config.results_config;

    return analysis_results;
}

AnalysisResults generateAnalysisForDeviceMarkers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers) {
    TT_ASSERT(std::is_sorted(
        device_markers.begin(), device_markers.end(), [](const auto& a, const auto& b) { return a.get() < b.get(); }));
    TT_FATAL(analysis_config.dimension == AnalysisDimension::OP, "Analysis config dimension must be across ops");

    switch (analysis_config.type) {
        case AnalysisType::OP_FIRST_TO_LAST_MARKER: return parse_duration(analysis_config, device_markers);
        default: TT_THROW("Invalid analysis type");
    }
}

std::map<OpId, OpsPerfResults::SingleOpPerfResults::OpMetaData> getMetaDataForOps(
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    std::map<OpId, OpsPerfResults::SingleOpPerfResults::OpMetaData> op_id_to_meta_data;

    std::unordered_map<OpId, std::unordered_set<CoreCoord>, OpIdHasher> fw_cores_per_op_id;
    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        if (op_id_to_meta_data.find({marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}) ==
            op_id_to_meta_data.end()) {
            const Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const tt_ClusterDescriptor* cluster_desc = cluster.get_cluster_desc();
            const ARCH device_arch = cluster_desc->get_arch(marker.chip_id);

            const uint8_t num_hw_cqs =
                tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
            const DispatchCoreConfig& dispatch_core_config =
                tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
            const CoreCoord compute_grid_size =
                tt::get_compute_grid_size(marker.chip_id, num_hw_cqs, dispatch_core_config);
            const uint32_t num_available_worker_cores = compute_grid_size.x * compute_grid_size.y;

            op_id_to_meta_data[{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}] = {
                .device_id = marker.chip_id,
                .device_arch = device_arch,
                .op_name = marker.op_name,
                .num_fw_cores = 0,
                .num_available_worker_cores = num_available_worker_cores,
            };
        }

        auto& it = op_id_to_meta_data[{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}];

        TT_ASSERT(it.device_id == marker.chip_id);
        TT_ASSERT(it.op_name == marker.op_name);

        if (marker
                .marker_name_keyword_flags[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
                    tracy::MarkerDetails::MarkerNameKeyword::_FW)]) {
            fw_cores_per_op_id[{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}].emplace(
                marker.core_x, marker.core_y);
        }
    }

    for (const auto& [op_id, fw_cores] : fw_cores_per_op_id) {
        op_id_to_meta_data[op_id].num_fw_cores = fw_cores.size();
    }

    return op_id_to_meta_data;
}

OpsPerfResults generatePerfResultsForOps(
    const std::vector<AnalysisConfig>& analysis_configs,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    ThreadPool& thread_pool) {
    ZoneScoped;

    OpsPerfResults ops_perf_results;
    std::map<OpId, OpsPerfResults::SingleOpPerfResults>& op_id_to_perf_results = ops_perf_results.op_id_to_perf_results;
    std::vector<AnalysisResultsConfig>& analysis_results_configs = ops_perf_results.analysis_results_configs;

    uint32_t i = 0;
    std::vector<AnalysisResults> analysis_results(analysis_configs.size());
    analysis_results_configs.resize(analysis_configs.size());
    for (const auto& analysis_config : analysis_configs) {
        thread_pool.enqueue([&analysis_config, &analysis_results_configs, &device_markers, &analysis_results, i]() {
            analysis_results[i] = generateAnalysisForDeviceMarkers(analysis_config, device_markers);
            analysis_results_configs[i] = analysis_results[i].results_config;
        });
        i++;
    }

    const std::map<OpId, OpsPerfResults::SingleOpPerfResults::OpMetaData> ops_meta_data =
        getMetaDataForOps(device_markers);

    thread_pool.wait();

    for (const auto& [op_id, op_meta_data] : ops_meta_data) {
        op_id_to_perf_results[op_id].op_meta_data = op_meta_data;
        OpsPerfResults::SingleOpPerfResults& op_perf_results = op_id_to_perf_results[op_id];

        for (const AnalysisResults& analysis_result : analysis_results) {
            TT_ASSERT(analysis_result.results_per_op_id.find(op_id) != analysis_result.results_per_op_id.end());
            const AnalysisResults::SingleResult& single_result = analysis_result.results_per_op_id.at(op_id);
            op_perf_results.analysis_results.push_back(single_result);
        }
        TT_ASSERT(op_perf_results.analysis_results.size() == analysis_results_configs.size());
    }

    return ops_perf_results;
}

void writeOpsPerfResultsToCSV(const OpsPerfResults& ops_perf_results, const std::filesystem::path& report_path) {
    ZoneScoped;

    std::scoped_lock lock(tt::tt_metal::MetalContext::instance().profiler_state_manager()->ops_perf_report_write_mutex);

    std::map<OpId, std::string> results_string_per_op_id;

    for (const auto& [op_id, op_perf_results] : ops_perf_results.op_id_to_perf_results) {
        results_string_per_op_id[op_id] =
            std::to_string(op_id.runtime_id) + "," +
            (op_id.trace_id == tracy::TTDeviceMarker::INVALID_NUM ? "" : std::to_string(op_id.trace_id)) + "," +
            (op_id.trace_id_counter == tracy::TTDeviceMarker::INVALID_NUM ? ""
                                                                          : std::to_string(op_id.trace_id_counter)) +
            "," + std::to_string(op_perf_results.op_meta_data.device_id) + "," +
            arch_to_str(op_perf_results.op_meta_data.device_arch) + "," + op_perf_results.op_meta_data.op_name + "," +
            std::to_string(op_perf_results.op_meta_data.num_fw_cores) + "," +
            std::to_string(op_perf_results.op_meta_data.num_available_worker_cores);

        for (uint32_t i = 0; i < op_perf_results.analysis_results.size(); i++) {
            const AnalysisResults::SingleResult& analysis_result = op_perf_results.analysis_results[i];
            const AnalysisResultsConfig& analysis_result_config = ops_perf_results.analysis_results_configs[i];
            results_string_per_op_id[op_id] += "," + (analysis_result == AnalysisResults::INVALID_SINGLE_RESULT
                                                          ? ""
                                                          : std::to_string(analysis_result.duration));
            if (analysis_result_config.display_start_and_end_timestamps) {
                results_string_per_op_id[op_id] += "," +
                                                   (analysis_result == AnalysisResults::INVALID_SINGLE_RESULT
                                                        ? ""
                                                        : std::to_string(analysis_result.start_timestamp)) +
                                                   "," +
                                                   (analysis_result == AnalysisResults::INVALID_SINGLE_RESULT
                                                        ? ""
                                                        : std::to_string(analysis_result.end_timestamp));
                std::to_string(analysis_result.end_timestamp);
            }
        }
    }

    TT_ASSERT(std::filesystem::exists(report_path.parent_path()));
    TT_ASSERT(report_path.extension() == ".csv");

    std::ofstream log_file_ofs;
    if (std::filesystem::exists(report_path)) {
        log_file_ofs.open(report_path, std::ios_base::app);
    } else {
        log_file_ofs.open(report_path);

        std::string header_string =
            "GLOBAL CALL COUNT,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,DEVICE ID,DEVICE ARCH,OP NAME,CORE "
            "COUNT,AVAILABLE WORKER CORE COUNT";

        for (const auto& analysis_result_config : ops_perf_results.analysis_results_configs) {
            header_string += "," + analysis_result_config.analysis_name;
            if (analysis_result_config.display_start_and_end_timestamps) {
                TT_FATAL(
                    analysis_result_config.start_timestamp_header.has_value(), "Start timestamp header is not set");
                TT_FATAL(analysis_result_config.end_timestamp_header.has_value(), "End timestamp header is not set");
                header_string += "," + analysis_result_config.start_timestamp_header.value() + "," +
                                 analysis_result_config.end_timestamp_header.value();
            }
        }

        log_file_ofs << header_string << std::endl;
    }

    for (const auto& [_, results_string] : results_string_per_op_id) {
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
            std::unordered_set<tracy::RiscType> risc_types;
            for (const auto& risc : j["risc"]) {
                risc_types.insert(risc.get<tracy::RiscType>());
            }
            config.risc_types = risc_types;
        } else {
            config.risc_types = {j.at("risc").get<tracy::RiscType>()};
        }
    } else {
        config.risc_types = AnalysisRiscTypesAny;
    }

    if (j.contains("marker_type")) {
        if (j["marker_type"].is_array()) {
            std::unordered_set<tracy::TTDeviceMarkerType> marker_types;
            for (const auto& marker_type : j["marker_type"]) {
                marker_types.insert(marker_type.get<tracy::TTDeviceMarkerType>());
            }
            config.marker_types = marker_types;
        } else {
            config.marker_types = {j.at("marker_type").get<tracy::TTDeviceMarkerType>()};
        }
    } else {
        config.marker_types = AnalysisMarkerTypesAny;
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
