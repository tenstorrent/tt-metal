// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <type_traits>
#include <fstream>

#include "assert.hpp"
#include "profiler_analysis.hpp"
#include "profiler_paths.hpp"

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

    if (markers.empty()) {
        return {};
    }

    DurationAnalysisResults duration_analysis_results;
    std::unordered_map<uint64_t, DurationAnalysisResults::SingleResult> results_per_runtime_id;

    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        if (matches_start_end_config(marker, analysis_config.start_config)) {
            if (results_per_runtime_id.find(marker.runtime_host_id) == results_per_runtime_id.end()) {
                results_per_runtime_id[marker.runtime_host_id].start_timestamp = marker.timestamp;
                results_per_runtime_id[marker.runtime_host_id].start_marker = marker_ref.get();
            }
        }
        if (matches_start_end_config(marker, analysis_config.end_config)) {
            TT_ASSERT(results_per_runtime_id.find(marker.runtime_host_id) != results_per_runtime_id.end());
            results_per_runtime_id[marker.runtime_host_id].end_timestamp = marker.timestamp;
            results_per_runtime_id[marker.runtime_host_id].end_marker = marker_ref.get();
        }
    }

    for (auto& [runtime_id, result] : results_per_runtime_id) {
        TT_ASSERT(result.start_timestamp < result.end_timestamp);
        result.duration = result.end_timestamp - result.start_timestamp;
        duration_analysis_results.addResultsForRuntimeId(runtime_id, result);
    }

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
    const std::vector<const AnalysisResults*>& analysis_results,
    const std::vector<std::vector<std::string>>& analysis_results_header_names) {
    TT_ASSERT(analysis_results.size() == analysis_results_header_names.size());

    std::string header_string = "GLOBAL CALL COUNT,DEVICE ID,OP NAME";

    uint32_t header_idx = 0;
    for (const AnalysisResults* analysis_result : analysis_results) {
        TT_ASSERT(analysis_result->getNumFieldsPerResult() == analysis_results_header_names[header_idx].size());

        for (const std::string& header : analysis_results_header_names[header_idx]) {
            header_string += "," + header;
        }
        header_idx++;
    }

    std::map<uint64_t, std::string> results_string_per_runtime_id;
    for (const AnalysisResults* analysis_result : analysis_results) {
        for (const uint64_t runtime_id : analysis_result->getRuntimeIds()) {
            if (results_string_per_runtime_id.find(runtime_id) == results_string_per_runtime_id.end()) {
                const AnalysisResults::RuntimeIdMetaData& meta_data =
                    analysis_result->getMetaDataForRuntimeId(runtime_id);
                results_string_per_runtime_id[runtime_id] =
                    std::to_string(runtime_id) + "," + std::to_string(meta_data.device_id) + "," + meta_data.op_name;
            }

            // if (results_string_per_runtime_id.find(runtime_id) != results_string_per_runtime_id.end()) {
            //     results_string_per_runtime_id[runtime_id] += ",";
            // }

            results_string_per_runtime_id[runtime_id] +=
                "," + analysis_result->getStringifiedResultsForRuntimeId(runtime_id);
        }
    }

    log_info(tt::LogMetal, "Writing analysis results to CSV");

    std::filesystem::create_directories(get_profiler_reports_dir());

    std::ofstream log_file_ofs;
    if (std::filesystem::exists(PROFILER_OPS_PERF_RESULTS_LOG)) {
        log_info(tt::LogMetal, "Appending to existing file {}", PROFILER_OPS_PERF_RESULTS_LOG);
        log_file_ofs.open(PROFILER_OPS_PERF_RESULTS_LOG, std::ios_base::app);
    } else {
        log_info(tt::LogMetal, "Creating new file at {}", PROFILER_OPS_PERF_RESULTS_LOG);
        log_file_ofs.open(PROFILER_OPS_PERF_RESULTS_LOG);
        log_file_ofs << header_string << std::endl;
    }

    if (!log_file_ofs.is_open()) {
        log_error(tt::LogMetal, "Failed to open file {} for writing", PROFILER_OPS_PERF_RESULTS_LOG);
        return;
    }

    log_info(tt::LogMetal, "Writing {} results to CSV", results_string_per_runtime_id.size());

    for (const auto& [_, results_string] : results_string_per_runtime_id) {
        log_file_ofs << results_string << std::endl;
    }

    log_file_ofs.flush();
    log_file_ofs.close();

    log_info(tt::LogMetal, "Successfully wrote analysis results to {}", PROFILER_OPS_PERF_RESULTS_LOG);
}

// std::vector<std::string> get_duration_headers(const AnalysisConfig& analysis_config){
//     TT_ASSERT(analysis_config.type == AnalysisType::OP_FIRST_TO_LAST_MARKER, "Unsupported analysis type");

//     std::vector<std::string> headers;

//     if (analysis_config.)

//     if (analysis_config.start_config.marker_type == AnalysisMarkerType::ZONE_START &&
//         analysis_config.end_config.marker_type == AnalysisMarkerType::ZONE_END) {
//         headers.push_back("DURATION");
//     }

//     return headers;
// }

// std::vector<std::string> get_headers_for_analysis_config(const AnalysisConfig& analysis_config){
//     switch (analysis_config.result_type) {
//         case AnalysisResultType::DURATION: return get_duration_headers(analysis_config);
//         default: TT_THROW("Invalid analysis result type");
//     }
// }
}  // namespace tt_metal

}  // namespace tt
