// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <type_traits>

#include "assert.hpp"
#include "profiler_analysis.hpp"

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

std::unordered_map<std::uint32_t, std::vector<uint64_t>> parse_duration(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    TT_FATAL(analysis_config.type == AnalysisType::OP_FIRST_TO_LAST_MARKER, "Unsupported analysis type");

    if (markers.empty()) {
        return {};
    }

    std::unordered_map<std::uint32_t, std::vector<uint64_t>> durations_per_runtime_id;

    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        if (matches_start_end_config(marker, analysis_config.start_config)) {
            if (durations_per_runtime_id.find(marker.runtime_host_id) == durations_per_runtime_id.end()) {
                durations_per_runtime_id[marker.runtime_host_id] = {marker.timestamp, 0, 0};
            }
            // durations_per_runtime_id[marker.runtime_host_id][0] =
            //     std::min(durations_per_runtime_id[marker.runtime_host_id][0], marker.timestamp);
        }
        if (matches_start_end_config(marker, analysis_config.end_config)) {
            TT_ASSERT(durations_per_runtime_id.find(marker.runtime_host_id) != durations_per_runtime_id.end());
            durations_per_runtime_id[marker.runtime_host_id][1] = marker.timestamp;
            // durations_per_runtime_id[marker.runtime_host_id][1] =
            //     std::max(durations_per_runtime_id[marker.runtime_host_id][1], marker.timestamp);
        }
    }

    // uint64_t end_timestamp = 0;
    // for (auto it = markers.rbegin(); it != markers.rend(); ++it) {
    //     const tracy::TTDeviceMarker& marker = it->get();
    //     if (matches_start_end_config(marker, analysis_config.end_config)) {
    //         end_timestamp = marker.timestamp;
    //         break;
    //     }
    // }

    for (auto& [runtime_id, durations] : durations_per_runtime_id) {
        const uint64_t start_timestamp = durations[0];
        const uint64_t end_timestamp = durations[1];

        TT_ASSERT(start_timestamp < end_timestamp);
        const uint64_t duration = end_timestamp - start_timestamp;
        durations[2] = duration;

        log_info(tt::LogMetal, "Runtime ID: {}, Duration: {}", runtime_id, duration);
    }

    return durations_per_runtime_id;
}

std::unordered_map<std::uint32_t, std::vector<uint64_t>> parse_timeseries_markers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& timeseries_markers) {
    TT_ASSERT(std::is_sorted(timeseries_markers.begin(), timeseries_markers.end(), [](const auto& a, const auto& b) {
        return a.get() < b.get();
    }));
    TT_FATAL(analysis_config.dimension == AnalysisDimension::OP, "Analysis config dimension must be across ops");

    switch (analysis_config.result_type) {
        case AnalysisResultType::DURATION: return parse_duration(analysis_config, timeseries_markers);
        default: TT_THROW("Invalid analysis result type");
    }
}
}  // namespace tt_metal

}  // namespace tt
