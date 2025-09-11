// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <array>
#include <unordered_set>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include <common/TracyTTDeviceData.hpp>

namespace tt {

namespace tt_metal {

// "marker" should be single marker/datapoint/timepoint

enum class AnalysisType { OP_FIRST_TO_LAST_MARKER, ADJACENT_MARKERS };

enum class AnalysisResultType { DURATION };

enum class AnalysisDimension { OP };

using AnalysisRisc = tracy::RiscType;
constexpr AnalysisRisc AnalysisRiscAny = AnalysisRisc::BRISC | AnalysisRisc::NCRISC | AnalysisRisc::TRISC_0 |
                                         AnalysisRisc::TRISC_1 | AnalysisRisc::TRISC_2 | AnalysisRisc::ERISC |
                                         AnalysisRisc::CORE_AGG;

using AnalysisMarkerType = tracy::TTDeviceMarkerType;
constexpr AnalysisMarkerType AnalysisMarkerTypeAny = AnalysisMarkerType::ZONE_START | AnalysisMarkerType::ZONE_END |
                                                     AnalysisMarkerType::ZONE_TOTAL | AnalysisMarkerType::TS_DATA |
                                                     AnalysisMarkerType::TS_EVENT;

using AnalysisMarkerNameKeywords = std::unordered_set<tracy::MarkerDetails::MarkerNameKeyword>;

struct AnalysisStartEndConfig {
    AnalysisRisc risc;
    AnalysisMarkerType marker_type;
    AnalysisMarkerNameKeywords marker_name_keywords;
};

struct AnalysisConfig {
    AnalysisType type;
    AnalysisResultType result_type;
    AnalysisDimension dimension;
    AnalysisStartEndConfig start_config;
    AnalysisStartEndConfig end_config;
};

// renaming headers could break other scripts
// look at process_ops_logs.py to see how to name headers
// have option to either autogen headers or manually specify headers
std::vector<std::string> get_headers_for_analysis_config(const AnalysisConfig& analysis_config);

// rename to parse_time_series_data_points
// convert second arg to class/struct with vector as part of it
// with struct/class, we can have caching
// look into sorting beforehand to avoid doing linear traversal
std::unordered_map<std::uint32_t, std::vector<uint64_t>> parse_timeseries_markers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& timeseries_markers);

}  // namespace tt_metal

}  // namespace tt
