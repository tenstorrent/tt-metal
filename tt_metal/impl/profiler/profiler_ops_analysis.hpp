// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "profiler.hpp"

namespace tt::metal {

// "marker" should be single marker/datapoint/timepoint

enum class AnalysisType { FIRST_TO_LAST_MARKER, ADJACENT_MARKERS };

enum class AnalysisResultType { DURATION };

enum class AnalysisDimension { OP };

enum class AnalysisCoreType { TENSIX, ETHERNET, ANY };

enum class AnalysisRisc { BRISC, NCRISC, TRISC0, TRISC1, TRISC2, ERISC, ANY };

enum class AnalysisZonePhase { START, END, ANY };

struct AnalysisStartEndConfig {
    AnalysisCoreType core_type;
    AnalysisRisc risc;
    AnalysisZonePhase zone_phase;
    std::vector<tt_metal::ZoneDetails::ZoneNameKeyword> zone_name_keywords;
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
std::vector<int64_t> parse_operation_data_points(
    const AnalysisConfig& analysis_config, const std::vector<tt_metal::OperationDataPoint>& operation_data_points);

}  // namespace tt::metal
