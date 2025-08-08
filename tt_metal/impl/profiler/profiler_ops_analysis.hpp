// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "profiler.hpp"

namespace tt::metal {

enum class AnalysisType { FIRST_TO_LAST_OP };

enum class AnalysisResultType { DURATION };

enum class AnalysisDimension { OP };

enum class AnalysisCoreType { ANY };

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

std::vector<std::string> get_headers_for_analysis_config(const AnalysisConfig& analysis_config);

std::vector<int64_t> parse_operation_data_points(
    const AnalysisConfig& analysis_config, const std::vector<tt_metal::OperationDataPoint>& operation_data_points);

}  // namespace tt::metal
