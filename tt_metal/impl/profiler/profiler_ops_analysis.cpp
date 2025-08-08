// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "profiler_ops_analysis.hpp"

namespace tt::metal {

bool does_op_data_point_match_config(
    const tt_metal::OperationDataPoint& op_data_point, const AnalysisConfig& analysis_config){return op_data_point.&& }

std::vector<int64_t> parse_duration(
    const AnalysisConfig& analysis_config, const std::vector<tt_metal::OperationDataPoint>& operation_data_points) {
    TT_FATAL(
        analysis_config.type == AnalysisType::FIRST_TO_LAST_OP, "Analysis config type must be from first to last op");

    uint64_t start_timestamp = UINT64_MAX;
    uint64_t end_timestamp = 0;
    for (const tt_metal::OperationDataPoint& op_data_point : operation_data_points) {
        if (does_op_data_point_match_config(op_data_point, analysis_config)) {
            start_timestamp = std::min(start_timestamp, op_data_point.timestamp);
            end_timestamp = std::max(end_timestamp, op_data_point.timestamp);
        }
    }

    return std::vector<int64_t>{
        static_cast<int64_t>(start_timestamp),
        static_cast<int64_t>(end_timestamp),
        static_cast<int64_t>(end_timestamp - start_timestamp)};
}

std::vector<int64_t> parse_operation_data_points(
    const AnalysisConfig& analysis_config, const std::vector<tt_metal::OperationDataPoint>& operation_data_points) {
    TT_FATAL(analysis_config.dimension == AnalysisDimension::OP, "Analysis config dimension must be across ops");

    std::vector<int64_t> result;

    switch (analysis_config.result_type) {
        case AnalysisResultType::DURATION: result = parse_duration(analysis_config, operation_data_points); break;
        default: TT_THROW("Invalid analysis result type");
    }

    return result;
}

}  // namespace tt::metal
