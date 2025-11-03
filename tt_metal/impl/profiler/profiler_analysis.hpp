// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <nlohmann/json.hpp>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <common/TracyTTDeviceData.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <tt_stl/assert.hpp>
#include "thread_pool.hpp"

namespace tt {

namespace tt_metal {

enum class AnalysisType { OP_FIRST_TO_LAST_MARKER };

enum class AnalysisDimension { OP };

using AnalysisRiscTypes = std::unordered_set<tracy::RiscType>;
inline const AnalysisRiscTypes AnalysisRiscTypesAny = {
    tracy::RiscType::BRISC,
    tracy::RiscType::NCRISC,
    tracy::RiscType::TRISC_0,
    tracy::RiscType::TRISC_1,
    tracy::RiscType::TRISC_2,
    tracy::RiscType::ERISC,
    tracy::RiscType::TENSIX_RISC_AGG};

using AnalysisMarkerTypes = std::unordered_set<tracy::TTDeviceMarkerType>;
inline const AnalysisMarkerTypes AnalysisMarkerTypesAny = {
    tracy::TTDeviceMarkerType::ZONE_START,
    tracy::TTDeviceMarkerType::ZONE_END,
    tracy::TTDeviceMarkerType::ZONE_TOTAL,
    tracy::TTDeviceMarkerType::TS_DATA,
    tracy::TTDeviceMarkerType::TS_EVENT};

using AnalysisMarkerNameKeywords = std::unordered_set<tracy::MarkerDetails::MarkerNameKeyword>;
struct AnalysisResultsConfig {
    std::string analysis_name;
    bool display_start_and_end_timestamps = false;
    std::optional<std::string> start_timestamp_header = std::nullopt;
    std::optional<std::string> end_timestamp_header = std::nullopt;
};

struct AnalysisResults {
    AnalysisResultsConfig results_config;
    std::unordered_map<OpId, OpSingleAnalysisResult> results_per_op_id;
};

struct AnalysisStartEndConfig {
    AnalysisRiscTypes risc_types = AnalysisRiscTypesAny;
    AnalysisMarkerTypes marker_types = AnalysisMarkerTypesAny;
    AnalysisMarkerNameKeywords marker_name_keywords;
};

struct AnalysisConfig {
    AnalysisType type = AnalysisType::OP_FIRST_TO_LAST_MARKER;
    AnalysisDimension dimension = AnalysisDimension::OP;
    AnalysisResultsConfig results_config{};
    AnalysisStartEndConfig start_config{};
    AnalysisStartEndConfig end_config{};
};

struct OpsPerfResults {
    struct SingleOpPerfResults {
        struct OpMetaData {
            ChipId device_id = 0;
            ARCH device_arch = ARCH::Invalid;
            std::string op_name;
            uint32_t num_fw_cores = 0;
            uint32_t num_available_worker_cores = 0;
        };
        OpMetaData op_meta_data;
        std::vector<OpSingleAnalysisResult> analysis_results;
    };

    std::vector<AnalysisResultsConfig> analysis_results_configs;
    std::map<OpId, SingleOpPerfResults> op_id_to_perf_results;
};

OpsPerfResults generatePerfResultsForOps(
    const std::vector<AnalysisConfig>& analysis_configs,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    ThreadPool& thread_pool);

void writeOpsPerfResultsToCSV(const OpsPerfResults& ops_perf_results, const std::filesystem::path& report_path);

std::vector<AnalysisConfig> loadAnalysisConfigsFromJSON(const std::filesystem::path& json_path);
}  // namespace tt_metal

}  // namespace tt
