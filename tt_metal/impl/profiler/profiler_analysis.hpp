// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include <common/TracyTTDeviceData.hpp>
#include <tt_stl/assert.hpp>
#include "thread_pool.hpp"

namespace tt {

namespace tt_metal {

enum class AnalysisType { OP_FIRST_TO_LAST_MARKER };

enum class AnalysisDimension { OP };

using AnalysisRisc = tracy::RiscType;
constexpr AnalysisRisc AnalysisRiscAny = AnalysisRisc::BRISC | AnalysisRisc::NCRISC | AnalysisRisc::TRISC_0 |
                                         AnalysisRisc::TRISC_1 | AnalysisRisc::TRISC_2 | AnalysisRisc::ERISC;

using AnalysisMarkerType = tracy::TTDeviceMarkerType;
constexpr AnalysisMarkerType AnalysisMarkerTypeAny = AnalysisMarkerType::ZONE_START | AnalysisMarkerType::ZONE_END |
                                                     AnalysisMarkerType::ZONE_TOTAL | AnalysisMarkerType::TS_DATA |
                                                     AnalysisMarkerType::TS_EVENT;

using AnalysisMarkerNameKeywords = std::unordered_set<tracy::MarkerDetails::MarkerNameKeyword>;

struct AnalysisResultsConfig {
    std::string analysis_name{};
    bool display_start_and_end_timestamps = false;
    std::optional<std::string> start_timestamp_header = std::nullopt;
    std::optional<std::string> end_timestamp_header = std::nullopt;
};

struct OpId {
    uint64_t runtime_id;
    uint64_t trace_id;
    uint64_t trace_id_counter;

    bool operator==(const OpId& other) const {
        return runtime_id == other.runtime_id && trace_id == other.trace_id &&
               trace_id_counter == other.trace_id_counter;
    }

    bool operator<(const OpId& other) const {
        if (runtime_id != other.runtime_id) {
            return runtime_id < other.runtime_id;
        }
        if (trace_id != other.trace_id) {
            return trace_id < other.trace_id;
        }
        return trace_id_counter < other.trace_id_counter;
    }
};

struct OpIdHasher {
    std::size_t operator()(const OpId& id) const {
        return std::hash<uint64_t>{}(id.runtime_id) ^ (std::hash<uint64_t>{}(id.trace_id) << 1) ^
               (std::hash<uint64_t>{}(id.trace_id_counter) << 2);
    }
};

struct AnalysisResults {
    struct SingleResult {
        tracy::TTDeviceMarker start_marker;
        tracy::TTDeviceMarker end_marker;
        uint64_t start_timestamp = UINT64_MAX;
        uint64_t end_timestamp = 0;
        uint64_t duration = 0;

        bool operator==(const SingleResult& other) const {
            return start_timestamp == other.start_timestamp && end_timestamp == other.end_timestamp &&
                   duration == other.duration && start_marker == other.start_marker && end_marker == other.end_marker;
        }

        bool operator!=(const SingleResult& other) const { return !(*this == other); }
    };

    static inline const SingleResult INVALID_SINGLE_RESULT = {
        .start_marker = tracy::TTDeviceMarker(),
        .end_marker = tracy::TTDeviceMarker(),
        .start_timestamp = UINT64_MAX,
        .end_timestamp = 0,
        .duration = 0,
    };

    AnalysisResultsConfig results_config;
    std::unordered_map<OpId, SingleResult, OpIdHasher> results_per_op_id;

    // virtual ~AnalysisResults() = default;

    // virtual std::string getStringifiedResultsForRuntimeId(uint64_t runtime_id) const = 0;

    // std::string getStringifiedHeaders() const {
    //     std::string headers;
    //     if (results_config.display_start_and_end_timestamps) {
    //         TT_FATAL(results_config.start_timestamp_header.has_value(), "Start timestamp header is not set");
    //         TT_FATAL(results_config.end_timestamp_header.has_value(), "End timestamp header is not set");
    //         headers +=
    //             results_config.start_timestamp_header.value() + "," + results_config.end_timestamp_header.value() +
    //             ",";
    //     }
    //     headers += results_config.analysis_name;
    //     return headers;
    // }
};

struct OpsPerfResults {
    struct SingleOpPerfResults {
        struct OpMetaData {
            chip_id_t device_id;
            ARCH device_arch;
            std::string op_name;
            uint32_t num_fw_cores;
            uint32_t num_available_worker_cores;
        };
        OpMetaData op_meta_data;
        std::vector<AnalysisResults::SingleResult> analysis_results;
    };

    std::vector<AnalysisResultsConfig> analysis_results_configs;
    std::map<OpId, SingleOpPerfResults> op_id_to_perf_results;
};

struct AnalysisStartEndConfig {
    AnalysisRisc risc = AnalysisRiscAny;
    AnalysisMarkerType marker_type = AnalysisMarkerTypeAny;
    AnalysisMarkerNameKeywords marker_name_keywords{};
};

struct AnalysisConfig {
    AnalysisType type = AnalysisType::OP_FIRST_TO_LAST_MARKER;
    AnalysisDimension dimension = AnalysisDimension::OP;
    AnalysisResultsConfig results_config{};
    AnalysisStartEndConfig start_config{};
    AnalysisStartEndConfig end_config{};
};

// AnalysisResults generateAnalysisForDeviceMarkers(
//     const AnalysisConfig& analysis_config,
//     const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers);

OpsPerfResults generatePerfResultsForOps(
    const std::vector<AnalysisConfig>& analysis_configs,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    ThreadPool& thread_pool);

void writeOpsPerfResultsToCSV(const OpsPerfResults& ops_perf_results, const std::filesystem::path& report_path);

std::vector<AnalysisConfig> loadAnalysisConfigsFromJSON(const std::filesystem::path& json_path);
}  // namespace tt_metal

}  // namespace tt
