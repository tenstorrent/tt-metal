// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <array>
#include <unordered_set>
#include <set>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include <common/TracyTTDeviceData.hpp>
#include "assert.hpp"
#include "device.hpp"

namespace tt {

namespace tt_metal {

enum class AnalysisType { OP_FIRST_TO_LAST_MARKER };

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

// struct AnalysisResultConfig {
//     AnalysisResultType result_type;
//     std::vector<std::string> header_names;
// };

// struct DurationAnalysisResultConfig : public AnalysisResultConfig {
//     static constexpr AnalysisResultType result_type = AnalysisResultType::DURATION;
//     std::string start_timestamp_header;
//     std::string end_timestamp_header;
//     std::string duration_header;
// };

struct AnalysisResults {
    struct RuntimeIdMetaData {
        chip_id_t device_id;
        std::string op_name;
    };

    virtual ~AnalysisResults() = default;

    virtual uint32_t getNumFieldsPerResult() const = 0;
    virtual std::string getStringifiedResultsForRuntimeId(uint64_t runtime_id) const = 0;

    std::unordered_set<uint64_t> getRuntimeIds() const { return runtime_ids; }

    RuntimeIdMetaData getMetaDataForRuntimeId(uint64_t runtime_id) const {
        TT_ASSERT(runtime_id_to_meta_data.find(runtime_id) != runtime_id_to_meta_data.end());
        return runtime_id_to_meta_data.at(runtime_id);
    }

protected:
    std::unordered_set<uint64_t> runtime_ids;
    std::unordered_map<uint64_t, RuntimeIdMetaData> runtime_id_to_meta_data;
};

struct DurationAnalysisResults : public AnalysisResults {
    struct SingleResult {
        tracy::TTDeviceMarker start_marker;
        tracy::TTDeviceMarker end_marker;
        uint64_t start_timestamp = UINT64_MAX;
        uint64_t end_timestamp = 0;
        uint64_t duration = 0;
    };

    uint32_t getNumFieldsPerResult() const override { return 3; }

    std::string getStringifiedResultsForRuntimeId(uint64_t runtime_id) const override {
        TT_ASSERT(results_per_runtime_id.find(runtime_id) != results_per_runtime_id.end());
        return std::to_string(results_per_runtime_id.at(runtime_id).start_timestamp) + "," +
               std::to_string(results_per_runtime_id.at(runtime_id).end_timestamp) + "," +
               std::to_string(results_per_runtime_id.at(runtime_id).duration);
    }

    void addResultsForRuntimeId(uint64_t runtime_id, const SingleResult& result) {
        results_per_runtime_id.emplace(runtime_id, result);
        runtime_ids.insert(runtime_id);

        TT_ASSERT(result.start_marker.chip_id == result.end_marker.chip_id);
        TT_ASSERT(result.start_marker.op_name == result.end_marker.op_name);
        runtime_id_to_meta_data[runtime_id] = {
            .device_id = result.start_marker.chip_id, .op_name = result.start_marker.op_name};
    }

private:
    std::unordered_map<uint64_t, SingleResult> results_per_runtime_id;
};

struct AnalysisStartEndConfig {
    AnalysisRisc risc;
    AnalysisMarkerType marker_type;
    AnalysisMarkerNameKeywords marker_name_keywords;
};

struct AnalysisResultsDisplayConfig {
    std::string analysis_name;
    bool display_start_and_end_timestamps;
    std::optional<std::string> start_timestamp_header;
    std::optional<std::string> end_timestamp_header;
};

struct AnalysisConfig {
    AnalysisType type;
    AnalysisDimension dimension;
    std::string analysis_name;
    AnalysisStartEndConfig start_config;
    AnalysisStartEndConfig end_config;
};

// renaming headers could break other scripts
// look at process_ops_logs.py to see how to name headers
// have option to either autogen headers or manually specify headers
// std::vector<std::string> get_headers_for_analysis_config(const AnalysisConfig& analysis_config);

// rename to parse_time_series_data_points
// convert second arg to class/struct with vector as part of it
// with struct/class, we can have caching
// look into sorting beforehand to avoid doing linear traversal

std::unique_ptr<AnalysisResults> generateAnalysisForDeviceMarkers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers);

void writeAnalysisResultsToCSV(
    const std::vector<const AnalysisResults*>& analysis_results,
    const std::vector<std::vector<std::string>>& header_names);

}  // namespace tt_metal

}  // namespace tt
