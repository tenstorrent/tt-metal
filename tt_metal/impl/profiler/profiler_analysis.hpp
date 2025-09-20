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
#include "assert.hpp"

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
struct AnalysisResults {
    struct RuntimeIdMetaData {
        chip_id_t device_id;
        ARCH device_arch;
        std::string op_name;
        uint32_t num_fw_cores;
        uint32_t num_available_worker_cores;
    };

    AnalysisResultsConfig results_config;

    virtual ~AnalysisResults() = default;

    virtual std::string getStringifiedResultsForRuntimeId(uint64_t runtime_id) const = 0;

    std::unordered_set<uint64_t> getRuntimeIds() const { return runtime_ids; }

    RuntimeIdMetaData getMetaDataForRuntimeId(uint64_t runtime_id) const {
        TT_ASSERT(runtime_id_to_meta_data.find(runtime_id) != runtime_id_to_meta_data.end());
        return runtime_id_to_meta_data.at(runtime_id);
    }

    void addMetaDataForRuntimeId(uint64_t runtime_id, const RuntimeIdMetaData& meta_data) {
        runtime_id_to_meta_data.emplace(runtime_id, meta_data);
    }

    std::string getStringifiedHeaders() const {
        std::string headers;
        if (results_config.display_start_and_end_timestamps) {
            TT_FATAL(results_config.start_timestamp_header.has_value(), "Start timestamp header is not set");
            TT_FATAL(results_config.end_timestamp_header.has_value(), "End timestamp header is not set");
            headers +=
                results_config.start_timestamp_header.value() + "," + results_config.end_timestamp_header.value() + ",";
        }
        headers += results_config.analysis_name;
        return headers;
    }

protected:
    std::unordered_set<uint64_t> runtime_ids;

private:
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

    std::string getStringifiedResultsForRuntimeId(uint64_t runtime_id) const override {
        std::string results;
        if (results_per_runtime_id.find(runtime_id) != results_per_runtime_id.end()) {
            if (results_config.display_start_and_end_timestamps) {
                results += std::to_string(results_per_runtime_id.at(runtime_id).start_timestamp) + "," +
                           std::to_string(results_per_runtime_id.at(runtime_id).end_timestamp) + ",";
            }
            results += std::to_string(results_per_runtime_id.at(runtime_id).duration);
        } else {
            if (results_config.display_start_and_end_timestamps) {
                results += ",,";
            }
        }
        return results;
    }

    void addResultsForRuntimeId(uint64_t runtime_id, const SingleResult& result) {
        results_per_runtime_id.emplace(runtime_id, result);
        runtime_ids.insert(runtime_id);
    }

private:
    std::unordered_map<uint64_t, SingleResult> results_per_runtime_id;
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

std::unique_ptr<AnalysisResults> generateAnalysisForDeviceMarkers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers);

void writeAnalysisResultsToCSV(
    const std::vector<std::unique_ptr<const AnalysisResults>>& analysis_results,
    const std::filesystem::path& report_path);
}  // namespace tt_metal

}  // namespace tt
