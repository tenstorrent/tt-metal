// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <common/TracyTTDeviceData.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include <tt_stl/assert.hpp>
#include "thread_pool.hpp"

namespace tt::tt_metal {

enum class AnalysisType { PROGRAM_FIRST_TO_LAST_MARKER };

enum class AnalysisDimension { PROGRAM };

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
    std::unordered_map<experimental::ProgramExecutionUID, experimental::ProgramSingleAnalysisResult>
        results_per_program_execution_uid;
};

struct AnalysisStartEndConfig {
    AnalysisRiscTypes risc_types = AnalysisRiscTypesAny;
    AnalysisMarkerTypes marker_types = AnalysisMarkerTypesAny;
    AnalysisMarkerNameKeywords marker_name_keywords;
};

struct AnalysisConfig {
    AnalysisType type = AnalysisType::PROGRAM_FIRST_TO_LAST_MARKER;
    AnalysisDimension dimension = AnalysisDimension::PROGRAM;
    AnalysisResultsConfig results_config{};
    AnalysisStartEndConfig start_config{};
    AnalysisStartEndConfig end_config{};
};

struct ProgramsPerfResults {
    struct SingleProgramPerfResults {
        struct ProgramMetaData {
            ChipId device_id = 0;
            ARCH device_arch = ARCH::Invalid;
            std::string program_name;
            uint32_t num_fw_cores = 0;
            uint32_t num_available_worker_cores = 0;
        };
        ProgramMetaData program_meta_data;
        std::vector<experimental::ProgramSingleAnalysisResult> analysis_results;
    };

    std::vector<AnalysisResultsConfig> analysis_results_configs;
    std::map<experimental::ProgramExecutionUID, SingleProgramPerfResults> program_execution_uid_to_perf_results;
};

ProgramsPerfResults generatePerfResultsForPrograms(
    const std::vector<AnalysisConfig>& analysis_configs,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    ThreadPool& thread_pool);

void writeProgramsPerfResultsToCSV(
    const ProgramsPerfResults& programs_perf_results, const std::filesystem::path& report_path);

std::vector<AnalysisConfig> loadAnalysisConfigsFromJSON(const std::filesystem::path& json_path);

namespace detail {

// Shared utility for building a quantized (uniform-width) histogram in nanoseconds.
//
// - `bucket_edges_ns` will have size (buckets + 1)
// - `bucket_counts` will have size (buckets)
// - The bucket width is quantized to `quantum_ns` (default 100ns) and chosen to cover the full [min_ns..max_ns] span.
experimental::DurationHistogram make_quantized_histogram_ns(
    const std::vector<uint64_t>& samples_ns,
    uint64_t min_ns,
    uint64_t max_ns,
    uint32_t buckets,
    uint64_t quantum_ns = 100);

}  // namespace detail
}  // namespace tt::tt_metal
