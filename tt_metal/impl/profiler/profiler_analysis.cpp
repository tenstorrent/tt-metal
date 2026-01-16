// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <string>
#include <optional>
#include <sstream>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include <fstream>

#include "core_coord.hpp"
#include "impl/context/metal_context.hpp"
#include "profiler_analysis.hpp"
#include "profiler_state_manager.hpp"
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

namespace std {
std::size_t hash<tt::tt_metal::experimental::ProgramExecutionUID>::operator()(
    const tt::tt_metal::experimental::ProgramExecutionUID& id) const {
    std::hash<uint64_t> hasher;
    std::size_t hash_value = 0;
    constexpr std::size_t hash_combine_prime = 0x9e3779b9;
    hash_value ^= hasher(id.runtime_id) + hash_combine_prime + (hash_value << 6) + (hash_value >> 2);
    hash_value ^= hasher(id.trace_id) + hash_combine_prime + (hash_value << 6) + (hash_value >> 2);
    hash_value ^= hasher(id.trace_id_counter) + hash_combine_prime + (hash_value << 6) + (hash_value >> 2);
    return hash_value;
}
};  // namespace std

namespace tracy {
NLOHMANN_JSON_SERIALIZE_ENUM(
    TTDeviceMarkerType,
    {{TTDeviceMarkerType::ZONE_START, "ZONE_START"},
     {TTDeviceMarkerType::ZONE_END, "ZONE_END"},
     {TTDeviceMarkerType::ZONE_TOTAL, "ZONE_TOTAL"},
     {TTDeviceMarkerType::TS_DATA, "TS_DATA"},
     {TTDeviceMarkerType::TS_EVENT, "TS_EVENT"}});

NLOHMANN_JSON_SERIALIZE_ENUM(
    RiscType,
    {{RiscType::BRISC, "BRISC"},
     {RiscType::NCRISC, "NCRISC"},
     {RiscType::TRISC_0, "TRISC_0"},
     {RiscType::TRISC_1, "TRISC_1"},
     {RiscType::TRISC_2, "TRISC_2"},
     {RiscType::TENSIX_RISC_AGG, "TENSIX_RISC_AGG"},
     {RiscType::ERISC, "ERISC"}});
}  // namespace tracy

namespace tt::tt_metal {

namespace detail {

static uint64_t choose_quantized_bucket_size(uint64_t min_ns, uint64_t max_ns, uint32_t buckets, uint64_t quantum_ns) {
    if (buckets == 0) {
        return 1;
    }
    if (quantum_ns == 0) {
        quantum_ns = 1;
    }
    if (max_ns < min_ns) {
        std::swap(min_ns, max_ns);
    }
    const uint64_t span = max_ns - min_ns;
    const uint64_t needed = std::max<uint64_t>(1, (span + buckets - 1) / buckets);  // ceil(span / buckets)

    const uint64_t rounded = ((needed + (quantum_ns / 2)) / quantum_ns) * quantum_ns;
    uint64_t bucket_size = std::max<uint64_t>(quantum_ns, rounded);

    if (static_cast<__int128>(bucket_size) * static_cast<__int128>(buckets) < static_cast<__int128>(span)) {
        bucket_size = ((needed + quantum_ns - 1) / quantum_ns) * quantum_ns;
    }

    return std::max<uint64_t>(1, bucket_size);
}

experimental::DurationHistogram make_quantized_histogram_ns(
    const std::vector<uint64_t>& samples_ns, uint64_t min_ns, uint64_t max_ns, uint32_t buckets, uint64_t quantum_ns) {
    experimental::DurationHistogram hist;
    hist.num_buckets = buckets;

    if (buckets == 0) {
        return hist;
    }

    if (max_ns < min_ns) {
        std::swap(min_ns, max_ns);
    }

    const uint64_t bucket_size = choose_quantized_bucket_size(min_ns, max_ns, buckets, quantum_ns);
    const uint64_t start = (min_ns / bucket_size) * bucket_size;
    const uint64_t end =
        static_cast<uint64_t>(static_cast<__int128>(start) + (static_cast<__int128>(bucket_size) * buckets));

    hist.min_ns = start;
    hist.max_ns = end;

    hist.bucket_edges_ns.resize(static_cast<size_t>(buckets) + 1);
    hist.bucket_counts.assign(static_cast<size_t>(buckets), 0);

    for (uint32_t i = 0; i <= buckets; ++i) {
        hist.bucket_edges_ns[i] =
            static_cast<uint64_t>(static_cast<__int128>(start) + (static_cast<__int128>(bucket_size) * i));
    }

    for (uint64_t sample : samples_ns) {
        if (sample < start) {
            hist.underflow++;
            continue;
        }
        if (sample >= end) {
            hist.overflow++;
            continue;
        }
        const uint64_t rel = sample - start;
        size_t bucket_idx = static_cast<size_t>(rel / bucket_size);
        if (bucket_idx >= hist.bucket_counts.size()) {
            bucket_idx = hist.bucket_counts.size() - 1;
        }
        hist.bucket_counts[bucket_idx] += 1;
    }

    return hist;
}

}  // namespace detail

// INVALID_NUM_PROGRAM_EXECUTION_UID and INVALID_NUM must be equal to ensure proper translation between TTDeviceMarker
// IDs and ProgramExecutionUID. INVALID_NUM cannot be used directly because ProgramExecutionUID is exposed in the public
// API, and INVALID_NUM is declared in the Tracy submodule which should not be exposed.
static_assert(tt::tt_metal::experimental::INVALID_NUM_PROGRAM_EXECUTION_UID == tracy::TTDeviceMarker::INVALID_NUM);

static inline const experimental::ProgramSingleAnalysisResult PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT = {
    .start_timestamp = UINT64_MAX, .end_timestamp = 0, .duration = 0};

bool experimental::ProgramSingleAnalysisResult::operator<(
    const experimental::ProgramSingleAnalysisResult& other) const {
    if (start_timestamp != other.start_timestamp) {
        return start_timestamp < other.start_timestamp;
    }
    if (end_timestamp != other.end_timestamp) {
        return end_timestamp < other.end_timestamp;
    }
    return duration < other.duration;
}

bool experimental::ProgramSingleAnalysisResult::operator==(
    const experimental::ProgramSingleAnalysisResult& other) const {
    return start_timestamp == other.start_timestamp && end_timestamp == other.end_timestamp &&
           duration == other.duration;
}

bool experimental::ProgramSingleAnalysisResult::operator!=(
    const experimental::ProgramSingleAnalysisResult& other) const {
    return !(*this == other);
}

bool experimental::ProgramExecutionUID::operator==(const experimental::ProgramExecutionUID& other) const {
    return runtime_id == other.runtime_id && trace_id == other.trace_id && trace_id_counter == other.trace_id_counter;
}

bool experimental::ProgramExecutionUID::operator<(const experimental::ProgramExecutionUID& other) const {
    if (runtime_id != other.runtime_id) {
        return runtime_id < other.runtime_id;
    }
    if (trace_id != other.trace_id) {
        return trace_id < other.trace_id;
    }
    return trace_id_counter < other.trace_id_counter;
}

bool experimental::ProgramAnalysisData::operator==(const experimental::ProgramAnalysisData& other) const {
    return this->program_execution_uid == other.program_execution_uid &&
           this->program_analyses_results == other.program_analyses_results &&
           this->core_count == other.core_count &&
           this->num_available_cores == other.num_available_cores;
}

bool experimental::ProgramAnalysisData::operator<(const experimental::ProgramAnalysisData& other) const {
    TT_ASSERT(this->program_analyses_results.contains("DEVICE FW DURATION [ns]"));
    TT_ASSERT(other.program_analyses_results.contains("DEVICE FW DURATION [ns]"));

    const experimental::ProgramSingleAnalysisResult& this_fw_duration_analysis =
        this->program_analyses_results.at("DEVICE FW DURATION [ns]");
    const ProgramSingleAnalysisResult& other_fw_duration_analysis =
        other.program_analyses_results.at("DEVICE FW DURATION [ns]");

    return this_fw_duration_analysis < other_fw_duration_analysis;
}

bool matches_start_end_risc(tracy::RiscType risc_type, const AnalysisRiscTypes& config_risc_types) {
    return config_risc_types.contains(risc_type);
}

bool matches_start_end_marker_type(
    tracy::TTDeviceMarkerType marker_type, const AnalysisMarkerTypes& config_marker_types) {
    return config_marker_types.contains(marker_type);
}

bool matches_start_end_marker_name_keywords(
    const std::array<
        bool,
        static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
            tracy::MarkerDetails::MarkerNameKeyword::COUNT)>& marker_name_keywords,
    const AnalysisMarkerNameKeywords& config_marker_name_keywords) {
    if (config_marker_name_keywords.empty()) {
        return true;
    }

    bool match = false;
    for (tracy::MarkerDetails::MarkerNameKeyword keyword : config_marker_name_keywords) {
        match |=
            marker_name_keywords[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(keyword)];
    }
    return match;
}

bool matches_start_end_config(const tracy::TTDeviceMarker& marker, const AnalysisStartEndConfig& start_end_config) {
    return matches_start_end_risc(marker.risc, start_end_config.risc_types) &&
           matches_start_end_marker_type(marker.marker_type, start_end_config.marker_types) &&
           matches_start_end_marker_name_keywords(
               marker.marker_name_keyword_flags, start_end_config.marker_name_keywords);
}

AnalysisResults parse_duration(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    ZoneScoped;

    TT_FATAL(analysis_config.type == AnalysisType::PROGRAM_FIRST_TO_LAST_MARKER, "Unsupported analysis type");

    AnalysisResults analysis_results;
    std::unordered_map<experimental::ProgramExecutionUID, experimental::ProgramSingleAnalysisResult>&
        results_per_program_execution_uid = analysis_results.results_per_program_execution_uid;
    ChipId device_id = -1;

    for (uint32_t i = 0; i < markers.size(); ++i) {
        const auto& marker_ref = markers[i];
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        const experimental::ProgramExecutionUID program_execution_uid = {
            marker.runtime_host_id, marker.trace_id, marker.trace_id_counter};
        auto [program_execution_uid_results_it, _] = results_per_program_execution_uid.try_emplace(
            program_execution_uid, PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT);
        experimental::ProgramSingleAnalysisResult& program_results = program_execution_uid_results_it->second;

        if (matches_start_end_config(marker, analysis_config.start_config)) {
            if (program_results == PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                program_results.start_timestamp = marker.timestamp;
            }
        }
        if (matches_start_end_config(marker, analysis_config.end_config)) {
            if (program_results != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                program_results.end_timestamp = marker.timestamp;
            }
        }

        if (i == 0) {
            device_id = marker.chip_id;
        }
        TT_ASSERT(device_id == marker.chip_id);
    }

    for (auto& [_, result] : results_per_program_execution_uid) {
        if (result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
            TT_ASSERT(result.start_timestamp <= result.end_timestamp);
            const int chip_frequency_mhz =
                tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
            result.duration = static_cast<uint64_t>(
                std::round((result.end_timestamp - result.start_timestamp) * 1000.0 / chip_frequency_mhz));
        }
    }

    analysis_results.results_config = analysis_config.results_config;

    return analysis_results;
}

std::map<experimental::ProgramExecutionUID, ProgramsPerfResults::SingleProgramPerfResults::ProgramMetaData>
getMetaDataForPrograms(const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& markers) {
    ZoneScoped;

    std::map<experimental::ProgramExecutionUID, ProgramsPerfResults::SingleProgramPerfResults::ProgramMetaData>
        program_execution_uid_to_meta_data;

    std::unordered_map<experimental::ProgramExecutionUID, std::unordered_set<CoreCoord>>
        fw_cores_per_program_execution_uid;
    for (const auto& marker_ref : markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        const experimental::ProgramExecutionUID program_execution_uid = {
            marker.runtime_host_id, marker.trace_id, marker.trace_id_counter};
        if (!program_execution_uid_to_meta_data.contains(program_execution_uid)) {
            const Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const umd::ClusterDescriptor* cluster_desc = cluster.get_cluster_desc();
            const ARCH device_arch = cluster_desc->get_arch(marker.chip_id);

            const uint8_t num_hw_cqs =
                tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
            const DispatchCoreConfig& dispatch_core_config =
                tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
            const CoreCoord compute_grid_size =
                tt::get_compute_grid_size(marker.chip_id, num_hw_cqs, dispatch_core_config);
            const uint32_t num_available_worker_cores = compute_grid_size.x * compute_grid_size.y;

            program_execution_uid_to_meta_data[program_execution_uid] = {
                .device_id = static_cast<ChipId>(marker.chip_id),
                .device_arch = device_arch,
                .program_name = marker.op_name,
                .num_fw_cores = 0,
                .num_available_worker_cores = num_available_worker_cores,
            };
        }

        auto& it =
            program_execution_uid_to_meta_data[{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}];

        TT_ASSERT(it.device_id == marker.chip_id);
        TT_ASSERT(it.program_name == marker.op_name);

        if (marker
                .marker_name_keyword_flags[static_cast<std::underlying_type_t<tracy::MarkerDetails::MarkerNameKeyword>>(
                    tracy::MarkerDetails::MarkerNameKeyword::_FW)]) {
            fw_cores_per_program_execution_uid[{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter}]
                .emplace(marker.core_x, marker.core_y);
        }
    }

    for (const auto& [program_execution_uid, fw_cores] : fw_cores_per_program_execution_uid) {
        program_execution_uid_to_meta_data[program_execution_uid].num_fw_cores = fw_cores.size();
    }

    return program_execution_uid_to_meta_data;
}

AnalysisResults generateAnalysisForDeviceMarkers(
    const AnalysisConfig& analysis_config,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers) {
    TT_ASSERT(std::is_sorted(
        device_markers.begin(), device_markers.end(), [](const auto& a, const auto& b) { return a.get() < b.get(); }));
    TT_FATAL(
        analysis_config.dimension == AnalysisDimension::PROGRAM, "Analysis config dimension must be across programs");

    switch (analysis_config.type) {
        case AnalysisType::PROGRAM_FIRST_TO_LAST_MARKER: return parse_duration(analysis_config, device_markers);
        default: TT_THROW("Invalid analysis type");
    }
}

ProgramsPerfResults generatePerfResultsForPrograms(
    const std::vector<AnalysisConfig>& analysis_configs,
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    ThreadPool& thread_pool) {
    ZoneScoped;

    ProgramsPerfResults programs_perf_results;
    std::map<experimental::ProgramExecutionUID, ProgramsPerfResults::SingleProgramPerfResults>&
        program_execution_uid_to_perf_results = programs_perf_results.program_execution_uid_to_perf_results;
    std::vector<AnalysisResultsConfig>& analysis_results_configs = programs_perf_results.analysis_results_configs;

    uint32_t i = 0;
    std::vector<AnalysisResults> analysis_results(analysis_configs.size());
    analysis_results_configs.resize(analysis_configs.size());
    for (const auto& analysis_config : analysis_configs) {
        thread_pool.enqueue([&analysis_config, &analysis_results_configs, &device_markers, &analysis_results, i]() {
            analysis_results[i] = generateAnalysisForDeviceMarkers(analysis_config, device_markers);
            analysis_results_configs[i] = analysis_results[i].results_config;
        });
        i++;
    }

    std::map<experimental::ProgramExecutionUID, ProgramsPerfResults::SingleProgramPerfResults::ProgramMetaData>
        programs_meta_data = getMetaDataForPrograms(device_markers);

    thread_pool.wait();

    for (auto& [program_execution_uid, program_meta_data] : programs_meta_data) {
        program_execution_uid_to_perf_results[program_execution_uid].program_meta_data = std::move(program_meta_data);
        ProgramsPerfResults::SingleProgramPerfResults& program_perf_results =
            program_execution_uid_to_perf_results[program_execution_uid];

        for (const AnalysisResults& analysis_result : analysis_results) {
            TT_ASSERT(analysis_result.results_per_program_execution_uid.contains(program_execution_uid));
            const experimental::ProgramSingleAnalysisResult& single_result =
                analysis_result.results_per_program_execution_uid.at(program_execution_uid);
            program_perf_results.analysis_results.push_back(single_result);
        }
        TT_ASSERT(program_perf_results.analysis_results.size() == analysis_results_configs.size());
    }

    return programs_perf_results;
}

void writeProgramsPerfResultsToCSV(
    const ProgramsPerfResults& programs_perf_results, const std::filesystem::path& report_path) {
    ZoneScoped;

    std::scoped_lock lock(
        tt::tt_metal::MetalContext::instance().profiler_state_manager()->programs_perf_report_write_mutex);

    struct CsvRowData {
        std::string base_columns;
        ChipId device_id = -1;
        size_t order_index = 0;
        std::optional<uint64_t> kernel_start_cycle;
        std::optional<uint64_t> kernel_end_cycle;
        std::optional<uint64_t> kernel_duration_ns;
        std::optional<uint64_t> dm_start_cycle;
        std::optional<uint64_t> dm_end_cycle;
        std::optional<uint64_t> dm_duration_ns;
        std::string kernel_latency_ns;
        std::string dm_latency_ns;
    };

    std::map<experimental::ProgramExecutionUID, CsvRowData> rows_per_uid;
    std::map<ChipId, std::vector<experimental::ProgramExecutionUID>> device_to_programs;
    std::map<ChipId, std::vector<uint64_t>> kernel_durations_ns_by_device;

    const auto get_analysis_index = [&](const std::string& analysis_name) -> std::optional<size_t> {
        for (size_t i = 0; i < programs_perf_results.analysis_results_configs.size(); ++i) {
            if (programs_perf_results.analysis_results_configs[i].analysis_name == analysis_name) {
                return i;
            }
        }
        return std::nullopt;
    };

    const std::optional<size_t> kernel_analysis_index = get_analysis_index("DEVICE KERNEL DURATION [ns]");
    const std::optional<size_t> dm_analysis_index = get_analysis_index("DEVICE KERNEL DURATION DM START [ns]");

    size_t order_counter = 0;
    for (const auto& [program_execution_uid, program_perf_results] :
         programs_perf_results.program_execution_uid_to_perf_results) {
        CsvRowData row;
        row.device_id = program_perf_results.program_meta_data.device_id;
        row.order_index = order_counter++;

        std::ostringstream row_stream;
        row_stream << program_execution_uid.runtime_id << ",";
        if (program_execution_uid.trace_id == tracy::TTDeviceMarker::INVALID_NUM) {
            row_stream << ",";
        } else {
            row_stream << program_execution_uid.trace_id << ",";
        }
        if (program_execution_uid.trace_id_counter == tracy::TTDeviceMarker::INVALID_NUM) {
            row_stream << ",";
        } else {
            row_stream << program_execution_uid.trace_id_counter << ",";
        }
        row_stream << program_perf_results.program_meta_data.device_id << ","
                   << arch_to_str(program_perf_results.program_meta_data.device_arch) << ","
                   << program_perf_results.program_meta_data.program_name << ","
                   << std::to_string(program_perf_results.program_meta_data.num_fw_cores) << ","
                   << std::to_string(program_perf_results.program_meta_data.num_available_worker_cores);

        for (size_t i = 0; i < program_perf_results.analysis_results.size(); ++i) {
            const experimental::ProgramSingleAnalysisResult& analysis_result = program_perf_results.analysis_results[i];
            const AnalysisResultsConfig& analysis_config = programs_perf_results.analysis_results_configs[i];
            row_stream << ",";
            if (analysis_result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                row_stream << analysis_result.duration;
            }

            if (analysis_config.display_start_and_end_timestamps) {
                TT_FATAL(analysis_config.start_timestamp_header.has_value(), "Start timestamp header is not set");
                TT_FATAL(analysis_config.end_timestamp_header.has_value(), "End timestamp header is not set");
                row_stream << ",";
                if (analysis_result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                    row_stream << analysis_result.start_timestamp;
                }
                row_stream << ",";
                if (analysis_result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                    row_stream << analysis_result.end_timestamp;
                }
            }

            if (kernel_analysis_index.has_value() && i == kernel_analysis_index.value() &&
                analysis_result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                row.kernel_duration_ns = analysis_result.duration;
                if (analysis_config.display_start_and_end_timestamps) {
                    row.kernel_start_cycle = analysis_result.start_timestamp;
                    row.kernel_end_cycle = analysis_result.end_timestamp;
                }
            }
            if (dm_analysis_index.has_value() && i == dm_analysis_index.value() &&
                analysis_result != PROGRAM_INVALID_SINGLE_ANALYSIS_RESULT) {
                row.dm_duration_ns = analysis_result.duration;
                if (analysis_config.display_start_and_end_timestamps) {
                    row.dm_start_cycle = analysis_result.start_timestamp;
                    row.dm_end_cycle = analysis_result.end_timestamp;
                }
            }
        }

        row.base_columns = row_stream.str();
        rows_per_uid.emplace(program_execution_uid, row);
        device_to_programs[row.device_id].push_back(program_execution_uid);
        if (row.kernel_duration_ns.has_value() && row.kernel_duration_ns.value() > 0) {
            kernel_durations_ns_by_device[row.device_id].push_back(row.kernel_duration_ns.value());
        }
    }

    auto compute_latency = [](uint64_t start_cycle,
                              uint64_t end_cycle,
                              uint64_t duration_ns,
                              uint64_t prev_end_cycle) -> std::optional<uint64_t> {
        if (end_cycle < start_cycle || start_cycle < prev_end_cycle) {
            return std::nullopt;
        }
        const uint64_t delta_cycles = end_cycle - start_cycle;
        if (delta_cycles == 0) {
            return std::nullopt;
        }
        const double ns_per_cycle = static_cast<double>(duration_ns) / static_cast<double>(delta_cycles);
        const double op_gap_cycles = static_cast<double>(start_cycle) - static_cast<double>(prev_end_cycle);
        if (op_gap_cycles < 0) {
            return std::nullopt;
        }
        return static_cast<uint64_t>(std::round(op_gap_cycles * ns_per_cycle));
    };

    for (auto& [device_id, program_uids] : device_to_programs) {
        std::sort(program_uids.begin(), program_uids.end(), [&](const auto& a, const auto& b) {
            const CsvRowData& row_a = rows_per_uid.at(a);
            const CsvRowData& row_b = rows_per_uid.at(b);
            if (row_a.kernel_start_cycle && row_b.kernel_start_cycle &&
                row_a.kernel_start_cycle.value() != row_b.kernel_start_cycle.value()) {
                return row_a.kernel_start_cycle.value() < row_b.kernel_start_cycle.value();
            }
            return row_a.order_index < row_b.order_index;
        });

        std::optional<uint64_t> prev_kernel_end_cycle;
        std::optional<uint64_t> prev_dm_end_cycle;

        for (const auto& uid : program_uids) {
            CsvRowData& row = rows_per_uid.at(uid);

            if (row.kernel_start_cycle && row.kernel_end_cycle && row.kernel_duration_ns) {
                if (prev_kernel_end_cycle) {
                    if (auto latency = compute_latency(
                            row.kernel_start_cycle.value(),
                            row.kernel_end_cycle.value(),
                            row.kernel_duration_ns.value(),
                            prev_kernel_end_cycle.value());
                        latency.has_value()) {
                        row.kernel_latency_ns = std::to_string(latency.value());
                    }
                } else {
                    row.kernel_latency_ns = "0";
                }
                prev_kernel_end_cycle = row.kernel_end_cycle;
            }

            if (row.dm_start_cycle && row.dm_end_cycle && row.dm_duration_ns) {
                if (prev_dm_end_cycle) {
                    if (auto latency = compute_latency(
                            row.dm_start_cycle.value(),
                            row.dm_end_cycle.value(),
                            row.dm_duration_ns.value(),
                            prev_dm_end_cycle.value());
                        latency.has_value()) {
                        row.dm_latency_ns = std::to_string(latency.value());
                    }
                } else {
                    row.dm_latency_ns = "0";
                }
                prev_dm_end_cycle = row.dm_end_cycle;
            }
        }
    }

    TT_ASSERT(std::filesystem::exists(report_path.parent_path()));
    TT_ASSERT(report_path.extension() == ".csv");

    std::ofstream log_file_ofs;
    if (std::filesystem::exists(report_path)) {
        log_file_ofs.open(report_path, std::ios_base::app);
    } else {
        log_file_ofs.open(report_path);

        std::string header_string =
            "GLOBAL CALL COUNT,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,DEVICE ID,DEVICE ARCH,OP NAME,CORE "
            "COUNT,AVAILABLE WORKER CORE COUNT";

        for (const auto& analysis_result_config : programs_perf_results.analysis_results_configs) {
            header_string += "," + analysis_result_config.analysis_name;
            if (analysis_result_config.display_start_and_end_timestamps) {
                TT_FATAL(
                    analysis_result_config.start_timestamp_header.has_value(), "Start timestamp header is not set");
                TT_FATAL(analysis_result_config.end_timestamp_header.has_value(), "End timestamp header is not set");
                header_string += "," + analysis_result_config.start_timestamp_header.value() + "," +
                                 analysis_result_config.end_timestamp_header.value();
            }
        }

        header_string += ",OP TO OP LATENCY [ns],OP TO OP LATENCY BR/NRISC START [ns]";

        log_file_ofs << header_string << std::endl;
    }

    for (const auto& [uid, row] : rows_per_uid) {
        (void)uid;
        log_file_ofs << row.base_columns << ",";
        log_file_ofs << row.kernel_latency_ns;
        log_file_ofs << "," << row.dm_latency_ns << "\n";
    }

    log_file_ofs.close();

    // Emit a compact stdout summary for kernel device time (useful for CI triage).
    // Histogram buckets are uniform-width with bucket size quantized to nearest 100ns and spanning observed min..max.
    constexpr uint32_t HIST_BUCKETS = 10;

    auto print_summary = [&](ChipId device_id, const std::vector<uint64_t>& samples) {
        if (samples.empty()) {
            log_info(
                tt::LogMetal,
                "Device kernel duration perf summary (device={}): no '{}' samples found in this dump",
                device_id,
                "DEVICE KERNEL DURATION [ns]");
            return;
        }

        const auto [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
        const uint64_t min_ns = *min_it;
        const uint64_t max_ns = *max_it;
        long double sum = 0.0L;
        for (uint64_t v : samples) {
            sum += static_cast<long double>(v);
        }
        const double avg_ns = static_cast<double>(sum / static_cast<long double>(samples.size()));

        log_info(
            tt::LogMetal,
            "Device kernel duration perf summary (device={}): count={}, min={}ns, avg={:.1f}ns, max={}ns",
            device_id,
            samples.size(),
            min_ns,
            avg_ns,
            max_ns);

        const experimental::DurationHistogram hist =
            detail::make_quantized_histogram_ns(samples, min_ns, max_ns, HIST_BUCKETS);
        const std::vector<uint64_t>& edges = hist.bucket_edges_ns;
        const std::vector<uint64_t>& counts = hist.bucket_counts;
        const uint64_t underflow = hist.underflow;
        const uint64_t overflow = hist.overflow;

        const uint64_t start = hist.min_ns;
        const uint64_t end = hist.max_ns;
        const uint64_t bucket_size = edges.size() >= 2 ? (edges[1] - edges[0]) : 1;

        // Render ASCII histogram.
        constexpr size_t BAR_WIDTH = 40;
        uint64_t max_count = 0;
        for (uint64_t c : counts) {
            max_count = std::max(max_count, c);
        }

        auto bar = [&](uint64_t c) -> std::string {
            if (max_count == 0) {
                return std::string(BAR_WIDTH, ' ');
            }
            const double frac = static_cast<double>(c) / static_cast<double>(max_count);
            size_t filled = static_cast<size_t>(std::llround(frac * static_cast<double>(BAR_WIDTH)));
            filled = std::min(filled, BAR_WIDTH);
            return std::string(filled, '#') + std::string(BAR_WIDTH - filled, ' ');
        };

        log_info(
            tt::LogMetal,
            "Device kernel duration perf histogram (device={}): (buckets={}, bucket_size={}ns, "
            "range=[{}..{}), "
            "underflow(<{}ns)={}, overflow(>={}ns)={})",
            device_id,
            HIST_BUCKETS,
            bucket_size,
            start,
            end,
            start,
            underflow,
            end,
            overflow);

        for (size_t i = 0; i < counts.size(); ++i) {
            const uint64_t lo = edges[i];
            const uint64_t hi = edges[i + 1];
            log_info(tt::LogMetal, "  [{:>12} .. {:>12}) |{}| {:>8}", lo, hi, bar(counts[i]), counts[i]);
        }
    };

    for (const auto& [device_id, samples] : kernel_durations_ns_by_device) {
        print_summary(device_id, samples);
    }
}

NLOHMANN_JSON_SERIALIZE_ENUM(
    AnalysisType, {{AnalysisType::PROGRAM_FIRST_TO_LAST_MARKER, "PROGRAM_FIRST_TO_LAST_MARKER"}});
NLOHMANN_JSON_SERIALIZE_ENUM(AnalysisDimension, {{AnalysisDimension::PROGRAM, "PROGRAM"}});

void from_json(const nlohmann::json& j, AnalysisResultsConfig& config) {
    j.at("analysis_name").get_to(config.analysis_name);
    config.display_start_and_end_timestamps = j.value("display_start_and_end_timestamps", false);
    config.start_timestamp_header = j.contains("start_timestamp_header")
                                        ? std::make_optional(j.at("start_timestamp_header").get<std::string>())
                                        : std::nullopt;
    config.end_timestamp_header = j.contains("end_timestamp_header")
                                      ? std::make_optional(j.at("end_timestamp_header").get<std::string>())
                                      : std::nullopt;
}

void from_json(const nlohmann::json& j, AnalysisStartEndConfig& config) {
    if (j.contains("risc")) {
        if (j["risc"].is_array()) {
            std::unordered_set<tracy::RiscType> risc_types;
            for (const auto& risc : j["risc"]) {
                risc_types.insert(risc.get<tracy::RiscType>());
            }
            config.risc_types = risc_types;
        } else {
            config.risc_types = {j.at("risc").get<tracy::RiscType>()};
        }
    } else {
        config.risc_types = AnalysisRiscTypesAny;
    }

    if (j.contains("marker_type")) {
        if (j["marker_type"].is_array()) {
            std::unordered_set<tracy::TTDeviceMarkerType> marker_types;
            for (const auto& marker_type : j["marker_type"]) {
                marker_types.insert(marker_type.get<tracy::TTDeviceMarkerType>());
            }
            config.marker_types = marker_types;
        } else {
            config.marker_types = {j.at("marker_type").get<tracy::TTDeviceMarkerType>()};
        }
    } else {
        config.marker_types = AnalysisMarkerTypesAny;
    }

    if (j.contains("marker_name_keywords")) {
        for (const auto& marker_name_keyword : j["marker_name_keywords"]) {
            config.marker_name_keywords.insert(
                tracy::MarkerDetails::marker_name_keywords_map.at(marker_name_keyword.get<std::string>()));
        }
    } else {
        config.marker_name_keywords = AnalysisMarkerNameKeywords();
    }
}

void from_json(const nlohmann::json& j, AnalysisConfig& config) {
    j.at("type").get_to(config.type);
    j.at("dimension").get_to(config.dimension);
    j.at("results_config").get_to(config.results_config);
    j.at("start_config").get_to(config.start_config);
    j.at("end_config").get_to(config.end_config);
}

std::vector<AnalysisConfig> loadAnalysisConfigsFromJSON(const std::filesystem::path& json_path) {
    TT_ASSERT(std::filesystem::exists(json_path));
    std::ifstream json_ifs(json_path);
    const nlohmann::json configs_json = nlohmann::json::parse(json_ifs);

    std::vector<AnalysisConfig> configs;
    for (const auto& config_json : configs_json) {
        configs.push_back(config_json.get<AnalysisConfig>());
    }
    return configs;
}
}  // namespace tt::tt_metal
