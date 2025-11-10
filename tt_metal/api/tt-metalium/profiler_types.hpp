// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

enum class ProfilerReadState { NORMAL, ONLY_DISPATCH_CORES, LAST_FD_READ };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource { L1, DRAM, DRAM_AND_L1 };

static constexpr uint64_t INVALID_NUM_PROGRAM_EXECUTION_UID = (1LL << 63);

struct ProgramExecutionUID {
    uint64_t runtime_id = INVALID_NUM_PROGRAM_EXECUTION_UID;
    uint64_t trace_id = INVALID_NUM_PROGRAM_EXECUTION_UID;
    uint64_t trace_id_counter = INVALID_NUM_PROGRAM_EXECUTION_UID;

    bool operator==(const ProgramExecutionUID& other) const;

    bool operator<(const ProgramExecutionUID& other) const;
};

struct ProgramSingleAnalysisResult {
    uint64_t start_timestamp = UINT64_MAX;
    uint64_t end_timestamp = 0;
    uint64_t duration = 0;

    bool operator==(const ProgramSingleAnalysisResult& other) const;

    bool operator!=(const ProgramSingleAnalysisResult& other) const;

    bool operator<(const ProgramSingleAnalysisResult& other) const;
};
struct ProgramAnalysisData {
    ProgramExecutionUID program_execution_uid;
    std::unordered_map<std::string, ProgramSingleAnalysisResult> program_analyses_results;

    bool operator==(const ProgramAnalysisData& other) const;

    bool operator<(const ProgramAnalysisData& other) const;
};

struct DeviceProgramId {
    uint32_t base_program_id = 0;
    ChipId device_id = 0;
    bool is_host_fallback_op = false;
};
}  // namespace tt::tt_metal

namespace std {
template <>
struct hash<tt::tt_metal::ProgramExecutionUID> {
    std::size_t operator()(const tt::tt_metal::ProgramExecutionUID& id) const;
};
}  // namespace std
