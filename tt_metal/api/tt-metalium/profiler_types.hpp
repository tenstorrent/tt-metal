// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

enum class ProfilerReadState { NORMAL, ONLY_DISPATCH_CORES, LAST_FD_READ };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource { L1, DRAM, DRAM_AND_L1 };

// struct OpId;

// struct OpAnalysisData {
//     OpId op_id;
//     std::unordered_map<std::string, AnalysisResults::SingleResult> op_analyses_results;

//     bool operator<(const OpAnalysisData& other) const {
//         TT_ASSERT(this->op_analyses_results.find("DEVICE FW DURATION [ns]") != this->op_analyses_results.end());
//         TT_ASSERT(other.op_analyses_results.find("DEVICE FW DURATION [ns]") != other.op_analyses_results.end());

//         const AnalysisResults::SingleResult& this_fw_duration_analysis =
//             this->op_analyses_results.at("DEVICE FW DURATION [ns]");
//         const AnalysisResults::SingleResult& other_fw_duration_analysis =
//             other.op_analyses_results.at("DEVICE FW DURATION [ns]");

//         return this_fw_duration_analysis < other_fw_duration_analysis;
//     }
// };

struct DeviceProgramId {
    uint32_t base_program_id = 0;
    ChipId device_id = 0;
    bool is_host_fallback_op = false;
};

}  // namespace tt::tt_metal
