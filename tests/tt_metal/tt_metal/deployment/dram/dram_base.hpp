// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _DRAM_BASE_H
#define _DRAM_BASE_H

#include "command_queue_fixture.hpp"
#include "kernels/common_dram.hpp"
#include <vector>
#include <cstdint>
#include <sstream>
#include <string>

namespace tt::tt_metal {

struct DramRunSummary {
    bool pass = true;
    uint32_t bank_id = 0;
    uint64_t checked_bytes = 0;
    uint64_t suspected_write_error_bytes = 0;
    uint64_t suspected_read_error_bytes = 0;

    uint64_t prepare_ticks = 0;
    uint64_t write_ticks = 0;
    uint64_t read_ticks = 0;
    uint64_t generate_ticks = 0;
    uint64_t ncrisc_blocked_wait_ticks = 0;
    uint64_t compare_brisc_ticks = 0;
    uint64_t compare_wait_ticks = 0;
    uint64_t compare_total_ticks = 0;
    uint64_t ncrisc_idle_ticks = 0;
    uint64_t ncrisc_write_active_ticks = 0;
    uint64_t ncrisc_read_active_ticks = 0;
    uint64_t ncrisc_diag_active_ticks = 0;
    uint64_t math_generate_active_ticks = 0;
    uint64_t pack_generate_active_ticks = 0;
    uint64_t math_compare_active_ticks = 0;
    uint64_t pack_compare_active_ticks = 0;
    uint64_t unpack_compare_active_ticks = 0;
    uint64_t job_total_ticks = 0;
};

struct DramDeploymentConfig {
    uint32_t bank_id;
    uint64_t bank_offset;
    uint32_t total_bytes;
    uint32_t chunk_bytes;
    uint32_t pattern_id;
    uint32_t write_noc;
    uint32_t read_noc;
    uint32_t transfer_len_mode;
    uint32_t max_burst_len;
    uint32_t skip_writes;
    uint32_t skip_reads;
};

struct DramBankWorkerAssignment {
    uint32_t bank_id;
    CoreCoord worker_core;
};

struct DramPerCoreResult {
    CoreCoord core;
    DramBaseResult result;
};

struct DramMultiInstanceSummary {
    DramRunSummary summary;
    std::vector<DramPerCoreResult> per_core_results;
};

DramMultiInstanceSummary run_dram_persistent_jobs_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& worker_cores,
    const std::vector<std::vector<DramWorkItem>>& jobs_per_core,
    uint32_t chunk_bytes,
    DataMovementProcessor processor);

DramMultiInstanceSummary run_dram_persistent_all_workers_dram_bank_sweep_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& worker_cores,
    const std::vector<DramWorkItem>& all_jobs,
    uint32_t chunk_bytes,
    DataMovementProcessor processor);

DramRunSummary run_dram_base_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0);

DramRunSummary run_dram_multi_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0);

DramRunSummary run_dram_multi_core_all_controllers_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0);

inline std::string format_duration_seconds(uint64_t total_seconds) {
    const uint64_t hours = total_seconds / 3600u;
    total_seconds %= 3600u;

    const uint64_t minutes = total_seconds / 60u;
    const uint64_t seconds = total_seconds % 60u;

    std::ostringstream oss;

    if (hours > 0) {
        oss << hours << "h ";
    }

    if (hours > 0 || minutes > 0) {
        oss << minutes << "min ";
    }

    oss << seconds << "sec";

    return oss.str();
}

std::vector<DramBankWorkerAssignment> get_optimal_dram_bank_worker_assignments(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, tt_metal::NOC noc);

}  // namespace tt::tt_metal

#endif /* _DRAM_BASE_H */
