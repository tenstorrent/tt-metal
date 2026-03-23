#pragma once

#include "command_queue_fixture.hpp"
#include "kernels/common_dram.hpp"

namespace tt::tt_metal {

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

bool run_dram_base_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0);

bool run_dram_multi_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0);

bool run_dram_multi_core_all_controllers_test(
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

}  // namespace tt::tt_metal
