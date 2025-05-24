// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <circular_buffer.hpp>
#include <device.hpp>
#include <kernel.hpp>
#include <tt-metalium/program.hpp>
#include <stdint.h>
#include <vector_aligned.hpp>
#include <array>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core_coord.hpp"
#include "dev_msgs.h"
#include "dispatch/dispatch_settings.hpp"
#include "kernel_types.hpp"
#include "program_impl.hpp"
#include "sub_device_types.hpp"
#include "dispatch/worker_config_buffer.hpp"
#include "trace/trace_node.hpp"

enum class CoreType;

namespace tt {

namespace tt_metal {
class IDevice;
class Program;
class Semaphore;
class SystemMemoryManager;
enum class ProgramBinaryStatus : uint8_t;
struct KernelGroup;
struct ProgramCommandSequence;

namespace program_dispatch {

struct ProgramDispatchMetadata {
    std::vector<ConfigBufferEntry> kernel_config_addrs;
    uint32_t sync_count;
    uint32_t stall_first;
    uint32_t stall_before_program;

    struct {
        uint32_t mesh_max_program_kernels_sizeB;
        bool is_cached;
        uint32_t offset;
    } prefetcher_cache_info;
};

struct ExpectedNumWorkerUpdates {
    // Worker count before the update
    uint32_t previous = 0;
    // Worker count after the update
    uint32_t current = 0;
    // Indicates if a wrapping occurred
    bool wrapped = false;
};

uint32_t configure_rta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset);

uint32_t configure_crta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t crta_base_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes);

uint32_t finalize_rt_args(
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t programmable_core_type_index,
    uint32_t& rta_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes);

uint32_t finalize_sems(
    uint32_t programmable_core_type_index,
    uint32_t sem_base_offset,
    const std::vector<Semaphore>& semaphores,
    uint32_t& semaphore_offset,
    uint32_t& semaphore_size);

uint32_t finalize_cbs(
    uint32_t programmable_core_type_index,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& cb_offset,
    uint32_t& cb_size,
    uint32_t& local_cb_size);

uint32_t finalize_kernel_bins(
    IDevice* device,
    uint32_t programmable_core_type_index,
    const std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& kernel_text_offset,
    uint32_t& kernel_text_size);

void insert_empty_program_dispatch_preamble_cmd(ProgramCommandSequence& program_command_sequence);

void insert_stall_cmds(ProgramCommandSequence& program_command_sequence, SubDeviceId sub_device_id, IDevice* device);

void initialize_worker_config_buf_mgr(WorkerConfigBufferMgr& config_buffer_mgr);

void reserve_space_in_kernel_config_buffer(
    WorkerConfigBufferMgr& config_buffer_mgr,
    const std::vector<uint32_t>& program_config_sizes,
    ProgramBinaryStatus program_binary_status,
    uint32_t num_program_workers,
    uint32_t expected_num_workers_completed,
    ProgramDispatchMetadata& dispatch_md);

void update_program_dispatch_commands(
    detail::ProgramImpl& program,
    ProgramCommandSequence& cached_program_command_sequence,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    CoreType dispatch_core_type,
    SubDeviceId sub_device_id,
    const ProgramDispatchMetadata& dispatch_md,
    ProgramBinaryStatus program_binary_status,
    std::pair<bool, int> unicast_go_signal_update = {false, -1});

void update_traced_program_dispatch_commands(
    const TraceNode& node,
    ProgramCommandSequence& cached_program_command_sequence,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    CoreType dispatch_core_type,
    SubDeviceId sub_device_id,
    ProgramBinaryStatus program_binary_status,
    std::pair<bool, int> unicast_go_signal_update = {false, -1});

TraceNode create_trace_node(detail::ProgramImpl& program, IDevice* device, bool use_prefetcher_cache);

void write_program_command_sequence(
    const ProgramCommandSequence& program_command_sequence,
    SystemMemoryManager& manager,
    uint32_t command_queue_id,
    CoreType dispatch_core_type,
    bool stall_first,
    bool stall_before_program,
    bool send_binary = true);

KernelHandle get_device_local_kernel_handle(KernelHandle kernel_handle);

void reset_config_buf_mgrs_and_expected_workers(
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgrs,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    uint32_t num_entries_to_reset,
    uint32_t worker_l1_unreserved_start);

void reset_worker_dispatch_state_on_device(
    IDevice* device,
    SystemMemoryManager& manager,
    uint8_t cq_id,
    CoreCoord dispatch_core,
    const DispatchArray<uint32_t>& expected_num_workers_completed,
    bool reset_launch_msg_state);

void set_num_worker_sems_on_dispatch(
    IDevice* device, SystemMemoryManager& manager, uint8_t cq_id, uint32_t num_worker_sems);

void set_go_signal_noc_data_on_dispatch(
    IDevice* device, const vector_aligned<uint32_t>& go_signal_noc_data, SystemMemoryManager& manager, uint8_t cq_id);

// Wait for number of workers to complete and then reset the counter on the device
void reset_expected_num_workers_completed_on_device(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::SubDeviceId sub_device_id,
    uint32_t num_expected_workers,
    uint8_t cq_id);

//
// Get the expected number of workers completed values for the given Program to run on the sub device.
// Expected number of workers is used for the wait command to stall until all workers are completed.
//
ExpectedNumWorkerUpdates get_expected_num_workers_completed_updates(
    uint32_t num_workers, uint32_t num_additional_workers);

void set_core_go_message_mapping_on_device(
    IDevice* device,
    const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping,
    SystemMemoryManager& manager,
    uint8_t cq_id);

}  // namespace program_dispatch

}  // namespace tt_metal

}  // namespace tt
