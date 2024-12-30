// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_command_queue.hpp"
#include "mesh_workload_utils.hpp"

namespace tt::tt_metal::distributed {

MeshCommandQueue::MeshCommandQueue(std::shared_ptr<MeshDevice>& mesh_device, uint32_t id) {
    this->mesh_device_ = mesh_device;
    this->id_ = id;

    this->config_buffer_mgr_ = tt::tt_metal::WorkerConfigBufferMgr();
    program_utils::initialize_worker_config_buf_mgr(this->config_buffer_mgr_);
    this->populate_virtual_program_dispatch_core();
    this->populate_dispatch_core_type();
}

uint32_t MeshCommandQueue::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) {
    if (core_type == HalProgrammableCoreType::TENSIX) {
        uint32_t num_workers = 0;
        for (auto& device : this->mesh_device_->get_devices()) {
            if (num_workers) {
                TT_FATAL(
                    num_workers == device->num_worker_cores(core_type, sub_device_id),
                    "Worker grid size must be consistent across all devices in a Mesh.");
            } else {
                num_workers = device->num_worker_cores(core_type, sub_device_id);
            }
        }
        return num_workers;
    } else {
        uint32_t min_num_worker_cores = std::numeric_limits<uint32_t>::max();
        for (auto& device : this->mesh_device_->get_devices()) {
            min_num_worker_cores = std::min(min_num_worker_cores, device->num_worker_cores(core_type, sub_device_id));
        }
        return min_num_worker_cores;
    }
}

void MeshCommandQueue::populate_virtual_program_dispatch_core() {
    int device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_ == device->virtual_program_dispatch_core(this->id_),
                "Expected Dispatch Cores to match across devices in a Mesh");
        } else {
            this->dispatch_core_ = device->virtual_program_dispatch_core(this->id_);
        }
        device_idx++;
    }
}

void MeshCommandQueue::populate_dispatch_core_type() {
    uint32_t device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_type_ == dispatch_core_manager::instance().get_dispatch_core_type(device->id()),
                "Expected the Dispatch Core Type to match across device in a Mesh");
        } else {
            this->dispatch_core_type_ = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        }
        device_idx++;
    }
}

CoreCoord MeshCommandQueue::virtual_program_dispatch_core() const { return this->dispatch_core_; }

CoreType MeshCommandQueue::dispatch_core_type() const { return this->dispatch_core_type_; }

void MeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    std::unordered_set<SubDeviceId> sub_device_ids = mesh_workload.determine_sub_device_ids(mesh_device_);
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    auto sub_device_id = *(sub_device_ids.begin());
    auto mesh_device_id = this->mesh_device_->get_mesh_id();
    TT_FATAL(
        mesh_workload.get_program_binary_status(mesh_device_id) != ProgramBinaryStatus::NotSent,
        "Expected program binaries to be written to the MeshDevice.");

    // Compute number of workers being used for this workload.
    uint32_t num_workers = 0;
    if (mesh_workload.runs_on_noc_multicast_only_cores()) {
        num_workers += this->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }
    if (mesh_workload.runs_on_noc_unicast_only_cores()) {
        num_workers += this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
    }

    program_utils::ProgramDispatchMetadata dispatch_metadata;
    // Reserve space in the L1 Kernel Config Ring Buffer for this workload.
    program_utils::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr_,
        mesh_workload.get_program_config_sizes(),
        mesh_workload.kernel_binary_always_stored_in_ringbuffer(),
        mesh_workload.get_program_binary_status(mesh_device_id),
        num_workers,
        this->expected_num_workers_completed_,
        dispatch_metadata);

    std::unordered_set<uint32_t> chip_ids_in_workload = {};
    // Iterate over all programs. Update dispatch commands per program to reflect
    // current device state. Write the finalized program command sequence to each
    // physical device tied to the program.
    for (auto& program_on_grid : mesh_workload.get_programs()) {
        auto& device_range = program_on_grid.first;
        auto& program_cmd_seq = mesh_workload.get_dispatch_cmds_for_program(program_on_grid.second);

        program_utils::update_program_dispatch_commands(
            program_on_grid.second,
            program_cmd_seq,
            this->worker_launch_message_buffer_state_.get_mcast_wptr(),
            this->worker_launch_message_buffer_state_.get_unicast_wptr(),
            this->expected_num_workers_completed_,
            this->virtual_program_dispatch_core(),
            this->dispatch_core_type(),
            sub_device_id,
            dispatch_metadata,
            mesh_workload.get_program_binary_status(mesh_device_id),
            this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));

        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                 logical_y++) {
                experimental::write_program_commands(
                    this->mesh_device_->get_device(logical_y, logical_x)->command_queue(this->id_),
                    program_cmd_seq,
                    num_workers,
                    sub_device_id,
                    dispatch_metadata.stall_first,
                    dispatch_metadata.stall_before_program,
                    false);
                chip_ids_in_workload.insert(this->mesh_device_->get_device(logical_y, logical_x)->id());
            }
        }
    }
    // Send go signals to devices not running a program to ensure consistent global state
    for (auto& device : this->mesh_device_->get_devices()) {
        if (chip_ids_in_workload.find(device->id()) == chip_ids_in_workload.end()) {
            experimental::write_go_signal(
                device->command_queue(this->id_),
                this->expected_num_workers_completed_,
                this->virtual_program_dispatch_core(),
                mesh_workload.runs_on_noc_multicast_only_cores(),
                mesh_workload.runs_on_noc_unicast_only_cores(),
                this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        }
    }
    // Increment Launch Message Buffer Write Pointers
    if (mesh_workload.runs_on_noc_multicast_only_cores()) {
        this->worker_launch_message_buffer_state_.inc_mcast_wptr(1);
    }
    if (mesh_workload.runs_on_noc_unicast_only_cores()) {
        this->worker_launch_message_buffer_state_.inc_unicast_wptr(1);
    }
    // Update the expected number of workers dispatch must wait on
    this->expected_num_workers_completed_ += num_workers;
    // From the dispatcher's perspective, binaries are now committed to DRAM
    mesh_workload.set_program_binary_status(mesh_device_id, ProgramBinaryStatus::Committed);
    mesh_workload.set_last_used_command_queue(this);

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::finish() {
    for (auto device : this->mesh_device_->get_devices()) {
        Finish(device->command_queue(this->id_));
    }
}

}  // namespace tt::tt_metal::distributed
