// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_buffer.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_workload.hpp>
#include <stdint.h>
#include <tt_metal/impl/program/program_command_sequence.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "buffer.hpp"
#include "buffer_constants.hpp"
#include "core_coord.hpp"
#include "hal.hpp"
#include "kernel_types.hpp"
#include "mesh_coord.hpp"
#include "mesh_device.hpp"
#include "program_device_map.hpp"
#include "program_impl.hpp"
#include "semaphore.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "util.hpp"

enum class CoreType;
namespace tt {
namespace tt_metal {
class IDevice;
class Kernel;
enum class HalProgrammableCoreType;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {
namespace {

// Returns an intersecting range from `programs` if it exists, otherwise returns std::nullopt.
std::optional<MeshCoordinateRange> find_intersection(
    const std::unordered_map<MeshCoordinateRange, Program>& programs, const MeshCoordinateRange& range) {
    for (const auto& [program_range, _] : programs) {
        if (program_range.intersects(range)) {
            return program_range;
        }
    }
    return std::nullopt;
}

}  // namespace

MeshWorkload::MeshWorkload() {
    // A MeshWorkload tracks maintains its own handles to kernels across all
    // encapsulated programs
    kernel_groups_.resize(hal_ref.get_programmable_core_type_count());
    kernels_.resize(hal_ref.get_programmable_core_type_count());
}

void MeshWorkload::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    auto potential_intersection = find_intersection(programs_, device_range);
    TT_FATAL(
        !potential_intersection,
        "Program range {} overlaps with the previously added range {}",
        device_range,
        *potential_intersection);
    programs_[device_range] = std::move(program);
}

void MeshWorkload::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    for (auto& [device_range, program] : programs_) {
        program.compile(mesh_device);
        program.allocate_circular_buffers(mesh_device);
        tt::tt_metal::detail::ValidateCircularBufferRegion(program, mesh_device);
    }
    program_dispatch::finalize_program_offsets(*this, mesh_device);
}

void MeshWorkload::load_binaries(MeshCommandQueue& mesh_cq) {
    // Load binaries for all programs to their respective devices in
    // the Mesh. Only done when the MeshWorkload is enqueued for the first
    // time.
    auto* mesh_device = mesh_cq.device();
    if (program_binary_status_.size()) {
        TT_FATAL(
            program_binary_status_.find(mesh_device->id()) != program_binary_status_.end(),
            "Reusing MeshWorkloads across MeshDevices is currently not supported.");
        TT_FATAL(
            program_binary_status_.at(mesh_device->id()) == ProgramBinaryStatus::Committed,
            "Expected Program Biinaries to be committed to DRAM.");
    } else {
        // Allocate kernel binary buffers of max size across all devices, to ensure we have lock step allocation.
        uint32_t max_kernel_bin_buf_size = 0;
        for (auto& [device_range, program] : programs_) {
            uint32_t curr_kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            max_kernel_bin_buf_size = std::max(max_kernel_bin_buf_size, curr_kernel_bin_size);
        }
        // In production cases, max_kernel_bin_buf_size will always be non-zero (programs have kernels). This check is
        // primarily for test workloads, where a program may not have an attached kernel.
        if (max_kernel_bin_buf_size) {
            // Allocate a MeshBuffer for kernel binaries on each device. This buffer is replicated along the MeshDevice
            // and matches the max kernel binary size across programs.
            DeviceLocalBufferConfig device_local_kernel_bin_buf_config = {
                .page_size = HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                .buffer_type = BufferType::DRAM,
                .buffer_layout = TensorMemoryLayout::INTERLEAVED,
            };
            ReplicatedBufferConfig global_kernel_bin_buf_config = {
                .size = max_kernel_bin_buf_size,
            };
            kernel_bin_buf_ =
                MeshBuffer::create(global_kernel_bin_buf_config, device_local_kernel_bin_buf_config, mesh_device);
            // Iterate over the sub-grids and EnqueueWriteMeshBuffer to each sub-grid that runs an individual program
            for (auto& [device_range, program] : this->programs_) {
                std::size_t kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
                global_kernel_bin_buf_config.size = kernel_bin_size;
                auto kernel_bin_buf_view = MeshBuffer::create(
                    global_kernel_bin_buf_config,
                    device_local_kernel_bin_buf_config,
                    mesh_device,
                    kernel_bin_buf_->address());

                mesh_device->mesh_command_queue().enqueue_write_shard_to_sub_grid(
                    *kernel_bin_buf_view, program.get_program_transfer_info().binary_data.data(), device_range, false);

                std::shared_ptr<Buffer> buffer_view = Buffer::create(
                    mesh_device,
                    kernel_bin_buf_->address(),
                    kernel_bin_size,
                    HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                    BufferType::DRAM,
                    TensorMemoryLayout::INTERLEAVED,
                    std::nullopt,
                    false);
                program.set_kernels_bin_buffer(buffer_view);
            }
        }
        program_binary_status_[mesh_device->id()] = ProgramBinaryStatus::InFlight;
    }
}

ProgramBinaryStatus MeshWorkload::get_program_binary_status(std::size_t mesh_id) const {
    if (program_binary_status_.find(mesh_id) != program_binary_status_.end()) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkload::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    program_binary_status_[mesh_id] = status;
}

void MeshWorkload::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto mesh_device = mesh_cq.device();
    for (auto& [device_range, program] : programs_) {
        program.generate_dispatch_commands(mesh_device);
    }
}

bool MeshWorkload::runs_on_noc_multicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can be multicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::runs_on_noc_unicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can only be unicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::kernel_binary_always_stored_in_ringbuffer() {
    // Return true if kernel binaries cannot be placed in a ring buffer for
    // any program in the MeshWorkload
    bool stored_in_ring_buf = true;
    for (auto& [device_range, program] : programs_) {
        stored_in_ring_buf &= program.kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkload::get_kernels(
    uint32_t programmable_core_type_index) {
    // Get all kernels across all programs in the MeshWorkload
    if (kernels_.at(programmable_core_type_index).empty()) {
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            const uint32_t device_range_handle = (device_range_idx++) << 16;
            for (const auto& kernel : program.get_kernels(programmable_core_type_index)) {
                KernelHandle handle = (device_range_handle | kernel.first);
                kernels_.at(programmable_core_type_index).insert({handle, kernel.second});
            }
        }
    }
    return kernels_.at(programmable_core_type_index);
}

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkload::get_kernel_groups(uint32_t programmable_core_type_index) {
    // Get all kernel groups across all programs in the MeshWorkload
    if (kernel_groups_.at(programmable_core_type_index).empty()) {
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            const uint32_t device_range_handle = (device_range_idx++) << 16;
            for (auto& kg : program.get_kernel_groups(programmable_core_type_index)) {
                for (auto& optional_kernel_id : kg->kernel_ids) {
                    if (optional_kernel_id.has_value()) {
                        optional_kernel_id = (device_range_handle | optional_kernel_id.value());
                    }
                }
                kernel_groups_.at(programmable_core_type_index).push_back(kg);
            }
        }
    }
    return kernel_groups_.at(programmable_core_type_index);
}

std::vector<Semaphore>& MeshWorkload::semaphores() {
    // Get all semaphores across all programs in the MeshWorkload
    if (not semaphores_.size()) {
        for (auto& [device_range, program] : programs_) {
            semaphores_.insert(semaphores_.end(), program.semaphores().begin(), program.semaphores().end());
        }
    }
    return semaphores_;
}

std::vector<uint32_t> MeshWorkload::get_program_config_sizes() {
    // Get the config sizes for all L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : programs_) {
        if (global_program_config_sizes.size()) {
            for (int i = 0; i < global_program_config_sizes.size(); i++) {
                TT_FATAL(
                    global_program_config_sizes[i] == program_on_grid.second.get_program_config_sizes()[i],
                    "Expected config sizes to be identical across all programs in a MeshWorkload.");
            }
        } else {
            global_program_config_sizes = program_on_grid.second.get_program_config_sizes();
        }
    }
    return global_program_config_sizes;
}

std::unordered_set<SubDeviceId> MeshWorkload::determine_sub_device_ids(MeshDevice* mesh_device) {
    // Get the sub device ids for all program across all devices in the Workload
    std::unordered_set<SubDeviceId> sub_devices_;
    for (auto& [device_range, program] : programs_) {
        IDevice* device = mesh_device->get_device(device_range.start_coord());
        auto sub_devs_for_program = program.determine_sub_device_ids(mesh_device);
        for (auto& sub_dev : sub_devs_for_program) {
            sub_devices_.insert(sub_dev);
        }
    }
    return sub_devices_;
}

ProgramCommandSequence& MeshWorkload::get_dispatch_cmds_for_program(Program& program, uint64_t command_hash) {
    // Get the dispatch commands associated with this program
    return program.get_cached_program_command_sequences().at(command_hash);
}

// The functions below are for testing purposes only
void MeshWorkload::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    last_used_command_queue_ = mesh_cq;
}

MeshCommandQueue* MeshWorkload::get_last_used_command_queue() const { return last_used_command_queue_; }

ProgramConfig& MeshWorkload::get_program_config(uint32_t index) {
    TT_FATAL(
        programs_.size() and is_finalized(),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return programs_.begin()->second.get_program_config(index);
}

uint32_t MeshWorkload::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal_ref.get_programmable_core_type_index(programmable_core_type)).sem_offset;
}

uint32_t MeshWorkload::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t sem_size = 0;
    uint32_t program_idx = 0;
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(sem_size == program.get_sem_size(mesh_device.get(), logical_core, core_type));
        } else {
            sem_size = program.get_sem_size(mesh_device.get(), logical_core, core_type);
        }
        program_idx++;
    }
    return sem_size;
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal_ref.get_programmable_core_type_index(programmable_core_type)).cb_offset;
}

uint32_t MeshWorkload::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t cb_size = 0;
    uint32_t program_idx = 0;
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(cb_size == program.get_cb_size(mesh_device.get(), logical_core, core_type));
        } else {
            cb_size = program.get_cb_size(mesh_device.get(), logical_core, core_type);
        }
        program_idx++;
    }
    return cb_size;
}

}  // namespace tt::tt_metal::distributed
