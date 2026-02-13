// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_buffer.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_workload.hpp>
#include <cstdint>
#include <tt_metal/impl/program/program_command_sequence.hpp>
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <atomic>

#include <tt_stl/assert.hpp>
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "core_coord.hpp"
#include "hal.hpp"
#include "kernel_types.hpp"
#include "mesh_coord.hpp"
#include "mesh_device.hpp"
#include "mesh_workload_impl.hpp"
#include "program/program_device_map.hpp"
#include "program/program_impl.hpp"
#include "tt-metalium/program.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "impl/buffers/semaphore.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/debug/inspector/inspector.hpp"

#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {
class IDevice;
class Kernel;
enum class HalProgrammableCoreType;
}  // namespace tt::tt_metal

namespace {
uint64_t get_next_counter() {
    static std::atomic<uint64_t> workload_counter = 0;
    return workload_counter++;
}
}  // namespace

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

MeshWorkloadImpl::MeshWorkloadImpl() : id(get_next_counter()) {
    Inspector::mesh_workload_created(this);
}

MeshWorkloadImpl::~MeshWorkloadImpl() { Inspector::mesh_workload_destroyed(this); }

void MeshWorkloadImpl::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    auto potential_intersection = find_intersection(programs_, device_range);
    TT_FATAL(
        !potential_intersection,
        "Program range {} overlaps with the previously added range {}",
        device_range,
        *potential_intersection);
    Inspector::mesh_workload_add_program(this, device_range, program.impl().get_id());
    programs_[device_range] = std::move(program);
}

void MeshWorkloadImpl::compile_program(const MeshCoordinateRange& device_range, MeshDevice* mesh_device) {
    auto& program = programs_.at(device_range);
    program.impl().compile(mesh_device);
    program.impl().allocate_circular_buffers(mesh_device);
    program.impl().validate_circular_buffer_region(mesh_device);
    program.impl().allocate_dataflow_buffers(mesh_device);
    program.impl().validate_dataflow_buffer_region(mesh_device);
}

void MeshWorkloadImpl::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    if (programs_.size() == 1) {
        // Compile from main thread for homogeneous workloads
        this->compile_program(programs_.begin()->first, mesh_device);
    } else {
        for (auto& [device_range, _] : programs_) {
            // Multi-Threaded Compile: Useful for heterogeneous MeshWorkloads
            mesh_device->enqueue_to_thread_pool(
                [device_range, mesh_device, this]() { this->compile_program(device_range, mesh_device); });
        }
        mesh_device->wait_for_thread_pool();
    }
    finalize_offsets(mesh_device);
}

void MeshWorkloadImpl::load_binaries(MeshCommandQueue& mesh_cq) {
    // Load binaries for all programs to their respective devices in
    // the Mesh. Only done when the MeshWorkload is enqueued for the first
    // time.
    auto* mesh_device = mesh_cq.device();
    if (!program_binary_status_.empty()) {
        TT_FATAL(
            program_binary_status_.contains(mesh_device->id()),
            "Reusing MeshWorkloads across MeshDevices is currently not supported.");
        TT_FATAL(
            program_binary_status_.at(mesh_device->id()) == ProgramBinaryStatus::Committed,
            "Expected Program Biinaries to be committed to DRAM.");
    } else {
        // Allocate kernel binary buffers of max size across all devices, to ensure we have lock step allocation.
        uint32_t max_kernel_bin_buf_size = 0;
        for (auto& [device_range, program] : programs_) {
            uint32_t curr_kernel_bin_size =
                program.impl().get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
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
                .bottom_up = false,
            };
            ReplicatedBufferConfig global_kernel_bin_buf_config = {
                .size = max_kernel_bin_buf_size,
            };
            kernel_bin_buf_ =
                MeshBuffer::create(global_kernel_bin_buf_config, device_local_kernel_bin_buf_config, mesh_device);
            // Iterate over the sub-grids and EnqueueWriteMeshBuffer to each sub-grid that runs an individual program
            for (auto& [device_range, program] : this->programs_) {
                std::size_t kernel_bin_size =
                    program.impl().get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
                global_kernel_bin_buf_config.size = kernel_bin_size;
                auto kernel_bin_buf_view = MeshBuffer::create(
                    global_kernel_bin_buf_config,
                    device_local_kernel_bin_buf_config,
                    mesh_device,
                    kernel_bin_buf_->address());

                mesh_cq.enqueue_write_shard_to_sub_grid(
                    *kernel_bin_buf_view,
                    program.impl().get_program_transfer_info().binary_data.data(),
                    device_range,
                    false);

                std::shared_ptr<Buffer> buffer_view = Buffer::create(
                    mesh_device,
                    kernel_bin_buf_->address(),
                    kernel_bin_size,
                    HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                    BufferType::DRAM,
                    std::nullopt,
                    false);
                program.impl().set_kernels_bin_buffer(buffer_view);
            }
        }
        set_program_binary_status(mesh_device->id(), ProgramBinaryStatus::InFlight);
    }
}

ProgramBinaryStatus MeshWorkloadImpl::get_program_binary_status(std::size_t mesh_id) const {
    if (program_binary_status_.contains(mesh_id)) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkloadImpl::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    program_binary_status_[mesh_id] = status;
    Inspector::mesh_workload_set_program_binary_status(this, mesh_id, status);
}

void MeshWorkloadImpl::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto* mesh_device = mesh_cq.device();
    auto dispatch_core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    uint32_t prefetcher_cache_sizeB = MetalContext::instance().dispatch_mem_map(dispatch_core_type).ringbuffer_size();

    bool use_prefetcher_cache =
        this->max_program_kernels_sizeB_ and this->max_program_kernels_sizeB_ <= prefetcher_cache_sizeB;
    for (auto& [device_range, program] : programs_) {
        program.impl().generate_dispatch_commands(mesh_device, use_prefetcher_cache);
    }
    this->use_prefetcher_cache_ = use_prefetcher_cache;
}

bool MeshWorkloadImpl::runs_on_noc_multicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can be multicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.impl().runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkloadImpl::runs_on_noc_unicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can only be unicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.impl().runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkloadImpl::kernel_binary_always_stored_in_ringbuffer() {
    // Return true if kernel binaries cannot be placed in a ring buffer for
    // any program in the MeshWorkload
    bool stored_in_ring_buf = true;
    for (auto& [device_range, program] : programs_) {
        stored_in_ring_buf &= program.impl().kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::vector<uint32_t> MeshWorkloadImpl::get_program_config_sizes() {
    // Get the max config sizes across all programs for L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& [_, program] : programs_) {
        auto& sizes = program.impl().get_program_config_sizes();
        if (global_program_config_sizes.empty()) {
            global_program_config_sizes = sizes;
        } else {
            for (size_t i = 0; i < global_program_config_sizes.size(); i++) {
                global_program_config_sizes[i] = std::max(global_program_config_sizes[i], sizes[i]);
            }
        }
    }
    return global_program_config_sizes;
}

std::unordered_set<SubDeviceId> MeshWorkloadImpl::determine_sub_device_ids(MeshDevice* mesh_device) {
    // Get the sub device ids for all program across all devices in the Workload
    std::unordered_set<SubDeviceId> sub_devices_;
    for (auto& [device_range, program] : programs_) {
        auto sub_devs_for_program = program.impl().determine_sub_device_ids(mesh_device);
        for (auto& sub_dev : sub_devs_for_program) {
            sub_devices_.insert(sub_dev);
        }
    }
    return sub_devices_;
}

ProgramCommandSequence& MeshWorkloadImpl::get_dispatch_cmds_for_program(Program& program, uint64_t command_hash) {
    // Get the dispatch commands associated with this program
    return program.impl().get_cached_program_command_sequences().at(command_hash);
}

// The functions below are for testing purposes only
void MeshWorkloadImpl::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    last_used_command_queue_ = mesh_cq;
}

MeshCommandQueue* MeshWorkloadImpl::get_last_used_command_queue() const { return last_used_command_queue_; }

ProgramConfig& MeshWorkloadImpl::get_program_config(uint32_t index, bool using_fast_dispatch) {
    TT_FATAL(
        !using_fast_dispatch or (!programs_.empty() and is_finalized()),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return programs_.begin()->second.impl().get_program_config(index);
}

uint32_t MeshWorkloadImpl::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(
                           MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type),
                           tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch())
                           .sem_offset;
}

uint32_t MeshWorkloadImpl::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t sem_size = 0;
    for (auto& [device_range, program] : programs_) {
        sem_size = std::max(sem_size, program.impl().get_sem_size(mesh_device.get(), logical_core, core_type));
    }
    return sem_size;
}

uint32_t MeshWorkloadImpl::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(
                           MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type),
                           tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch())
                           .cb_offset;
}

uint32_t MeshWorkloadImpl::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t cb_size = 0;
    for (auto& [device_range, program] : programs_) {
        cb_size = std::max(cb_size, program.impl().get_cb_size(mesh_device.get(), logical_core, core_type));
    }
    return cb_size;
}

void MeshWorkloadImpl::finalize_offsets(MeshDevice* mesh_device) {
    if (is_finalized()) {
        return;
    }

    // Finalize each program independently using its own data
    for (auto& [_, program] : programs_) {
        program.impl().finalize_offsets(mesh_device);
    }

    // Determine max kernel binary size across all programs
    this->max_program_kernels_sizeB_ = 0;
    for (auto& [_, program] : programs_) {
        this->max_program_kernels_sizeB_ =
            std::max(this->max_program_kernels_sizeB_, program.impl().kernel_bins_sizeB);
    }

    set_finalized();
}

// MeshWorkload PIMPL Implementation

MeshWorkload::MeshWorkload() : pimpl_(std::make_unique<MeshWorkloadImpl>()) {}
MeshWorkload::~MeshWorkload() = default;
MeshWorkload::MeshWorkload(MeshWorkload&& other) noexcept = default;
MeshWorkload& MeshWorkload::operator=(MeshWorkload&& other) noexcept = default;

void MeshWorkload::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    pimpl_->add_program(device_range, std::move(program));
}

std::unordered_map<MeshCoordinateRange, Program>& MeshWorkload::get_programs() { return pimpl_->get_programs(); }

const std::unordered_map<MeshCoordinateRange, Program>& MeshWorkload::get_programs() const {
    return pimpl_->get_programs();
}

// For testing purposes only
void MeshWorkload::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    pimpl_->set_last_used_command_queue_for_testing(mesh_cq);
}

MeshCommandQueue* MeshWorkload::get_last_used_command_queue() const { return pimpl_->get_last_used_command_queue(); }

uint32_t MeshWorkload::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_sem_base_addr(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_sem_size(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_cb_base_addr(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_cb_size(mesh_device, logical_core, core_type);
}

}  // namespace tt::tt_metal::distributed
