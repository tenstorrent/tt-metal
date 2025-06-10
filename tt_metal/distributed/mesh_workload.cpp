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
#include "semaphore.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "util.hpp"
#include "tracy/Tracy.hpp"

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

MeshWorkloadImpl::MeshWorkloadImpl() {
    ZoneScoped;
    // A MeshWorkload tracks maintains its own handles to kernels across all
    // encapsulated programs
    kernel_groups_.resize(MetalContext::instance().hal().get_programmable_core_type_count());
    kernels_.resize(MetalContext::instance().hal().get_programmable_core_type_count());
}

void MeshWorkloadImpl::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    ZoneScoped;
    auto potential_intersection = find_intersection(programs_, device_range);
    TT_FATAL(
        !potential_intersection,
        "Program range {} overlaps with the previously added range {}",
        device_range,
        *potential_intersection);
    programs_[device_range] = std::move(program);
}

void MeshWorkloadImpl::compile_program(const MeshCoordinateRange& device_range, MeshDevice* mesh_device) {
    ZoneScoped;
    auto& program = programs_.at(device_range);
    program.compile(mesh_device);
    program.allocate_circular_buffers(mesh_device);
    tt::tt_metal::detail::ValidateCircularBufferRegion(program, mesh_device);
}

void MeshWorkloadImpl::compile(MeshDevice* mesh_device) {
    ZoneScoped;
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    if (programs_.size() == 1) {
        // Compile from main thread for homogenous workloads
        this->compile_program(programs_.begin()->first, mesh_device);
    } else {
        for (auto& [device_range, _] : programs_) {
            // Multi-Threaded Compile: Useful for heterogenous MeshWorkloads
            mesh_device->enqueue_to_thread_pool(
                [device_range, mesh_device, this]() { this->compile_program(device_range, mesh_device); });
        }
        mesh_device->wait_for_thread_pool();
    }
    finalize_offsets(mesh_device);
}

void MeshWorkloadImpl::load_binaries(MeshCommandQueue& mesh_cq) {
    ZoneScoped;
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
                .buffer_layout = TensorMemoryLayout::INTERLEAVED,
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

                mesh_device->mesh_command_queue().enqueue_write_shard_to_sub_grid(
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
                    TensorMemoryLayout::INTERLEAVED,
                    std::nullopt,
                    false);
                program.set_kernels_bin_buffer(buffer_view);
            }
        }
        program_binary_status_[mesh_device->id()] = ProgramBinaryStatus::InFlight;
    }
}

ProgramBinaryStatus MeshWorkloadImpl::get_program_binary_status(std::size_t mesh_id) const {
    ZoneScoped;
    if (program_binary_status_.find(mesh_id) != program_binary_status_.end()) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkloadImpl::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    ZoneScoped;
    program_binary_status_[mesh_id] = status;
}

void MeshWorkloadImpl::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    ZoneScoped;
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto mesh_device = mesh_cq.device();
    for (auto& [device_range, program] : programs_) {
        program.generate_dispatch_commands(mesh_device);
    }
}

bool MeshWorkloadImpl::runs_on_noc_multicast_only_cores() {
    ZoneScoped;
    // Return true if any program in the MeshWorkload runs on cores
    // that can be multicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkloadImpl::runs_on_noc_unicast_only_cores() {
    ZoneScoped;
    // Return true if any program in the MeshWorkload runs on cores
    // that can only be unicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkloadImpl::kernel_binary_always_stored_in_ringbuffer() {
    ZoneScoped;
    // Return true if kernel binaries cannot be placed in a ring buffer for
    // any program in the MeshWorkload
    bool stored_in_ring_buf = true;
    for (auto& [device_range, program] : programs_) {
        stored_in_ring_buf &= program.kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkloadImpl::get_kernels(
    uint32_t programmable_core_type_index) {
    ZoneScoped;
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

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkloadImpl::get_kernel_groups(uint32_t programmable_core_type_index) {
    ZoneScoped;
    // Get all kernel groups across all programs in the MeshWorkload
    if (kernel_groups_.at(programmable_core_type_index).empty()) {
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            const uint32_t device_range_handle = (device_range_idx++) << 16;
            for (auto& kg : program.impl().get_kernel_groups(programmable_core_type_index)) {
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

std::vector<Semaphore>& MeshWorkloadImpl::semaphores() {
    ZoneScoped;
    // Get all semaphores across all programs in the MeshWorkload
    if (not semaphores_.size()) {
        for (auto& [device_range, program] : programs_) {
            semaphores_.insert(semaphores_.end(), program.semaphores().begin(), program.semaphores().end());
        }
    }
    return semaphores_;
}

std::vector<uint32_t> MeshWorkloadImpl::get_program_config_sizes() {
    ZoneScoped;
    // Get the config sizes for all L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : programs_) {
        if (global_program_config_sizes.size()) {
            for (int i = 0; i < global_program_config_sizes.size(); i++) {
                TT_FATAL(
                    global_program_config_sizes[i] == program_on_grid.second.impl().get_program_config_sizes()[i],
                    "Expected config sizes to be identical across all programs in a MeshWorkload.");
            }
        } else {
            global_program_config_sizes = program_on_grid.second.impl().get_program_config_sizes();
        }
    }
    return global_program_config_sizes;
}

std::unordered_set<SubDeviceId> MeshWorkloadImpl::determine_sub_device_ids(MeshDevice* mesh_device) {
    ZoneScoped;
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

ProgramCommandSequence& MeshWorkloadImpl::get_dispatch_cmds_for_program(Program& program, uint64_t command_hash) {
    ZoneScoped;
    // Get the dispatch commands associated with this program
    return program.get_cached_program_command_sequences().at(command_hash);
}

// The functions below are for testing purposes only
void MeshWorkloadImpl::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    ZoneScoped;
    last_used_command_queue_ = mesh_cq;
}

MeshCommandQueue* MeshWorkloadImpl::get_last_used_command_queue() const { return last_used_command_queue_; }

ProgramConfig& MeshWorkloadImpl::get_program_config(uint32_t index) {
    ZoneScoped;
    TT_FATAL(
        programs_.size() and is_finalized(),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return programs_.begin()->second.impl().get_program_config(index);
}

uint32_t MeshWorkloadImpl::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord /*logical_core*/, CoreType core_type) {
    ZoneScoped;
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr +
           get_program_config(MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
               .sem_offset;
}

uint32_t MeshWorkloadImpl::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    ZoneScoped;
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

uint32_t MeshWorkloadImpl::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord /*logical_core*/, CoreType core_type) {
    ZoneScoped;
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr +
           get_program_config(MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
               .cb_offset;
}

uint32_t MeshWorkloadImpl::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    ZoneScoped;
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

void MeshWorkloadImpl::finalize_offsets(MeshDevice* mesh_device) {
    ZoneScoped;
    if (is_finalized()) {
        return;
    }

    tt::tt_metal::detail::KernelsGetter kernels_getter =
        [this](uint32_t index) -> std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& {
        return this->get_kernels(index);
    };

    tt::tt_metal::detail::KernelGroupsGetter kernel_groups_getter =
        [this](uint32_t index) -> std::vector<std::shared_ptr<KernelGroup>>& { return this->get_kernel_groups(index); };

    tt::tt_metal::detail::SemaphoresGetter semaphores_getter = [this]() -> const std::vector<Semaphore>& {
        return this->semaphores();
    };

    // Create a span with all programs
    std::vector<tt::tt_metal::detail::ProgramImpl*> program_impls;
    program_impls.reserve(programs_.size());
    for (auto& [_, program] : programs_) {
        program_impls.push_back(&program.impl());
    }
    tt::stl::Span<tt::tt_metal::detail::ProgramImpl*> programs(program_impls.data(), program_impls.size());

    tt::tt_metal::detail::ProgramImpl::finalize_program_offsets(
        mesh_device, kernels_getter, kernel_groups_getter, semaphores_getter, programs);

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
