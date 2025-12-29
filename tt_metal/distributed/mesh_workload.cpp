// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    // A MeshWorkload tracks maintains its own handles to kernels across all
    // encapsulated programs
    kernel_groups_.resize(MetalContext::instance().hal().get_programmable_core_type_count());
    kernels_.resize(MetalContext::instance().hal().get_programmable_core_type_count());
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
}

void MeshWorkloadImpl::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1

    // Filter device ranges to only compile those that belong to this submesh (if it's a submesh)
    std::vector<MeshCoordinateRange> device_ranges_to_compile;
    if (mesh_device->get_parent_mesh()) {
        // This is a submesh - filter programs that belong to this submesh
        auto* parent_mesh = mesh_device->get_parent_mesh().get();
        for (auto& [device_range, program] : programs_) {
            bool belongs_to_submesh = false;
            for (const auto& coord : device_range) {
                auto submesh_for_coord = parent_mesh->get_submesh_for_coordinate(coord);
                if (submesh_for_coord && submesh_for_coord.get() == mesh_device) {
                    belongs_to_submesh = true;
                    break;
                }
            }
            if (belongs_to_submesh) {
                device_ranges_to_compile.push_back(device_range);
            }
        }
    } else {
        // Parent mesh - compile all programs
        for (auto& [device_range, program] : programs_) {
            device_ranges_to_compile.push_back(device_range);
        }
    }

    if (device_ranges_to_compile.size() == 1) {
        // Compile from main thread for homogeneous workloads
        this->compile_program(device_ranges_to_compile[0], mesh_device);
    } else if (!device_ranges_to_compile.empty()) {
        for (const auto& device_range : device_ranges_to_compile) {
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
    // if (mesh_device->is_parent_mesh()) {
    //     auto submeshes = mesh_device->get_submeshes();
    //     for (auto& submesh : submeshes) {
    //         auto submesh
    //         load_binaries(mesh_cq);
    //     }
    //     return;
    // }

    if (!program_binary_status_.empty()) {
        TT_FATAL(
            program_binary_status_.find(mesh_device->id()) != program_binary_status_.end(),
            "Reusing MeshWorkloads across MeshDevices is currently not supported.");
        TT_FATAL(
            program_binary_status_.at(mesh_device->id()) == ProgramBinaryStatus::Committed,
            "Expected Program Binaries to be committed to DRAM.");
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

                // Convert device_range from parent mesh coordinates to submesh local coordinates
                // For submeshes, we need to convert parent coordinates to submesh local coordinates
                MeshCoordinateRange local_device_range = device_range;
                if (mesh_device->get_parent_mesh()) {
                    // This is a submesh - check if device_range belongs to this submesh via parent mesh
                    auto* parent_mesh = mesh_device->get_parent_mesh().get();
                    bool belongs_to_submesh = false;
                    for (const auto& coord : device_range) {
                        auto submesh_for_coord = parent_mesh->get_submesh_for_coordinate(coord);
                        if (submesh_for_coord && submesh_for_coord.get() == mesh_device) {
                            belongs_to_submesh = true;
                            break;
                        }
                    }

                    if (belongs_to_submesh) {
                        // For a submesh, use its local coordinate space (typically [0,0] for 1x1 submeshes)
                        // The submesh's shape defines its local coordinate range
                        local_device_range = MeshCoordinateRange(mesh_device->shape());
                    } else {
                        // Skip this program if it doesn't belong to this submesh
                        continue;
                    }
                }

                mesh_cq.enqueue_write_shard_to_sub_grid(
                    *kernel_bin_buf_view,
                    program.impl().get_program_transfer_info().binary_data.data(),
                    local_device_range,
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
    if (program_binary_status_.find(mesh_id) != program_binary_status_.end()) {
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
        // For submeshes, only generate dispatch commands for programs that belong to this submesh
        if (mesh_device->get_parent_mesh()) {
            // This is a submesh - check if device_range belongs to this submesh via parent mesh
            auto* parent_mesh = mesh_device->get_parent_mesh().get();
            bool belongs_to_submesh = false;
            for (const auto& coord : device_range) {
                auto submesh_for_coord = parent_mesh->get_submesh_for_coordinate(coord);
                if (submesh_for_coord && submesh_for_coord.get() == mesh_device) {
                    belongs_to_submesh = true;
                    break;
                }
            }

            if (!belongs_to_submesh) {
                // Skip this program if it doesn't belong to this submesh
                continue;
            }
        }
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

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkloadImpl::get_kernels(
    uint32_t programmable_core_type_index) {
    // Get all kernels across all programs in the MeshWorkload
    if (kernels_.at(programmable_core_type_index).empty()) {
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            const uint32_t device_range_handle = (device_range_idx++) << 16;
            for (const auto& kernel : program.impl().get_kernels(programmable_core_type_index)) {
                KernelHandle handle = (device_range_handle | kernel.first);
                kernels_.at(programmable_core_type_index).insert({handle, kernel.second});
            }
        }
    }
    return kernels_.at(programmable_core_type_index);
}

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkloadImpl::get_kernel_groups(uint32_t programmable_core_type_index) {
    // Get all kernel groups across all programs in the MeshWorkload
    if (kernel_groups_.at(programmable_core_type_index).empty()) {
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            const uint32_t device_range_handle = (device_range_idx++) << 16;
            for (auto& kg : program.impl().get_kernel_groups(programmable_core_type_index)) {
                for (auto& kernel_id : kg->kernel_ids) {
                    kernel_id |= device_range_handle;
                }
                kernel_groups_.at(programmable_core_type_index).push_back(kg);
            }
        }
    }
    return kernel_groups_.at(programmable_core_type_index);
}

std::vector<Semaphore>& MeshWorkloadImpl::semaphores() {
    // Get all semaphores across all programs in the MeshWorkload
    if (semaphores_.empty()) {
        for (auto& [device_range, program] : programs_) {
            semaphores_.insert(
                semaphores_.end(), program.impl().semaphores().begin(), program.impl().semaphores().end());
        }
    }
    return semaphores_;
}

std::vector<uint32_t> MeshWorkloadImpl::get_program_config_sizes() {
    // Get the config sizes for all L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : programs_) {
        if (!global_program_config_sizes.empty()) {
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
    uint32_t program_idx = 0;
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(sem_size == program.impl().get_sem_size(mesh_device.get(), logical_core, core_type));
        } else {
            sem_size = program.impl().get_sem_size(mesh_device.get(), logical_core, core_type);
        }
        program_idx++;
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
    uint32_t program_idx = 0;
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(cb_size == program.impl().get_cb_size(mesh_device.get(), logical_core, core_type));
        } else {
            cb_size = program.impl().get_cb_size(mesh_device.get(), logical_core, core_type);
        }
        program_idx++;
    }
    return cb_size;
}

void MeshWorkloadImpl::finalize_offsets(MeshDevice* mesh_device) {
    if (is_finalized()) {
        return;
    }

    // Filter programs to only include those that belong to this submesh (if it's a submesh)
    std::vector<MeshCoordinateRange> device_ranges_to_finalize;
    std::unordered_set<uint32_t> filtered_device_range_handles;
    if (mesh_device->get_parent_mesh()) {
        // This is a submesh - filter programs that belong to this submesh
        auto* parent_mesh = mesh_device->get_parent_mesh().get();
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            bool belongs_to_submesh = false;
            for (const auto& coord : device_range) {
                auto submesh_for_coord = parent_mesh->get_submesh_for_coordinate(coord);
                if (submesh_for_coord && submesh_for_coord.get() == mesh_device) {
                    belongs_to_submesh = true;
                    break;
                }
            }
            if (belongs_to_submesh) {
                device_ranges_to_finalize.push_back(device_range);
                filtered_device_range_handles.insert(device_range_idx << 16);
            }
            device_range_idx++;
        }
    } else {
        // Parent mesh - finalize all programs
        uint32_t device_range_idx = 0;
        for (auto& [device_range, program] : programs_) {
            device_ranges_to_finalize.push_back(device_range);
            filtered_device_range_handles.insert(device_range_idx << 16);
            device_range_idx++;
        }
    }

    // Create filtered kernels and kernel groups getters that only return kernels from filtered programs
    // Store filtered results in member variables to avoid thread_local caching issues
    struct FilteredGetters {
        std::unordered_map<uint32_t, std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> filtered_kernels;
        std::unordered_map<uint32_t, std::vector<std::shared_ptr<KernelGroup>>> filtered_kernel_groups;
        std::vector<Semaphore> filtered_semaphores;
    };
    auto filtered_getters = std::make_shared<FilteredGetters>();

    // Populate filtered kernels and kernel groups
    for (uint32_t index = 0; index < MetalContext::instance().hal().get_programmable_core_type_count(); ++index) {
        auto& all_kernels = this->get_kernels(index);
        for (const auto& kernel : all_kernels) {
            uint32_t device_range_handle = (kernel.first >> 16) << 16;
            if (filtered_device_range_handles.find(device_range_handle) != filtered_device_range_handles.end()) {
                filtered_getters->filtered_kernels[index][kernel.first] = kernel.second;
            }
        }

        auto& all_kernel_groups = this->get_kernel_groups(index);
        for (auto& kg : all_kernel_groups) {
            // Check if any kernel in this group belongs to a filtered device range
            bool belongs_to_filtered = false;
            for (auto kernel_id : kg->kernel_ids) {
                uint32_t device_range_handle = (kernel_id >> 16) << 16;
                if (filtered_device_range_handles.find(device_range_handle) != filtered_device_range_handles.end()) {
                    belongs_to_filtered = true;
                    break;
                }
            }
            if (belongs_to_filtered) {
                filtered_getters->filtered_kernel_groups[index].push_back(kg);
            }
        }
    }

    // Populate filtered semaphores
    for (const auto& device_range : device_ranges_to_finalize) {
        auto& program_semaphores = programs_.at(device_range).impl().semaphores();
        filtered_getters->filtered_semaphores.insert(
            filtered_getters->filtered_semaphores.end(), program_semaphores.begin(), program_semaphores.end());
    }

    tt::tt_metal::detail::KernelsGetter kernels_getter =
        [filtered_getters](uint32_t index) -> std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& {
        return filtered_getters->filtered_kernels[index];
    };

    tt::tt_metal::detail::KernelGroupsGetter kernel_groups_getter =
        [filtered_getters](uint32_t index) -> std::vector<std::shared_ptr<KernelGroup>>& {
        return filtered_getters->filtered_kernel_groups[index];
    };

    tt::tt_metal::detail::SemaphoresGetter semaphores_getter = [filtered_getters]() -> const std::vector<Semaphore>& {
        return filtered_getters->filtered_semaphores;
    };

    // Create a span with only filtered programs
    std::vector<tt::tt_metal::detail::ProgramImpl*> program_impls;
    program_impls.reserve(device_ranges_to_finalize.size());
    for (const auto& device_range : device_ranges_to_finalize) {
        program_impls.push_back(&programs_.at(device_range).impl());
    }
    tt::stl::Span<tt::tt_metal::detail::ProgramImpl*> programs(program_impls.data(), program_impls.size());

    this->max_program_kernels_sizeB_ = tt::tt_metal::detail::ProgramImpl::finalize_program_offsets(
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
