// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_buffer.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <tt_stl/fmt.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_workload.hpp>
#include <cstdint>
#include "impl/buffers/buffer_impl.hpp"
#include <tt_metal/impl/program/program_command_sequence.hpp>
#include "distributed/mesh_command_queue_base.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
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
#include "impl/kernels/kernel.hpp"
#include "impl/lightmetal/host_api_capture_helpers.hpp"
#include "llrt/tt_cluster.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"

#include <tt-umd/types/core_coordinates.hpp>
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

MeshWorkloadImpl::MeshWorkloadImpl() : id(get_next_counter()) {}

MeshWorkloadImpl::~MeshWorkloadImpl() = default;

MeshWorkloadImpl::FinalizedMetadata& MeshWorkloadImpl::get_finalized_metadata() {
    TT_ASSERT(finalized_metadata_.has_value());
    return *finalized_metadata_;
}

void MeshWorkloadImpl::set_finalized(uint32_t max_program_kernels_sizeB, int mesh_device_id) {
    TT_ASSERT(!is_finalized());
    FinalizedMetadata metadata{
        .max_program_kernels_sizeB = max_program_kernels_sizeB, .mesh_device_id = mesh_device_id};
    for (auto& [device_range, program] : programs_) {
        auto& program_impl = program.impl();
        metadata.num_program_devices += device_range.shape().mesh_size();
        metadata.runs_on_noc_multicast_only_cores |= program_impl.runs_on_noc_multicast_only_cores();

        const auto& program_config_sizes = program_impl.get_program_config_sizes();
        if (metadata.program_config_sizes.empty()) {
            metadata.program_config_sizes = program_config_sizes;
        } else {
            TT_FATAL(
                metadata.program_config_sizes.size() == program_config_sizes.size(),
                "Expected config sizes to be identical across all programs in a MeshWorkload.");
            for (size_t i = 0; i < metadata.program_config_sizes.size(); i++) {
                TT_FATAL(
                    metadata.program_config_sizes[i] == program_config_sizes[i],
                    "Expected config sizes to be identical across all programs in a MeshWorkload.");
            }
        }

        if (!program_impl.get_per_core_cross_node_dfbs().empty()) {
            metadata.cross_node_program_ids.push_back(program_impl.get_id());
        }
    }
    finalized_metadata_ = std::move(metadata);
}

void MeshWorkloadImpl::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    TT_FATAL(!is_finalized(), "Cannot add programs to a MeshWorkload after it has been finalized.");
    auto potential_intersection = find_intersection(programs_, device_range);
    TT_FATAL(
        !potential_intersection,
        "Program range {} overlaps with the previously added range {}",
        device_range,
        *potential_intersection);
    programs_[device_range] = std::move(program);
}

uint32_t coordinate_index_in_range(const MeshCoordinateRange& range, const MeshCoordinate& coord) {
    TT_FATAL(range.contains(coord), "Coordinate {} is not in range {}", coord, range);
    const auto shape = range.shape();
    uint32_t index = 0;
    for (size_t dim = 0; dim < coord.dims(); ++dim) {
        index = index * shape[dim] + (coord[dim] - range.start_coord()[dim]);
    }
    return index;
}

std::vector<ResolvedPerDeviceRuntimeArgsField> resolve_per_device_runtime_args(
    const PerDeviceRuntimeArgs& per_device_args, const ProgramCommandSequence& sequence) {
    std::vector<ResolvedPerDeviceRuntimeArgsField> resolved;
    resolved.reserve(per_device_args.fields.size());
    for (const auto& field : per_device_args.fields) {
        if (field->core.has_value()) {
            const auto it =
                sequence.unique_rta_locations.find(unique_rta_slot_key(field->kernel, field->core->x, field->core->y));
            TT_FATAL(
                it != sequence.unique_rta_locations.end(),
                "Per-device runtime args: kernel {} has no runtime args on core {} in the program's commands (set the "
                "program's runtime args there before its first enqueue)",
                field->kernel,
                *field->core);
            resolved.push_back({it->second + field->first_word, field.get()});
        } else {
            const auto it = sequence.common_rta_locations.find(field->kernel);
            TT_FATAL(
                it != sequence.common_rta_locations.end() && !it->second.empty(),
                "Per-device runtime args: kernel {} has no common runtime args in the program's commands",
                field->kernel);
            for (uint32_t* location : it->second) {
                resolved.push_back({location + field->first_word, field.get()});
            }
        }
    }
    return resolved;
}

const std::vector<ResolvedPerDeviceRuntimeArgsField>& resolve_per_device_runtime_args_cached(
    const PerDeviceRuntimeArgs& per_device_args, const ProgramCommandSequence& sequence) {
    auto& cache = per_device_args.resolved_cache;
    if (cache.assembly_id != sequence.assembly_id || cache.generation != per_device_args.generation ||
        cache.assembly_id == 0) {
        cache.resolved = resolve_per_device_runtime_args(per_device_args, sequence);
        cache.assembly_id = sequence.assembly_id;
        cache.generation = per_device_args.generation;
    }
    return cache.resolved;
}

void per_device_runtime_args_patches(
    const std::vector<ResolvedPerDeviceRuntimeArgsField>& resolved,
    uint32_t coord_index,
    std::vector<RtaPatch>& patches) {
    patches.clear();
    patches.reserve(resolved.size());
    for (const auto& [location, field] : resolved) {
        const bool present = field->present[coord_index] != 0;
        patches.push_back(
            {location,
             present ? field->values.data() + static_cast<size_t>(coord_index) * field->num_words : location,
             field->num_words});
    }
}

namespace {
void validate_per_device_runtime_args_mode() {
    TT_FATAL(
        !LightMetalCaptureContext::get().is_tracing(),
        "Per-device runtime args are not supported during light-metal capture");
}
}  // namespace

PerDeviceRuntimeArgsField& MeshWorkloadImpl::per_device_runtime_args_field(
    const MeshCoordinateRange& program_range,
    KernelHandle kernel_handle,
    const std::optional<CoreCoord>& core,
    uint32_t first_word,
    uint32_t num_words) {
    validate_per_device_runtime_args_mode();
    const auto program_it = programs_.find(program_range);
    TT_FATAL(
        program_it != programs_.end(),
        "Per-device runtime args: the MeshWorkload has no program on range {}",
        program_range);
    auto& program = program_it->second;
    TT_FATAL(
        !program.impl().has_metal2_registry() && !program.impl().created_from_spec(),
        "Per-device runtime args are not supported for Metal 2.0 (ProgramSpec) programs");
    TT_FATAL(num_words > 0, "Per-device runtime args: a field needs at least one word");
    TT_FATAL(
        !program_range.wraparound_shape().has_value(),
        "Per-device runtime args are not supported for wraparound program ranges ({})",
        program_range);
    const auto kernel = program.impl().get_kernel(kernel_handle);
    TT_FATAL(
        kernel != nullptr,
        "Per-device runtime args: the program on range {} has no kernel {}",
        program_range,
        kernel_handle);
    uint32_t logical_words = 0;
    if (core.has_value()) {
        TT_FATAL(
            kernel->is_on_logical_core(*core),
            "Per-device runtime args: kernel {} is not placed on core {}",
            kernel_handle,
            *core);
        logical_words = static_cast<uint32_t>(kernel->logical_runtime_arg_count(*core));
    } else {
        const auto& common = kernel->common_runtime_args();
        logical_words = common.size() > kernel->watcher_count_word_offset()
                            ? static_cast<uint32_t>(common.size() - kernel->watcher_count_word_offset())
                            : 0;
    }
    TT_FATAL(
        first_word <= logical_words && num_words <= logical_words - first_word,
        "Per-device runtime args: words [{}, {}) of kernel {}{} lie outside its {} runtime args (set the program's "
        "runtime args first)",
        first_word,
        first_word + num_words,
        kernel_handle,
        core.has_value() ? fmt::format(" on core {}", *core) : std::string(" common"),
        logical_words);

    auto& per_device = per_device_runtime_args_[program_range];
    if (per_device.fields.empty()) {
        per_device.program_id = program.impl().get_id();
    }
    TT_FATAL(
        per_device.program_id == program.impl().get_id(),
        "Per-device runtime args of range {} were set for another program",
        program_range);
    for (auto& field : per_device.fields) {
        if (field->kernel != kernel_handle || field->core != core) {
            continue;
        }
        if (field->first_word == first_word) {
            TT_FATAL(
                field->num_words == num_words,
                "Per-device runtime args: the field at word {} of kernel {} has {} words, not {}",
                first_word,
                kernel_handle,
                field->num_words,
                num_words);
            return *field;
        }
        TT_FATAL(
            first_word + num_words <= field->first_word || field->first_word + field->num_words <= first_word,
            "Per-device runtime args: words [{}, {}) of kernel {} overlap the field at words [{}, {})",
            first_word,
            first_word + num_words,
            kernel_handle,
            field->first_word,
            field->first_word + field->num_words);
    }
    const size_t num_coords = program_range.shape().mesh_size();
    auto field = std::make_unique<PerDeviceRuntimeArgsField>();
    field->kernel = kernel_handle;
    field->core = core;
    field->first_word = first_word;
    field->num_words = num_words;
    field->values.assign(num_coords * num_words, 0);
    field->present.assign(num_coords, 0);
    per_device.fields.push_back(std::move(field));
    per_device.generation++;
    per_device.values_version++;
    return *per_device.fields.back();
}

void MeshWorkloadImpl::set_per_device_runtime_args(
    const MeshCoordinateRange& program_range,
    const MeshCoordinate& device,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word,
    ttsl::Span<const uint32_t> values) {
    TT_FATAL(
        program_range.contains(device), "Per-device runtime args: device {} is not in range {}", device, program_range);
    auto& field =
        per_device_runtime_args_field(program_range, kernel, core, first_word, static_cast<uint32_t>(values.size()));
    const uint32_t index = coordinate_index_in_range(program_range, device);
    std::copy(values.begin(), values.end(), field.values.begin() + static_cast<size_t>(index) * field.num_words);
    field.present[index] = 1;
    per_device_runtime_args_.at(program_range).values_version++;
}

void MeshWorkloadImpl::set_per_device_runtime_args(
    const MeshCoordinateRange& program_range,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word,
    uint32_t num_words,
    ttsl::Span<const uint32_t> values_in_range_order) {
    TT_FATAL(
        values_in_range_order.size() == program_range.shape().mesh_size() * num_words,
        "Per-device runtime args: expected {} values ({} devices x {} words), got {}",
        program_range.shape().mesh_size() * num_words,
        program_range.shape().mesh_size(),
        num_words,
        values_in_range_order.size());
    auto& field = per_device_runtime_args_field(program_range, kernel, core, first_word, num_words);
    std::copy(values_in_range_order.begin(), values_in_range_order.end(), field.values.begin());
    std::fill(field.present.begin(), field.present.end(), 1);
    per_device_runtime_args_.at(program_range).values_version++;
}

void MeshWorkloadImpl::clear_per_device_runtime_args(
    const MeshCoordinateRange& program_range,
    const MeshCoordinate& device,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word) {
    auto it = per_device_runtime_args_.find(program_range);
    if (it == per_device_runtime_args_.end()) {
        return;
    }
    for (auto& field : it->second.fields) {
        if (field->kernel == kernel && field->core == core && field->first_word == first_word) {
            field->present[coordinate_index_in_range(program_range, device)] = 0;
            it->second.values_version++;
        }
    }
}

const PerDeviceRuntimeArgs* MeshWorkloadImpl::per_device_runtime_args(const MeshCoordinateRange& program_range) const {
    if (per_device_runtime_args_.empty()) {
        return nullptr;
    }
    const auto it = per_device_runtime_args_.find(program_range);
    if (it == per_device_runtime_args_.end()) {
        return nullptr;
    }
    validate_per_device_runtime_args_mode();
    const auto program_it = programs_.find(program_range);
    TT_FATAL(
        program_it != programs_.end() && program_it->second.impl().get_id() == it->second.program_id,
        "Per-device runtime args of range {} were set for a program that is no longer there (the program was replaced "
        "or moved out of the MeshWorkload)",
        program_range);
    return &it->second;
}

void MeshWorkloadImpl::compile_program(const MeshCoordinateRange& device_range, MeshDevice* mesh_device) {
    auto& program = programs_.at(device_range);
    program.impl().compile_and_allocate(mesh_device, false);
}

void MeshWorkloadImpl::compile(MeshDevice* mesh_device) {
    if (is_finalized()) {
        const int finalized_mesh_device_id = get_finalized_metadata().mesh_device_id;
        TT_FATAL(
            finalized_mesh_device_id == mesh_device->id(),
            "MeshWorkload was finalized for MeshDevice {} and cannot be compiled for MeshDevice {}. Reusing "
            "MeshWorkloads across MeshDevices is currently not supported.",
            finalized_mesh_device_id,
            mesh_device->id());
    }
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    // This runs on every enqueue, not just the first. Once the workload has been finalized its
    // programs are already compiled and laid out, so compile_program is a flag check per program;
    // handing those to the thread pool costs a worker wake-up and a join for no work. Only the
    // first compile of a multi-program workload does enough to be worth parallelizing.
    if (programs_.size() == 1 || this->is_finalized()) {
        // Compile from main thread for homogeneous workloads
        for (auto& [device_range, _] : programs_) {
            this->compile_program(device_range, mesh_device);
        }
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
            // We can't load while capturing a trace, it needs to already be in program cache.
            const bool is_capturing_trace = mesh_cq.trace_id().has_value();
            TT_FATAL(
                !is_capturing_trace,
                "Cannot load new binaries during trace capture."
                "This program is not yet in program cache. Warm up before capturing a trace."
                "See the operation's hash signature for arguments that must match.");
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
            // Cache hits normally reuse an initialized workload, but initialization is lazy: a cached workload whose
            // first enqueue did not reach this point can still allocate its kernel binary buffer on a later hit. Tag
            // the allocation at its source so program-cache filtering does not depend on which dispatch path got here.
            // Preserve the richer op-specific context installed by TTNN on the normal cache-miss path.
            std::optional<AllocationContextGuard> program_cache_context;
            if (!current_allocation_context().starts_with(kProgramCacheAllocationContextPrefix)) {
                program_cache_context.emplace(std::string(kProgramCacheAllocationContextPrefix) + " kernel binaries");
            }
            kernel_bin_buf_ =
                MeshBuffer::create(global_kernel_bin_buf_config, device_local_kernel_bin_buf_config, mesh_device);
            MeshCommandQueueBase::ScopedProgramBinaryUpload program_binary_upload;
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

                std::shared_ptr<Buffer> buffer_view = BufferImpl::create(
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

std::shared_ptr<MeshBuffer> MeshWorkloadImpl::prepare_for_command_list(MeshCommandQueue& mesh_cq) {
    auto* mesh_device = mesh_cq.device();
    TT_FATAL(mesh_device != nullptr, "Cannot prepare a MeshWorkload using a command queue without a MeshDevice");
    TT_FATAL(!programs_.empty(), "Cannot prepare an empty MeshWorkload for a command list");
    TT_FATAL(per_device_runtime_args_.empty(), "Per-device runtime args are not supported in command lists yet");

    for (const auto& [device_range, program] : programs_) {
        TT_FATAL(
            program.impl().created_from_spec(),
            "Command lists only support Metal 2.0 programs; program {} on mesh range {} was not created from a "
            "ProgramSpec",
            program.impl().get_id(),
            device_range);
    }

    compile(mesh_device);

    const ProgramBinaryStatus binary_status = get_program_binary_status(mesh_device->id());
    TT_FATAL(
        binary_status != ProgramBinaryStatus::InFlight,
        "Cannot prepare MeshWorkload {} for a command list while its kernel binaries are in flight",
        id);

    if (binary_status == ProgramBinaryStatus::NotSent) {
        load_binaries(mesh_cq);
        mesh_cq.finish();
        set_program_binary_status(mesh_device->id(), ProgramBinaryStatus::Committed);
    }

    generate_dispatch_commands(mesh_cq);
    return kernel_bin_buf_;
}

ProgramBinaryStatus MeshWorkloadImpl::get_program_binary_status(std::size_t mesh_id) const {
    if (program_binary_status_.contains(mesh_id)) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkloadImpl::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    program_binary_status_[mesh_id] = status;
}

void MeshWorkloadImpl::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto* mesh_device = mesh_cq.device();
    uint32_t prefetcher_cache_sizeB = mesh_device->impl().metal_context().dispatch_mem_map().ringbuffer_size();

    const uint32_t max_program_kernels_sizeB = get_finalized_metadata().max_program_kernels_sizeB;
    bool use_prefetcher_cache = max_program_kernels_sizeB and max_program_kernels_sizeB <= prefetcher_cache_sizeB;
    for (auto& [device_range, program] : programs_) {
        program.impl().generate_dispatch_commands(mesh_device, use_prefetcher_cache);
    }
    this->use_prefetcher_cache_ = use_prefetcher_cache;
}

bool MeshWorkloadImpl::runs_on_noc_multicast_only_cores() {
    return get_finalized_metadata().runs_on_noc_multicast_only_cores;
}

// kernels_ is sized to the programmable-core-type count by finalize_offsets, which reaches this
// method only through the getter it hands to finalize_program_offsets. Any new caller that runs
// before finalize_offsets sees an empty vector and throws from at().
std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkloadImpl::get_kernels(
    uint32_t programmable_core_type_index) {
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

// Same sizing guarantee, and the same failure mode for a premature caller, as get_kernels above.
std::vector<std::shared_ptr<KernelGroup>>& MeshWorkloadImpl::get_kernel_groups(uint32_t programmable_core_type_index) {
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

const std::vector<uint32_t>& MeshWorkloadImpl::get_program_config_sizes() {
    return get_finalized_metadata().program_config_sizes;
}

const std::vector<uint64_t>& MeshWorkloadImpl::get_cross_node_program_ids() {
    return get_finalized_metadata().cross_node_program_ids;
}

const std::unordered_set<SubDeviceId>& MeshWorkloadImpl::determine_sub_device_ids(MeshDevice* mesh_device) {
    auto& sub_device_ids_by_manager = sub_device_ids_by_mesh_and_manager_[mesh_device->id()];
    auto [entry, inserted] = sub_device_ids_by_manager.try_emplace(*mesh_device->get_active_sub_device_manager_id());
    auto& sub_device_ids = entry->second;
    if (inserted) {
        for (auto& [device_range, program] : programs_) {
            auto sub_devices_for_program = program.impl().determine_sub_device_ids(mesh_device);
            sub_device_ids.insert(sub_devices_for_program.begin(), sub_devices_for_program.end());
        }
    }
    return sub_device_ids;
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

namespace {
// Emulated devices launch each program through the slow-dispatch path (tt-emule's queue), as slow dispatch does,
// so their workloads carry per-program configs rather than the FD workload finalization.
bool uses_fd_program_configs(MetalEnvImpl& env) {
    return env.get_rtoptions().get_fast_dispatch() &&
           env.get_cluster().get_target_device_type() != tt::TargetDevice::Emule;
}
}  // namespace

ProgramConfig& MeshWorkloadImpl::get_program_config(uint32_t index, bool using_fast_dispatch) {
    TT_FATAL(!programs_.empty(), "Program Configs can only be queried if a MeshWorkload is populated.");
    const bool requires_finalized_config = using_fast_dispatch && !is_service_workload_.value_or(false);
    TT_FATAL(
        !requires_finalized_config || is_finalized(),
        "Program Configs on a fast-dispatch MeshWorkload can only be queried after finalization.");
    return programs_.begin()->second.impl().get_program_config(index);
}

uint32_t MeshWorkloadImpl::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    auto& env = mesh_device->impl().metal_env();
    return base_addr +
           get_program_config(
               env.get_hal().get_programmable_core_type_index(programmable_core_type), uses_fd_program_configs(env))
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
    auto& env = mesh_device->impl().metal_env();
    return base_addr +
           get_program_config(
               env.get_hal().get_programmable_core_type_index(programmable_core_type), uses_fd_program_configs(env))
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

    // Sizing the kernel tables is what makes get_kernels / get_kernel_groups indexable, and the
    // getters built below are their only callers. Keep that ordering if this function is reworked.
    const uint32_t num_core_types = mesh_device->impl().metal_env().get_hal().get_programmable_core_type_count();
    kernel_groups_.resize(num_core_types);
    kernels_.resize(num_core_types);

    tt::tt_metal::detail::KernelsGetter kernels_getter =
        [this](uint32_t index) -> std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& {
        return this->get_kernels(index);
    };

    tt::tt_metal::detail::KernelGroupsGetter kernel_groups_getter =
        [this](uint32_t index) -> std::vector<std::shared_ptr<KernelGroup>>& { return this->get_kernel_groups(index); };

    tt::tt_metal::detail::SemaphoresGetter semaphores_getter = [this]() -> const std::vector<Semaphore>& {
        return this->semaphores();
    };

    std::vector<tt::tt_metal::detail::ProgramImpl*> program_impls;
    program_impls.reserve(programs_.size());
    for (auto& [_, program] : programs_) {
        program_impls.push_back(&program.impl());
    }
    ttsl::Span<tt::tt_metal::detail::ProgramImpl*> programs(program_impls.data(), program_impls.size());

    const uint32_t max_program_kernels_sizeB = tt::tt_metal::detail::ProgramImpl::finalize_program_offsets(
        mesh_device->impl().get_context_id(),
        mesh_device,
        kernels_getter,
        kernel_groups_getter,
        semaphores_getter,
        programs);

    set_finalized(max_program_kernels_sizeB, mesh_device->id());
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

void SetRuntimeArgs(
    MeshWorkload& mesh_workload,
    const MeshCoordinateRange& program_range,
    const MeshCoordinate& device,
    KernelHandle kernel,
    const CoreCoord& core,
    uint32_t first_word,
    ttsl::Span<const uint32_t> values) {
    mesh_workload.impl().set_per_device_runtime_args(program_range, device, kernel, core, first_word, values);
}

void SetCommonRuntimeArgs(
    MeshWorkload& mesh_workload,
    const MeshCoordinateRange& program_range,
    const MeshCoordinate& device,
    KernelHandle kernel,
    uint32_t first_word,
    ttsl::Span<const uint32_t> values) {
    mesh_workload.impl().set_per_device_runtime_args(program_range, device, kernel, std::nullopt, first_word, values);
}

void SetRuntimeArgsPerDevice(
    MeshWorkload& mesh_workload,
    const MeshCoordinateRange& program_range,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word,
    uint32_t num_words,
    ttsl::Span<const uint32_t> values_in_range_order) {
    mesh_workload.impl().set_per_device_runtime_args(
        program_range, kernel, core, first_word, num_words, values_in_range_order);
}

bool PerDeviceRuntimeArgsSupported(MeshDevice& mesh_device, const MeshCoordinateRange& program_range) {
    return !program_range.wraparound_shape().has_value() && !LightMetalCaptureContext::get().is_tracing() &&
           mesh_device.impl().metal_env().get_cluster().get_target_device_type() != tt::TargetDevice::Emule;
}

void ClearRuntimeArgs(
    MeshWorkload& mesh_workload,
    const MeshCoordinateRange& program_range,
    const MeshCoordinate& device,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word) {
    mesh_workload.impl().clear_per_device_runtime_args(program_range, device, kernel, core, first_word);
}

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

namespace tt::tt_metal::experimental::program_preparation {

void prepare(distributed::MeshDevice& mesh_device, distributed::MeshWorkload& workload) {
    // EnqueueMeshWorkload is a no-op on a MeshDevice without local devices, so there is nothing to prepare.
    TT_FATAL(!mesh_device.get_device_ids().empty(), "Cannot prepare a MeshWorkload for an inactive MeshDevice");
    // Checked before compile(), which finalizes the workload; a finalized workload rejects add_program(), so a later
    // check would leave the caller unable to fix the workload and retry.
    TT_FATAL(!workload.get_programs().empty(), "Cannot prepare a MeshWorkload that has no programs");
    workload.impl().compile(&mesh_device);
}

}  // namespace tt::tt_metal::experimental::program_preparation
