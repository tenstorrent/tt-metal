// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "program/program_impl.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"

namespace tt::tt_metal {
namespace distributed {
class MeshWorkloadImpl;
class MeshDevice;
}  // namespace distributed

namespace program_dispatch {
// NOLINTNEXTLINE(readability-redundant-declaration)
uint32_t program_base_addr_on_core(distributed::MeshWorkloadImpl&, distributed::MeshDevice*, HalProgrammableCoreType);
}  // namespace program_dispatch
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

// Per-device runtime args of one program of a MeshWorkload (see distributed::SetRuntimeArgs on a MeshWorkload):
// words [first_word, first_word + num_words) of one runtime-arg slot (a kernel's unique args on a core, or its
// common args) whose value is set per device. Other words, and devices without a value, keep the program's own args.
struct PerDeviceRuntimeArgsField {
    uint32_t kernel = 0;            // device-local kernel handle
    std::optional<CoreCoord> core;  // nullopt: the kernel's common runtime args
    uint32_t first_word = 0;
    uint32_t num_words = 0;
    std::vector<uint32_t> values;  // [coordinate index in the program's range][num_words]
    std::vector<uint8_t> present;  // [coordinate index in the program's range]
};
// A field's words in one command sequence: where they sit, and the field (values per device).
struct ResolvedPerDeviceRuntimeArgsField {
    const uint32_t* location;
    const PerDeviceRuntimeArgsField* field;
};

struct PerDeviceRuntimeArgs {
    uint64_t program_id = 0;                                         // the Program the fields were set for
    std::vector<std::unique_ptr<PerDeviceRuntimeArgsField>> fields;  // stable while others are added
    uint64_t generation = 0;                                         // bumps when a field is added
    // The fields resolved against the last command sequence enqueued (see resolve_per_device_runtime_args).
    struct ResolvedCache {
        uint64_t assembly_id = 0;
        uint64_t generation = 0;
        std::vector<ResolvedPerDeviceRuntimeArgsField> resolved;
    };
    mutable ResolvedCache resolved_cache;
    uint64_t values_version = 0;  // bumps when a value is set or cleared (or a field is added)
    // Each local device's copy of the region of the packed command that holds the fields (fast-dispatch eager
    // enqueue): words outside the fields track the packed command by delta, so a hit writes each device's command as
    // packed prefix + its region + packed suffix with no per-field work.
    // Like resolved_cache, it is only touched by enqueue, which holds the mesh device's API lock on every command
    // queue, and by the setters, which must be serialized with enqueue.
    struct DeviceImages {
        uint64_t assembly_id = 0;
        uint64_t generation = 0;
        uint64_t values_version = 0;
        bool stall_first = false;
        bool stall_before_program = false;
        uint32_t size_words = 0;
        std::vector<uint32_t> coord_indices;  // the devices the images are for, in write order
        uint32_t begin = 0;                   // region in the packed command, in words
        uint32_t words = 0;
        std::vector<uint32_t> snapshot;  // the packed region when the images were last brought up to date
        std::vector<uint8_t> in_field;   // per region word: a field covers it
        std::vector<uint32_t> images;    // [device][words]
    };
    mutable DeviceImages device_images;
};

// Linear index of `coord` in `range`, in the range's iteration order.
uint32_t coordinate_index_in_range(const MeshCoordinateRange& range, const MeshCoordinate& coord);
// Locations of every field (each copy of common args) in `sequence`; fails if a slot has no recorded location.
std::vector<ResolvedPerDeviceRuntimeArgsField> resolve_per_device_runtime_args(
    const PerDeviceRuntimeArgs& per_device_args, const ProgramCommandSequence& sequence);
// As above, cached on `per_device_args` for the sequence's assembly and the fields' generation.
const std::vector<ResolvedPerDeviceRuntimeArgsField>& resolve_per_device_runtime_args_cached(
    const PerDeviceRuntimeArgs& per_device_args, const ProgramCommandSequence& sequence);
// A copy of the fields and their current values (trace nodes and command lists keep the values of their capture).
std::shared_ptr<const PerDeviceRuntimeArgs> snapshot_per_device_runtime_args(
    const PerDeviceRuntimeArgs& per_device_args);
// The patches for the device at `coord_index`: each field's value on that device, or the sequence's own words.
void per_device_runtime_args_patches(
    const std::vector<ResolvedPerDeviceRuntimeArgsField>& resolved,
    uint32_t coord_index,
    std::vector<RtaPatch>& patches);

class MeshCommandQueue;
class FDMeshCommandQueue;

class MeshWorkloadImpl {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
    // The current iteration supports the following compute paradigms:
    //  - Single Program Multi Device (Completely Homogeneous MeshWorkload)
    //  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
    // Support for configurable runtime arguments will be added in future versions.
private:
    uint64_t id;

    void load_binaries(MeshCommandQueue& mesh_cq);
    void generate_dispatch_commands(MeshCommandQueue& mesh_cq);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::vector<Semaphore>& semaphores();
    const std::vector<uint32_t>& get_program_config_sizes();
    bool is_finalized() const { return finalized_metadata_.has_value(); }
    void set_finalized(uint32_t max_program_kernels_sizeB, int mesh_device_id);
    ProgramBinaryStatus get_program_binary_status(std::size_t mesh_id) const;
    void set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status);
    ProgramConfig& get_program_config(uint32_t index, bool using_fast_dispatch);
    ProgramCommandSequence& get_dispatch_cmds_for_program(Program& program, uint64_t command_hash);
    const std::vector<uint64_t>& get_cross_node_program_ids();
    void compile_program(const MeshCoordinateRange& device_range, MeshDevice* mesh_device);
    void finalize_offsets(MeshDevice* mesh_device);

    struct FinalizedMetadata {
        std::vector<uint32_t> program_config_sizes;
        std::vector<uint64_t> cross_node_program_ids;
        // Total devices covered by all program ranges. Fixed once programs_ is frozen, so enqueue
        // can compare it against the mesh size instead of re-walking every range.
        size_t num_program_devices = 0;
        uint32_t max_program_kernels_sizeB = 0;
        // Offsets and binary sizes are computed for this MeshDevice and are not recomputed for another.
        int mesh_device_id = 0;
        bool runs_on_noc_multicast_only_cores = false;
    };
    FinalizedMetadata& get_finalized_metadata();
    uint32_t get_max_program_kernels_sizeB() { return get_finalized_metadata().max_program_kernels_sizeB; }
    size_t get_num_program_devices() { return get_finalized_metadata().num_program_devices; }

    std::unordered_map<std::size_t, ProgramBinaryStatus> program_binary_status_;
    std::unordered_map<std::size_t, std::unordered_map<uint64_t, std::unordered_set<SubDeviceId>>>
        sub_device_ids_by_mesh_and_manager_;
    std::shared_ptr<MeshBuffer> kernel_bin_buf_;
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_;
    std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_;
    std::vector<Semaphore> semaphores_;
    std::unordered_map<MeshCoordinateRange, Program> programs_;
    // Presence marks the workload finalized and makes all derived enqueue metadata available together.
    std::optional<FinalizedMetadata> finalized_metadata_;
    // Per-device runtime args per program range (see PerDeviceRuntimeArgs); empty for almost every workload.
    std::unordered_map<MeshCoordinateRange, PerDeviceRuntimeArgs> per_device_runtime_args_;
    PerDeviceRuntimeArgsField& per_device_runtime_args_field(
        const MeshCoordinateRange& program_range,
        KernelHandle kernel,
        const std::optional<CoreCoord>& core,
        uint32_t first_word,
        uint32_t num_words,
        bool metal2 = false);
    MeshCommandQueue* last_used_command_queue_ = nullptr;

    // Cached service-vs-normal classification (see EnqueueMeshWorkload), computed before normal finalization.
    // Service workloads bypass finalization, so this state cannot live in finalized_metadata_. nullopt = not yet
    // classified. Re-enqueues skip the O(programs*coords*cores) no-mixing scan.
    // Classify-once holds because the classification depends only on core placement (fixed at build) and
    // the claimed-core set (worker cores never become service cores, and service workloads are only
    // re-enqueued onto still-claimed cores - which the dispatch path re-checks).
    std::optional<bool> is_service_workload_;

    friend uint32_t program_dispatch::program_base_addr_on_core(
        MeshWorkloadImpl&, ::tt::tt_metal::distributed::MeshDevice*, HalProgrammableCoreType);
    friend void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);
    friend FDMeshCommandQueue;
    friend class tt::tt_metal::Program;

    bool use_prefetcher_cache_ = false;

public:
    // Main User-Facing API building blocks
    MeshWorkloadImpl();
    ~MeshWorkloadImpl();

    uint64_t get_id() const { return id; }

    // Per-device runtime args (see distributed::SetRuntimeArgs on a MeshWorkload).
    void set_per_device_runtime_args(
        const MeshCoordinateRange& program_range,
        const MeshCoordinate& device,
        KernelHandle kernel,
        const std::optional<CoreCoord>& core,
        uint32_t first_word,
        ttsl::Span<const uint32_t> values);
    void set_per_device_runtime_args(
        const MeshCoordinateRange& program_range,
        KernelHandle kernel,
        const std::optional<CoreCoord>& core,
        uint32_t first_word,
        uint32_t num_words,
        ttsl::Span<const uint32_t> values_in_range_order);
    void clear_per_device_runtime_args(
        const MeshCoordinateRange& program_range,
        const MeshCoordinate& device,
        KernelHandle kernel,
        const std::optional<CoreCoord>& core,
        uint32_t first_word);
    // Metal 2.0 (ProgramSpec) programs: the words of named run args, resolved by
    // experimental::SetProgramRunArgsPerDevice. Same semantics as the legacy setter.
    void set_per_device_named_runtime_args(
        const MeshCoordinateRange& program_range,
        const MeshCoordinate& device,
        KernelHandle kernel,
        const std::optional<CoreCoord>& core,
        uint32_t first_word,
        ttsl::Span<const uint32_t> values);
    bool has_per_device_runtime_args() const { return !per_device_runtime_args_.empty(); }
    // The per-device runtime args of a range as they are, so a multi-field update can undo itself when a later field
    // fails (restoring invalidates any cached resolution of them).
    struct PerDeviceRuntimeArgsBackup {
        std::shared_ptr<const PerDeviceRuntimeArgs> snapshot;  // null: the range had none
        uint64_t generation = 0;
        uint64_t values_version = 0;
    };
    PerDeviceRuntimeArgsBackup backup_per_device_runtime_args(const MeshCoordinateRange& program_range) const;
    void restore_per_device_runtime_args(
        const MeshCoordinateRange& program_range, const PerDeviceRuntimeArgsBackup& backup);
    // The per-device runtime args of the program at `program_range`, or null; fails if the program there is not the
    // one they were set for.
    const PerDeviceRuntimeArgs* per_device_runtime_args(const MeshCoordinateRange& program_range) const;

    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs() { return programs_; }
    const std::unordered_map<MeshCoordinateRange, Program>& get_programs() const { return programs_; }
    void compile(MeshDevice* mesh_device);
    // Compile/finalize the workload and commit its kernel binaries to DRAM without executing it.
    // The returned owner must remain alive while commands referencing the binary addresses may execute.
    std::shared_ptr<MeshBuffer> prepare_for_command_list(MeshCommandQueue& mesh_cq);
    const std::unordered_set<SubDeviceId>& determine_sub_device_ids(MeshDevice* mesh_device);
    bool runs_on_noc_multicast_only_cores();
    uint32_t max_program_kernels_size() const {
        return finalized_metadata_.has_value() ? finalized_metadata_->max_program_kernels_sizeB : 0;
    }

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
};
}  // namespace tt::tt_metal::distributed
