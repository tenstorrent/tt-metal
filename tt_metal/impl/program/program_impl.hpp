// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "jit_build/build.hpp"
#include "program_command_sequence.hpp"

#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/command_queue.hpp"
#include "tt-metalium/core_coord.hpp"
#include "dev_msgs.h"                          // DISPATCH_CLASS_MAX
#include "tt-metalium/hal_types.hpp"           // HalProgrammableCoreType
#include "tt-metalium/kernel.hpp"              // Kernel
#include "tt-metalium/kernel_types.hpp"        // KernelHandle
#include "tt-metalium/program.hpp"             // KernelGroup
#include "program_device_map.hpp"              // ProgramTransferInfo
#include "tt-metalium/semaphore.hpp"
#include "tt-metalium/sub_device_types.hpp"
#include "tt_metal.hpp"

#include <umd/device/tt_core_coordinates.h>             // CoreType
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t

#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <tt_stl/span.hpp>

namespace tt {
namespace tt_metal {

class CircularBufferConfig;
class IDevice;
class JitBuildOptions;

namespace experimental {
class GlobalCircularBuffer;
}

namespace program_dispatch {

void assemble_device_commands(
    ProgramCommandSequence& program_command_sequence,
    detail::ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id,
    bool use_prefetcher_cache);
}

using kernel_id_array_t = std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX>;

struct KernelGroup {
    uint32_t programmable_core_type_index;
    CoreRangeSet core_ranges;
    kernel_id_array_t kernel_ids;
    uint32_t rta_sizes[DISPATCH_CLASS_MAX];
    uint32_t total_rta_size;
    uint32_t kernel_text_offsets[NUM_PROCESSORS_PER_CORE_TYPE];
    uint32_t kernel_bin_sizes[NUM_PROCESSORS_PER_CORE_TYPE];
    launch_msg_t launch_msg;
    go_msg_t go_msg;

    KernelGroup();
    KernelGroup(
        const detail::ProgramImpl& program,
        uint32_t programmable_core_type_index,
        kernel_id_array_t kernel_ids,
        bool erisc_is_idle,
        uint32_t local_cb_mask,
        uint32_t min_remote_cb_start_index,
        const CoreRangeSet& new_ranges);

    uint32_t get_programmable_core_type_index() const;

    CoreType get_core_type() const;
};

// Contains the program's worker memory map
struct ProgramConfig {
    uint32_t rta_offset;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_offsets;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_sizes;
    uint32_t sem_offset;
    uint32_t sem_size;
    uint32_t cb_offset;
    uint32_t cb_size;
    uint32_t local_cb_size;
    uint32_t kernel_text_offset;  // offset of first kernel bin
    uint32_t kernel_text_size;    // max size of all kernel bins across all kernel groups
};

namespace detail {

struct ProgramOffsetsState {
    // Base offset for Program Configs across all core types, wrt kernel config slot start address
    uint32_t config_base_offset = 0;
    // Incremental offset. Will correspond to the size of the program config per core, once the
    // program is finalized.
    uint32_t offset = 0;
    // Unique RTA offset.
    uint32_t rta_offset = 0;
    // Common RTA offsets and sizes.
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_offsets;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_sizes;
    // Semaphore offsets and sizes.
    uint32_t sem_offset = 0;
    uint32_t sem_size = 0;
    // CB offsets and sizes.
    uint32_t cb_offset = 0;
    uint32_t cb_size = 0;
    uint32_t local_cb_size = 0;
    // Kernel binary offsets and sizes.
    uint32_t kernel_text_offset = 0;
    uint32_t kernel_text_size = 0;
};

// Callable types for dependency injection
using KernelsGetter = std::function<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>&(uint32_t index)>;
using KernelGroupsGetter = std::function<std::vector<std::shared_ptr<KernelGroup>>&(uint32_t index)>;
using SemaphoresGetter = std::function<const std::vector<Semaphore>&()>;

// Internal class for holding a group of programs for parallel compilation.
class ProgramCompileGroup {
private:
    std::unordered_map<IDevice*, std::unique_ptr<Program>> program_device_map_;

public:
    ProgramCompileGroup() = default;

    ~ProgramCompileGroup();

    // Add a program to the compile group. Throws if the program already exists in the group.
    void add_program(IDevice* device, std::unique_ptr<Program> program);

    // Compiles all programs in the group
    void compile_all(bool force_slow_dispatch);

    // Write runtime args for all programs in the group
    void write_runtime_args(bool force_slow_dispatch);

    // Remove and return a program from the compile group
    std::unique_ptr<Program> remove_program(IDevice* device);

    void clear();

    bool contains(IDevice* device);
};

// The internal implementation of the Program class. Program is a view of this class that's usable by API clients.
class ProgramImpl : public std::enable_shared_from_this<ProgramImpl> {
public:
    ProgramImpl();

    ProgramImpl(const ProgramImpl& other) = delete;
    ProgramImpl& operator=(const ProgramImpl& other) = delete;

    ProgramImpl(ProgramImpl&& other) = default;
    ProgramImpl& operator=(ProgramImpl&& other) = default;

    ~ProgramImpl() noexcept;

    void set_runtime_id(uint64_t id);
    uint64_t get_runtime_id() const;
    uint64_t get_id() const;
    size_t num_kernels() const;
    const std::vector<std::shared_ptr<CircularBuffer>>& circular_buffers() const;
    const std::vector<Semaphore>& semaphores() const;
    KernelGroup* kernels_on_core(const CoreCoord& core, uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    void add_buffer(std::shared_ptr<Buffer> buf);
    void release_buffers();
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_core(const CoreCoord& core) const;
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_corerange(const CoreRange& cr) const;
    std::vector<CoreRange> circular_buffers_unique_coreranges() const;
    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(
        const CoreCoord& core, CoreType core_type) const;
    size_t num_semaphores() const;
    void init_semaphores(
        const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const;
    // XXXXX TODO: this should return a const reference
    std::vector<std::vector<CoreCoord>> logical_cores() const;
    void compile(IDevice* device, bool force_slow_dispatch = false);
    void invalidate_circular_buffer_allocation();
    void allocate_circular_buffers(const IDevice* device);
    uint32_t get_cb_memory_size() const;
    bool is_finalized() const;
    void set_finalized();
    void allocate_kernel_bin_buf_on_device(IDevice* device);
    bool is_cached() const { return this->cached_device_hash_.has_value(); }
    ProgramBinaryStatus get_program_binary_status(std::size_t device_id) const {
        if (auto it = this->binaries_on_device_.find(device_id); it != this->binaries_on_device_.end()) {
            return it->second;
        }
        return ProgramBinaryStatus::NotSent;
    }
    void set_cached(uint64_t device_hash) { this->cached_device_hash_ = device_hash; }
    const std::optional<uint64_t>& get_cached() const { return this->cached_device_hash_; }
    void set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status);
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;
    ProgramConfig& get_program_config(uint32_t programmable_core_type_index);
    const ProgramConfig& get_program_config(uint32_t programmable_core_type_index) const;
    const std::vector<SubDeviceId>& determine_sub_device_ids(const IDevice* device);

    void generate_trace_dispatch_commands(IDevice* device, bool use_prefetcher_cache);
    std::unordered_map<uint64_t, ProgramCommandSequence>& get_trace_cached_program_command_sequences() noexcept;

    // debug/test
    uint32_t get_sem_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const;
    uint32_t get_cb_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const;
    void set_last_used_command_queue_for_testing(CommandQueue* queue);
    CommandQueue* get_last_used_command_queue() const;
    void populate_dispatch_data(IDevice* device);

    void finalize_offsets(IDevice* device);

    // Helper function to finalize program offsets with custom getters. Returns the maximum kernel binaries size among
    // all the programs, to determine whether the mesh workload can fit in the prefetcher cache all of the programs in
    // it.
    static uint32_t finalize_program_offsets(
        IDevice* device,
        const KernelsGetter& kernels_getter,
        const KernelGroupsGetter& kernel_groups_getter,
        const SemaphoresGetter& semaphores_getter,
        tt::stl::Span<ProgramImpl*> programs);

    std::vector<uint32_t>& get_program_config_sizes() noexcept { return program_config_sizes_; }

private:
    CommandQueue* last_used_command_queue_for_testing = nullptr;

    // Buffers temporarily owned by the program
    std::vector<std::shared_ptr<Buffer>> owned_buffer_pool = {};

    // The buffer that holds the kernel/binaries/etc for this program
    std::unordered_map<chip_id_t, std::shared_ptr<Buffer>> kernels_buffer_;
    ProgramTransferInfo program_transfer_info;

    bool finalized_;
    // Used only when devices do not have virtualization enabled and used to check that programs are only rerun on
    // the same device
    std::optional<uint64_t> cached_device_hash_;

    // TODO: Should map based on the hash of the configured sub-devices
    // This way we can cache it agnostic of the device
    std::unordered_map<chip_id_t, std::unordered_map<SubDeviceManagerId, std::vector<SubDeviceId>>> sub_device_ids_;

    struct CircularBufferAllocator {
        CircularBufferAllocator(const CoreRange& core_range_) : core_range(core_range_) {}

        // Circular buffers are created and allocated at core range granularity
        CoreRange core_range;

        // Holds vector of addresses where circular buffers are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning
        // multiple cores must share the same address To enable this, circular buffer address is the maximum address
        // amongst all of its target cores This vector is sorted from lower to higher address spaces
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

        // Returns address for next circular buffer
        // Circular buffers are placed sequentially on a core so the next available address gets appended to the last
        // L1 region
        uint64_t get_cb_region_end() const { return this->l1_regions.empty() ? 0 : this->l1_regions.back().second; }

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size, uint64_t base_address);

        // Reset when circular buffer allocation is invalidated
        void reset_available_addresses() { this->l1_regions.clear(); }
    };
    uint32_t programmable_core_count_;
    uint64_t id;  // Need to make non-const due to move constructor
    uint64_t runtime_id;
    static std::atomic<uint64_t> program_counter;
    // Programmable core type index -> KernelHandle -> Kernel
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_;
    std::vector<CoreCoord> grid_extent_;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_;
    std::unordered_map<CBHandle, std::shared_ptr<CircularBuffer>> circular_buffer_by_id_;
    // Tracks which circular buffer indices are being used
    std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_cb_indices_;
    std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_local_cb_indices_;
    std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_remote_cb_indices_;
    std::unordered_map<std::size_t, ProgramBinaryStatus> binaries_on_device_;
    // Used to generate circular buffer addresses. There is one CircularBufferAllocator per unique CoreRange
    std::vector<CircularBufferAllocator> cb_allocators_;

    std::vector<Semaphore> semaphores_;

    std::unordered_set<uint32_t> compiled_;
    bool local_circular_buffer_allocation_needed_;

    static constexpr uint8_t core_to_kernel_group_invalid_index = 0xff;
    std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_;
    std::vector<std::vector<uint8_t>> core_to_kernel_group_index_table_;

    std::vector<ProgramConfig> program_configs_;
    // Counts how much space is needed for each core + each launch buffer msg queue.
    std::vector<uint32_t> program_config_sizes_;

    uint32_t kernel_bins_sizeB = 0;

    // The rta_updates from one cached command sequence may reference data in another cached command sequence.
    std::unordered_map<uint64_t, ProgramCommandSequence> cached_program_command_sequences_;
    std::unordered_map<uint64_t, ProgramCommandSequence> trace_cached_program_command_sequences_;

    friend std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program& program, CBHandle id);
    friend void ValidateCircularBufferRegion(const Program& program, const IDevice* device);

    friend KernelHandle AddKernel(
        Program& program, const std::shared_ptr<Kernel>& kernel, HalProgrammableCoreType core_type);

    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& core_type);

    CBHandle add_circular_buffer_(const std::shared_ptr<CircularBuffer>& circular_buffer);
    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);
    std::shared_ptr<CircularBuffer> get_circular_buffer(CBHandle cb_id) const;

    void add_semaphore(const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    // Ensures that statically allocated circular buffers do not grow into L1 buffer space
    void validate_circular_buffer_region(const IDevice* device);

    void set_remote_circular_buffer_init(const std::shared_ptr<Kernel>& kernel) const;

    void set_cb_data_fmt(const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const;

    void set_cb_tile_dims(const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const;

    void update_kernel_groups(uint32_t programmable_core_type_index);

    uint32_t& get_program_config_size(uint32_t programmable_core_type_index);

    void set_launch_msg_sem_offsets();

    bool runs_on_noc_unicast_only_cores();
    bool runs_on_noc_multicast_only_cores();
    bool kernel_binary_always_stored_in_ringbuffer();
    void set_program_offsets_and_sizes(uint32_t index, const ProgramOffsetsState& state);
    void set_program_attrs_across_core_types(IDevice* device);

    const ProgramTransferInfo& get_program_transfer_info() const noexcept;
    std::shared_ptr<Buffer> get_kernels_buffer(IDevice* device) const noexcept;

    friend void program_dispatch::assemble_device_commands(
        ProgramCommandSequence& program_command_sequence,
        ProgramImpl& program,
        IDevice* device,
        SubDeviceId sub_device_id,
        bool use_prefetcher_cache);

    friend HWCommandQueue;
    friend EnqueueProgramCommand;
    friend Program;
    friend Internal_;
    friend distributed::MeshWorkload;
    friend distributed::MeshWorkloadImpl;
};

}  // namespace detail
}  // namespace tt_metal
}  // namespace tt
