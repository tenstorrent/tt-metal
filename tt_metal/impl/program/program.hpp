// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <memory>
#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/impl/program/program_device_map.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

// Fwd declares
namespace detail{
    void ValidateCircularBufferRegion(const Program &program, const Device *device);
    KernelHandle AddKernel ( Program & program, std::shared_ptr<Kernel> kernel, const CoreType &core_type);
    std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id);
    std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id);
    void AddConfigBuffer(Program &program, std::shared_ptr<Buffer> config_buffer);
}

typedef std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX> kernel_id_array_t;

struct KernelGroup {
    CoreType core_type;
    CoreRangeSet core_ranges;
    kernel_id_array_t kernel_ids;
    uint32_t rta_sizes[DISPATCH_CLASS_MAX];
    uint32_t total_rta_size;
    launch_msg_t launch_msg;

    KernelGroup();
    KernelGroup(
        const Program &program,
        CoreType core_type,
        kernel_id_array_t kernel_ids,
        bool erisc_is_idle,
        int last_cb_index,
        const CoreRangeSet &new_ranges);

    CoreType get_core_type() const;
};

// Contains the program's worker memory map
struct ProgramConfig {
    uint32_t rta_offset;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_offsets;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_sizes;
};

// TODO: why is this in program.hpp
template <typename CoreRangeContainer>
vector<pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(Device* device, const CoreRangeContainer& ranges, const CoreType core_type) {
    // This API extracts all the pairs of noc multicast encodings given a set of core ranges
    vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info;
    dst_noc_multicast_info.reserve(ranges.size());
    for (const CoreRange& core_range : ranges) {
        CoreCoord physical_start = device->physical_core_from_logical_core(core_range.start_coord, core_type);
        CoreCoord physical_end = device->physical_core_from_logical_core(core_range.end_coord, core_type);

        uint32_t num_receivers = core_range.size();
        dst_noc_multicast_info.push_back(std::make_pair(CoreRange(physical_start, physical_end), num_receivers));
    }
    return dst_noc_multicast_info;
}

class Program {
    friend class KernelGroup;

   public:
    Program();

    Program(const Program &other) = delete;
    Program& operator=(const Program &other) = delete;

    Program(Program &&other) = default;
    Program& operator=(Program &&other) = default;

    ~Program();

    void construct_core_range_set_for_worker_cores();

    const uint64_t get_id() const { return this->id; }

    size_t num_kernels() const {
      size_t count = 0;
      for (const auto& [core_type, kernels] : kernels_) {
        count += kernels.size();
      }
      return count;
    }

    const std::vector<std::shared_ptr<CircularBuffer>> &circular_buffers() const { return circular_buffers_; }

    const std::vector< Semaphore > & semaphores() const { return semaphores_; }

    KernelGroup * kernels_on_core(const CoreCoord &core, const CoreType &core_type);
    std::vector<KernelGroup>& get_kernel_groups(const CoreType &core_type);
    inline void add_buffer(std::shared_ptr<Buffer> buf) { owned_buffer_pool.push_back(buf); }
    inline void release_buffers() { owned_buffer_pool = {}; }
    const std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_core(const CoreCoord &core) const;

    const std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_corerange(const CoreRange &cr) const;

    const std::vector<CoreRange> circular_buffers_unique_coreranges() const;

    auto semaphores_on_core(const CoreCoord &core) const {
        std::vector<std::reference_wrapper<const Semaphore>> semaphores;
        for ( const Semaphore & s : this->semaphores_) {
            if (s.initialized_on_logical_core(core)) {
                semaphores.emplace_back(std::cref(s));
            }
        }
        return semaphores;
    }

    size_t num_semaphores ( const CoreCoord & core ) const;
    size_t num_semaphores () const;
    void init_semaphores ( const Device & device, const CoreCoord &logical_core, const CoreType core_type) const;
    std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores() const;

    // Is worker_crs_ used anywhere?
    const CoreRangeSet& get_worker_core_range_set() const { return worker_crs_; };

    void compile(Device * device, bool fd_bootloader_mode = false);

    void invalidate_compile();

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers();

    bool is_finalized() const { return this->finalized_; }
    void finalize();
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    void capture_multi_device_dependencies() { capture_multi_device_dependencies_ = true; }
    bool has_multi_device_dependencies() { return capture_multi_device_dependencies_; }

   private:
    void populate_dispatch_data(Device *device);

    // Buffers temporarily owned by the program
    std::vector<std::shared_ptr<Buffer>> owned_buffer_pool = {};

    // The buffer that holds the kernel/binaries/etc for this program
    std::shared_ptr<Buffer> kernels_buffer = nullptr;
    ProgramTransferInfo program_transfer_info;

    bool finalized_;
    struct CircularBufferAllocator {
        CircularBufferAllocator(const CoreRange &core_range_) : core_range(core_range_) {}

        // Circular buffers are created and allocated at core range granularity
        CoreRange core_range;

        // Holds vector of addresses where circular buffers are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning multiple cores must share the same address
        // To enable this, circular buffer address is the maximum address amongst all of its target cores
        // This vector is sorted from lower to higher address spaces
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions = {{L1_UNRESERVED_BASE, L1_UNRESERVED_BASE}};

        // Returns address for next circular buffer
        // Circular buffers are placed sequentially on a core so the next available address gets appended to the last L1 region
        uint64_t get_cb_region_end() const {
            return this->l1_regions.back().second;
        }

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size);

        // Reset when circular buffer allocation is invalidated
        void reset_available_addresses() { this->l1_regions = {{L1_UNRESERVED_BASE, L1_UNRESERVED_BASE}}; }
    };

    uint64_t id; // Need to make non-const due to move constructor
    static std::atomic<uint64_t> program_counter;
    std::unordered_map<CoreType, std::unordered_map<KernelHandle, std::shared_ptr<Kernel> >> kernels_;
    std::unordered_map<CoreType, CoreCoord> grid_extent_;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_;
    std::unordered_map<CBHandle,  std::shared_ptr<CircularBuffer>> circular_buffer_by_id_;
    // Tracks which circular buffer indices are being used
    std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_cb_indices_;
    // Used to generate circular buffer addresses. There is one CircularBufferAllocator per unique CoreRange
    std::vector<CircularBufferAllocator> cb_allocators_;

    std::vector<Semaphore> semaphores_;

    CoreRangeSet worker_crs_;
    std::unordered_map<chip_id_t, bool> compile_needed_;
    bool local_circular_buffer_allocation_needed_;

    static constexpr uint8_t core_to_kernel_group_invalid_index = 0xff;
    std::unordered_map<CoreType, std::vector<KernelGroup>> kernel_groups_;
    std::unordered_map<CoreType, std::vector<uint8_t>> core_to_kernel_group_index_table_;
    uint32_t tensix_go_signal_count_;

    std::vector<std::shared_ptr<Buffer>> config_buffers_;

    std::vector<ProgramConfig> program_configs_;
    std::vector<uint32_t> program_config_sizes_;
    bool capture_multi_device_dependencies_ = false;
    friend CBHandle CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config);
    friend std::shared_ptr<CircularBuffer> detail::GetCircularBuffer(const Program &program, CBHandle id);
    friend void detail::ValidateCircularBufferRegion(const Program &program, const Device *device);

    friend KernelHandle detail::AddKernel(Program &program, std::shared_ptr<Kernel> kernel, const CoreType &core_type);
    friend std::shared_ptr<Kernel> detail::GetKernel(const Program &program, KernelHandle kernel_id);

    friend uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value, CoreType core_type);
    KernelHandle add_kernel(std::shared_ptr<Kernel> kernel, const CoreType &core_type);

    CBHandle add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);
    std::shared_ptr<CircularBuffer> get_circular_buffer(CBHandle cb_id) const;

    void add_semaphore(const CoreRangeSet & crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type=CoreType::WORKER);

    friend void detail::AddConfigBuffer(Program &program, std::shared_ptr<Buffer> config_buffer);
    void add_config_buffer(std::shared_ptr<Buffer> config_buffer);

    // Ensures that statically allocated circular buffers do not grow into L1 buffer space
    void validate_circular_buffer_region(const Device *device) const;

    void set_cb_data_fmt( Device *device, const std::vector<CoreRange> & crs, JitBuildOptions& build_options) const;

    void update_kernel_groups(const CoreType &core_type);

    ProgramConfig& get_program_config(CoreType core_type);
    uint32_t& get_program_config_size(CoreType core_type);

    uint32_t finalize_rt_args(CoreType core_type, uint32_t base_offset);

    friend class HWCommandQueue;
    friend class EnqueueProgramCommand;
};

}  // namespace tt_metal

}  // namespace tt
