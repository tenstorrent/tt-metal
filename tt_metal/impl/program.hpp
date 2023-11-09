/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <bitset>
#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"

// XXXX TODO(PGK): fix include paths so device can export interfaces
#include "tt_metal/src/firmware/riscv/common/dev_msgs.h"

namespace tt {

namespace tt_metal {

// Fwd declares
namespace detail{
    void ValidateCircularBufferRegion(const Program &program, const Device *device);
    KernelID AddKernel ( Program & program, Kernel * kernel);
    Kernel *GetKernel(const Program &program, KernelID kernel_id);
    std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CircularBufferID id);
}

struct KernelGroup {
    CoreRangeSet core_ranges;
    std::optional<KernelID> compute_id = std::nullopt;
    std::optional<KernelID> riscv0_id = std::nullopt;
    std::optional<KernelID> riscv1_id = std::nullopt;
    launch_msg_t launch_msg;

    KernelGroup();
    KernelGroup(const Program& program,
                std::optional<KernelID> brisc_id,
                std::optional<KernelID> ncrisc_id,
                std::optional<KernelID> trisc_id,
                int last_cb_index,
                const CoreRangeSet& new_ranges);
};

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

    size_t num_kernels() const { return kernels_.size(); }

    const std::vector<std::shared_ptr<CircularBuffer>> &circular_buffers() const { return circular_buffers_; }

    const std::vector< Semaphore > & semaphores() const { return semaphores_; }

    KernelGroup * kernels_on_core(const CoreCoord &core);

    std::vector<KernelGroup>& get_kernel_groups();

    const std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_core(const CoreCoord &core) const;

    const std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_corerange(const CoreRange &cr) const;

    auto semaphores_on_core(const CoreCoord &core) const;

    size_t num_semaphores ( const CoreCoord & core ) const;
    size_t num_semaphores () const;
    uint32_t semaphore_address ( uint32_t sem_index ) const;
    void init_semaphores ( const Device & device, const CoreCoord &logical_core ) const;
    std::vector<CoreCoord> logical_cores() const;

    const CoreRangeSet& get_worker_core_range_set() const { return worker_crs_; };

    std::vector<std::string> cores_to_ops() const;

    void compile(Device * device);

    void invalidate_compile();

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers();

   private:
    struct CircularBufferAllocator {
        // Tracks which circular buffer indices are being used
        std::bitset<NUM_CIRCULAR_BUFFERS> indices;

        // Holds vector of addresses where circular buffers are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning multiple cores must share the same address
        // To enable this, circular buffer address is the maximum address amongst all of its target cores
        // This vector is sorted from lower to higher address spaces
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions = {{L1_UNRESERVED_BASE, L1_UNRESERVED_BASE}};

        // Sets `indices` at position index
        void add_index(uint32_t index);

        // Returns address for next circular buffer
        // Circular buffers are placed sequentially on a core so the next available address gets appended to the last L1 region
        uint64_t get_address_candidate() const {
            return this->l1_regions.back().second;
        }

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size);

        uint64_t get_cb_region_end() const {
            return this->l1_regions.back().second;
        }

        // Reset when circular buffer allocation is invalidated
        void reset_available_addresses() { this->l1_regions = {{L1_UNRESERVED_BASE, L1_UNRESERVED_BASE}}; }
    };

    uint64_t id; // Need to make non-const due to move constructor
    static std::atomic<uint64_t> program_counter;
    std::vector<Kernel*> kernels_;
    CoreCoord grid_extent_;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_;
    std::unordered_map<CircularBufferID,  std::shared_ptr<CircularBuffer>> circular_buffer_by_id_;
    std::unordered_map<CoreCoord, CircularBufferAllocator> per_core_cb_allocator_;

    std::vector<Semaphore> semaphores_;

    CoreRangeSet worker_crs_;
    std::unordered_map<chip_id_t, bool> compile_needed_;
    bool local_circular_buffer_allocation_needed_;

    static constexpr uint8_t core_to_kernel_group_invalid_index = 0xff;
    std::vector<KernelGroup> kernel_groups_;
    std::vector<uint8_t> core_to_kernel_group_index_table_;

    friend CircularBufferID CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config);
    friend std::shared_ptr<CircularBuffer> detail::GetCircularBuffer(const Program &program, CircularBufferID id);
    friend void detail::ValidateCircularBufferRegion(const Program &program, const Device *device);

    friend KernelID detail::AddKernel(Program &program, Kernel *kernel);
    friend Kernel *detail::GetKernel(const Program &program, KernelID kernel_id);

    friend uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value);
    KernelID add_kernel(Kernel *kernel);
    Kernel *get_kernel(KernelID kernel_id) const;

    CircularBufferID add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);
    std::shared_ptr<CircularBuffer> get_circular_buffer(CircularBufferID cb_id) const;

    void add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value);

    // Ensures that statically allocated circular buffers do not grow into L1 buffer space
    void validate_circular_buffer_region(const Device *device) const;

    void set_cb_data_fmt( Device *device, Kernel *kernel, build_kernel_for_riscv_options_t &build_options) const;

    void update_kernel_groups();
};

}  // namespace tt_metal

}  // namespace tt
