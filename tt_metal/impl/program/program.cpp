// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/program_dispatch_utils.hpp"
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include "buffers/circular_buffer_types.hpp"
#include "common/executor.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/graph/graph_tracking.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/program.hpp"
#include "tracy/Tracy.hpp"

namespace tt::tt_metal {

namespace {
std::atomic<bool> enable_persistent_kernel_cache = false;

void GenerateBinaries(Device *device, JitBuildOptions &build_options, const std::shared_ptr<Kernel>& kernel) {
    //ZoneScoped;
    //const std::string tracyPrefix = "GenerateBinaries_";
    //ZoneName((tracyPrefix + build_options.name).c_str(), build_options.name.length() + tracyPrefix.length());
    try {
        jit_build_genfiles_descriptors(device->build_env(), build_options);
        kernel->generate_binaries(device, build_options);
    } catch (std::runtime_error &ex) {
        TT_THROW("Failed to generate binaries for {} {}", kernel->name(), ex.what());
    }
}

#ifdef GENERATE_HASH_LOG
#include <fstream>
#endif

size_t KernelCompileHash(const std::shared_ptr<Kernel>& kernel, JitBuildOptions &build_options, uint32_t build_key, size_t device_kernel_defines_hash) {
    // Store the build key into the KernelCompile hash. This will be unique per command queue
    // configuration (necessary for dispatch kernels).
    // Also account for watcher/dprint enabled in hash because they enable additional code to
    // be compiled into the kernel.
    string compile_hash_str = fmt::format(
        "{}_{}_{}_{}_{}",
        build_key,
        std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc)),
        kernel->compute_hash(),
        device_kernel_defines_hash,
        tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled());

    for (int i = 0; i < llrt::RunTimeDebugFeatureCount; i++) {
        compile_hash_str += "_";
        compile_hash_str += tt::llrt::RunTimeOptions::get_instance().get_feature_hash_string((llrt::RunTimeDebugFeatures)i);
    }
    size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

#ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        unique_lock<mutex> lock;
        f << kernel->name() << " :: " << build_key << "::" << std::hash<tt_hlk_desc>{}(build_options.hlk_desc)
          << " :: " << kernel->compute_hash() << " :: " << device_kernel_defines_hash << " :: " << compile_hash_str << " " << compile_hash << std::endl
          << std::flush;
    }
#endif
    return compile_hash;
}
}  // namespace
namespace detail {

class Program_ {
   public:
    Program_();

    Program_(const Program_ &other) = delete;
    Program_& operator=(const Program_ &other) = delete;

    Program_(Program_ &&other) = default;
    Program_& operator=(Program_ &&other) = default;

    void set_runtime_id(uint64_t id);
    ~Program_() noexcept = default;

    uint64_t get_id() const;
    uint64_t get_runtime_id() const;

    size_t num_kernels() const;

    const std::vector<std::shared_ptr<CircularBuffer>> &circular_buffers() const;

    const std::vector< Semaphore > & semaphores() const;

    KernelGroup * kernels_on_core(const CoreCoord &core, uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    void add_buffer(std::shared_ptr<Buffer> buf);
    void release_buffers();
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_core(const CoreCoord &core) const;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_corerange(const CoreRange &cr) const;

    std::vector<CoreRange> circular_buffers_unique_coreranges() const;

    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(const CoreCoord &core, CoreType core_type) const;

    size_t num_semaphores () const;
    void init_semaphores ( const Device & device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const;
    // XXXXX TODO: this should return a const reference
    std::vector<std::vector<CoreCoord>> logical_cores() const;

    void compile(Device * device, bool fd_bootloader_mode = false);

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers(const Device *device);

    bool is_finalized() const;
    void allocate_kernel_bin_buf_on_device(Device* device);
    void finalize(Device *device);
    bool is_cached() const { return this->cached_; }
    ProgramBinaryStatus get_program_binary_status(std::size_t device_id) const {
        if (auto it = this->binaries_on_device_.find(device_id); it != this->binaries_on_device_.end()) {
            return it->second;
        }
        return ProgramBinaryStatus::NotSent;
    }
    void set_cached() { this->cached_ = true; }
    void set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status) {
        this->binaries_on_device_[device_id] = status;
    }
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    ProgramConfig& get_program_config(uint32_t programmable_core_type_index);

    const std::vector<SubDeviceId> &determine_sub_device_ids(const Device *device);

    // debug/test
    uint32_t get_sem_base_addr(Device *device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(Device *device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(Device *device, CoreCoord logical_core, CoreType core_type) const;
    uint32_t get_cb_size(Device *device, CoreCoord logical_core, CoreType core_type) const;
    void set_last_used_command_queue_for_testing(HWCommandQueue *queue);
    void populate_dispatch_data(Device *device);

   private:
    HWCommandQueue *last_used_command_queue_for_testing = nullptr;

    // Buffers temporarily owned by the program
    std::vector<std::shared_ptr<Buffer>> owned_buffer_pool = {};

    // The buffer that holds the kernel/binaries/etc for this program
    std::unordered_map<chip_id_t, std::shared_ptr<Buffer>> kernels_buffer_;
    ProgramTransferInfo program_transfer_info;

    bool finalized_;
    bool cached_;

    std::unordered_map<SubDeviceManagerId, std::vector<SubDeviceId>> sub_device_ids_;

    struct CircularBufferAllocator {
        CircularBufferAllocator(const CoreRange &core_range_) : core_range(core_range_) {}

        // Circular buffers are created and allocated at core range granularity
        CoreRange core_range;

        // Holds vector of addresses where circular buffers are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning multiple cores must share the same address
        // To enable this, circular buffer address is the maximum address amongst all of its target cores
        // This vector is sorted from lower to higher address spaces
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

        // Returns address for next circular buffer
        // Circular buffers are placed sequentially on a core so the next available address gets appended to the last L1 region
        uint64_t get_cb_region_end() const {
            return this->l1_regions.empty() ? 0 : this->l1_regions.back().second;
        }

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size, uint64_t base_address);

        // Reset when circular buffer allocation is invalidated
        void reset_available_addresses() { this->l1_regions.clear(); }
    };

    uint64_t id; // Need to make non-const due to move constructor
    uint64_t runtime_id;
    static std::atomic<uint64_t> program_counter;
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel> >> kernels_;
    std::vector<CoreCoord> grid_extent_;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_;
    std::unordered_map<CBHandle,  std::shared_ptr<CircularBuffer>> circular_buffer_by_id_;
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

    std::vector<std::shared_ptr<Buffer>> config_buffers_;

    std::vector<ProgramConfig> program_configs_;
    // Counts how much space is needed for each core + each launch buffer msg queue.
    std::vector<uint32_t> program_config_sizes_;

    std::unordered_map<uint64_t, ProgramCommandSequence> cached_program_command_sequences_;

    friend std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id);
    friend void ValidateCircularBufferRegion(const Program &program, const Device *device);

    friend KernelHandle AddKernel(Program &program, const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType core_type);

    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType &core_type);

    CBHandle add_circular_buffer_(
        const CoreRangeSet& core_range_set, const std::shared_ptr<CircularBuffer>& circular_buffer);
    CBHandle add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const v1::experimental::GlobalCircularBuffer& global_circular_buffer);
    std::shared_ptr<CircularBuffer> get_circular_buffer(CBHandle cb_id) const;

    void add_semaphore(const CoreRangeSet & crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    friend void AddConfigBuffer(Program &program, const std::shared_ptr<Buffer>& config_buffer);
    void add_config_buffer(const std::shared_ptr<Buffer>& config_buffer);

    // Ensures that statically allocated circular buffers do not grow into L1 buffer space
    void validate_circular_buffer_region(const Device *device);

    void set_remote_circular_buffer_init(const std::shared_ptr<Kernel>& kernel) const;

    void set_cb_data_fmt(const std::vector<CoreRange> & crs, JitBuildOptions& build_options) const;

    void set_cb_tile_dims(const std::vector<CoreRange> & crs, JitBuildOptions& build_options) const;

    void update_kernel_groups(uint32_t programmable_core_type_index);

    uint32_t& get_program_config_size(uint32_t programmable_core_type_index);

    uint32_t finalize_rt_args(uint32_t programmable_core_type_index, uint32_t base_offset);
    uint32_t finalize_sems(uint32_t programmable_core_type_index, uint32_t base_offset);
    uint32_t finalize_cbs(uint32_t programmable_core_type_index, uint32_t base_offset);
    uint32_t finalize_kernel_bins(Device *device, uint32_t programmable_core_type_index, uint32_t base_offset);
    void set_launch_msg_sem_offsets();

    bool runs_on_noc_unicast_only_cores();
    bool runs_on_noc_multicast_only_cores();
    bool kernel_binary_always_stored_in_ringbuffer();

    friend HWCommandQueue;
    friend EnqueueProgramCommand;
    friend Program;
    friend Internal_;
};

KernelHandle AddKernel (Program &program, const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType core_type) {
    return program.pimpl_->add_kernel(std::move(kernel), core_type);
}

std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id) {
    return program.get_kernel(kernel_id);
}

std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id) {
    return program.pimpl_->get_circular_buffer(id);
}

// Checks that circular buffers do not grow into L1 buffer space
void ValidateCircularBufferRegion(const Program &program, const Device *device) {
    program.pimpl_->validate_circular_buffer_region(device);
}

void AddConfigBuffer(Program &program, const std::shared_ptr<Buffer>& config_buffer) {
    program.pimpl_->add_config_buffer(std::move(config_buffer));
}

void EnablePersistentKernelCache() { enable_persistent_kernel_cache = true; }

void DisablePersistentKernelCache() { enable_persistent_kernel_cache = false; }

class Internal_ {
   public:
    using map_type = decltype(detail::Program_::circular_buffer_by_id_);

    static const map_type &get_circular_buffers_by_id(const Program &program) noexcept {
        return program.pimpl_->circular_buffer_by_id_;
    }
};

}  // namespace detail

std::atomic<uint64_t> detail::Program_::program_counter = 0;

detail::Program_::Program_() :
    id(program_counter++),
    runtime_id(0),
    local_circular_buffer_allocation_needed_(false),
    finalized_(false),
    cached_(false) {

    uint32_t programmable_core_count = hal.get_programmable_core_type_count();
    for (uint32_t i = 0; i < programmable_core_count; i++) {
        kernels_.push_back({});
        grid_extent_.push_back({});
        kernel_groups_.push_back({});
        core_to_kernel_group_index_table_.push_back({});
    }

    program_configs_.resize(programmable_core_count);
    program_config_sizes_.resize(programmable_core_count + 2);
}

Program::Program() : pimpl_(std::make_unique<detail::Program_>()) {}

KernelHandle detail::Program_::add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType &programmable_core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add kernel to an already compiled program {}", this->id);
    // Id is unique across all kernels on all core types
    KernelHandle id = this->num_kernels();
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);
    kernels_[index].insert({id, kernel});
    kernel_groups_[index].resize(0);
    core_to_kernel_group_index_table_[index].clear();
    return id;
}

std::shared_ptr<Kernel> detail::Program_::get_kernel(KernelHandle kernel_id) const {
    // TT_ASSERT(kernel_id < this->kernels_.size(), "Expected Kernel with ID {} to be in Program {}", kernel_id,
    // this->id);
    //  find coretype based on kernel_id
    for (const auto &kernels : this->kernels_) {
        if (kernels.find(kernel_id) != kernels.end()) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

std::shared_ptr<Kernel> Program::get_kernel(KernelHandle kernel_id) const { return pimpl_->get_kernel(kernel_id); }

KernelGroup::KernelGroup() : core_ranges(CoreRangeSet()) {}

KernelGroup::KernelGroup(
    const detail::Program_& program,
    uint32_t programmable_core_type_index,
    kernel_id_array_t kernel_ids,
    bool erisc_is_idle,
    uint32_t max_local_cb_end_index,
    uint32_t min_remote_cb_start_index,
    const CoreRangeSet& new_ranges) :
    core_ranges(CoreRangeSet()) {
    this->programmable_core_type_index = programmable_core_type_index;
    this->core_ranges = this->core_ranges.merge(new_ranges);
    this->kernel_ids = kernel_ids;
    this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DEDICATED_NOC;

    std::memset(&this->launch_msg, 0, sizeof(launch_msg_t));

    // Slow dispatch uses fixed addresses for the kernel config, configured here statically
    // Fast dispatch kernel config mangement happens under the CQ and will re-program the base
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        this->launch_msg.kernel_config.kernel_config_base[index] =
            hal.get_dev_addr(index, HalL1MemAddrType::KERNEL_CONFIG);
    }

    uint32_t processor_classes = hal.get_processor_classes_count(programmable_core_type_index);
    for (int class_id = 0; class_id < processor_classes; class_id++) {
        auto& optional_id = kernel_ids[class_id];
        if (optional_id) {
            const auto kernel = program.get_kernel(optional_id.value());
            this->launch_msg.kernel_config.watcher_kernel_ids[class_id] = kernel->get_watcher_kernel_id();
            this->launch_msg.kernel_config.enables |= 1 << class_id;

            if (programmable_core_type_index == hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)) {
                // The code below sets the brisc_noc_id for use by the device firmware
                // Use 0 if neither brisc nor ncrisc specify a noc
                if (class_id == utils::underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_0)) {
                    // Use brisc's noc if brisc specifies a noc
                    this->launch_msg.kernel_config.brisc_noc_id = std::get<DataMovementConfig>(kernel->config()).noc;
                    // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to DM_DEDICATED_NOC
                    if (std::get<DataMovementConfig>(kernel->config()).noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                        this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
                    }
                } else if (class_id == utils::underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_1)) {
                    // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
                    // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
                    this->launch_msg.kernel_config.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
                    // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to DM_DEDICATED_NOC
                    if (this->launch_msg.kernel_config.brisc_noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                        this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
                    }
                }
            }
        }
    }

    for (uint32_t index = 0; index < NUM_PROCESSORS_PER_CORE_TYPE; index ++) {
        this->kernel_bin_sizes[index] = 0;
        this->kernel_text_offsets[index] = 0;
        this->launch_msg.kernel_config.kernel_text_offset[index] = 0;
    }
    this->launch_msg.kernel_config.ncrisc_kernel_size16 = 0;

    this->launch_msg.kernel_config.exit_erisc_kernel = false;
    this->launch_msg.kernel_config.max_local_cb_end_index = max_local_cb_end_index;
    this->launch_msg.kernel_config.min_remote_cb_start_index = min_remote_cb_start_index;
    this->go_msg.signal = RUN_MSG_GO;
}

CoreType KernelGroup::get_core_type() const {
    return hal.get_core_type(this->programmable_core_type_index);
};

std::vector<std::shared_ptr<KernelGroup>> &detail::Program_::get_kernel_groups(uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    return kernel_groups_[programmable_core_type_index];
}

std::vector<std::shared_ptr<KernelGroup>> &Program::get_kernel_groups(uint32_t programmable_core_type_index) {
    return pimpl_->get_kernel_groups(programmable_core_type_index);
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& detail::Program_::get_kernels(uint32_t programmable_core_type_index) {
    return this->kernels_.at(programmable_core_type_index);
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& Program::get_kernels(uint32_t programmable_core_type_index) {
    return pimpl_->get_kernels(programmable_core_type_index);
}

KernelGroup *detail::Program_::kernels_on_core(const CoreCoord &core, uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    if (core.x >= grid_extent_[programmable_core_type_index].x || core.y >= grid_extent_[programmable_core_type_index].y)
        return nullptr;
    uint8_t index = core_to_kernel_group_index_table_[programmable_core_type_index].at(core.y * grid_extent_[programmable_core_type_index].x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : kernel_groups_[programmable_core_type_index].at(index).get();
}

KernelGroup *Program::kernels_on_core(const CoreCoord &core, uint32_t programmable_core_type_index) {
    return pimpl_->kernels_on_core(core, programmable_core_type_index);
}

struct KernelGroupInt {
    bool valid;
    kernel_id_array_t kernel_ids;

    bool operator==(const KernelGroupInt &b) const;
    // fix this
    void update(dispatch_core_processor_classes proc_class, size_t kernel_idx) {
        this->kernel_ids[proc_class] = static_cast<KernelHandle>(kernel_idx);
    }
};

bool KernelGroupInt::operator==(const KernelGroupInt &b) const {
    for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
        if (this->kernel_ids[class_id] != b.kernel_ids[class_id]) {
            return false;
        }
    }

    return true;
}

struct KernelGroupIntHasher {
    std::size_t operator()(const KernelGroupInt &x) const {
        return
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value_or(0)) << 0 |
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value_or(0)) << 16 |
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value_or(0)) << 32;
    }
};

void detail::Program_::update_kernel_groups(uint32_t programmable_core_type_index) {
    if (core_to_kernel_group_index_table_[programmable_core_type_index].size() == 0) {
        bool erisc_is_idle = false;

        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(), std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_[programmable_core_type_index] = {0, 0};
        for (auto [id, kernel] : kernels_[programmable_core_type_index]) {
            for (auto core : kernel->logical_cores()) {
                if (core.x > grid_extent_[programmable_core_type_index].x)
                    grid_extent_[programmable_core_type_index].x = core.x;
                if (core.y > grid_extent_[programmable_core_type_index].y)
                    grid_extent_[programmable_core_type_index].y = core.y;
                if (core.x < base.x)
                    base.x = core.x;
                if (core.y < base.y)
                    base.y = core.y;
            }
            erisc_is_idle = kernel->is_idle_eth();
        }
        grid_extent_[programmable_core_type_index].x++;
        grid_extent_[programmable_core_type_index].y++;

        // grid maps cores to sets-of-kernels running on that core
        std::vector<KernelGroupInt> grid;
        grid.resize(grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y);
        for (auto [id, kernel] : kernels_[programmable_core_type_index]) {
            for (auto core : kernel->logical_cores()) {
                int core_index = core.y * grid_extent_[programmable_core_type_index].x + core.x;
                grid[core_index].valid = true;
                grid[core_index].update(magic_enum::enum_cast<dispatch_core_processor_classes>(kernel->dispatch_class()).value(), id);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::unordered_map<KernelGroupInt, std::set<CoreRange>, KernelGroupIntHasher> map;
        for (auto y = base.y; y < grid_extent_[programmable_core_type_index].y; y++) {
            for (auto x = base.x; x < grid_extent_[programmable_core_type_index].x; x++) {
                int index = y * grid_extent_[programmable_core_type_index].x + x;
                if (grid[index].valid) {
                    std::set<CoreRange> &set = map[grid[index]];
                    set.insert(CoreRange({x, y}, {x, y}));
                }
            }
        }

        // Build the list of KernelGroups with merged core range sets from the
        // mapping of sets-of-kernels to cores
        TT_ASSERT(map.size() < core_to_kernel_group_invalid_index);
        kernel_groups_.reserve(map.size());
        int index = 0;
        core_to_kernel_group_index_table_[programmable_core_type_index].resize(
            grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y, core_to_kernel_group_invalid_index);
        for (auto &kg_to_cores : map) {
            // Start inclusive, max exclusive
            uint32_t max_local_cb_end_index = 0;
            uint32_t min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : kg_to_cores.second) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                        core_to_kernel_group_index_table_[programmable_core_type_index][y * grid_extent_[programmable_core_type_index].x + x] = index;

                        if (not hal.get_supports_cbs(programmable_core_type_index)) continue;
                        auto core = CoreCoord({x, y});
                        auto local_val = per_core_local_cb_indices_.find(core);
                        if (local_val != per_core_local_cb_indices_.end()) {
                            max_local_cb_end_index = std::max(
                                max_local_cb_end_index,
                                NUM_CIRCULAR_BUFFERS - (uint32_t)__builtin_clz(local_val->second.to_ulong()));
                        }
                        auto remote_val = per_core_remote_cb_indices_.find(core);
                        if (remote_val != per_core_remote_cb_indices_.end()) {
                            min_remote_cb_start_index = std::min(
                                min_remote_cb_start_index, (uint32_t)__builtin_ctz(remote_val->second.to_ulong()));
                        }
                    }
                }
            }
            TT_FATAL(
                max_local_cb_end_index <= min_remote_cb_start_index,
                "Circular buffer indices overlap for KernelGroup {} on programmable core type {}. Local end index {}, "
                "Remote start index {}",
                index,
                programmable_core_type_index,
                max_local_cb_end_index,
                min_remote_cb_start_index);
            kernel_groups_[programmable_core_type_index].push_back(
                std::make_shared<KernelGroup>(
                    *this,
                    programmable_core_type_index,
                    kg_to_cores.first.kernel_ids,
                    erisc_is_idle,
                    max_local_cb_end_index,
                    min_remote_cb_start_index,
                    kg_to_cores.second)
                );
            index++;
        }
    }

}

void detail::Program_::CircularBufferAllocator::mark_address(uint64_t address, uint64_t size, uint64_t base_address) {
    if (this->l1_regions.empty()) {
        this->l1_regions.emplace_back(base_address, base_address);
    }
    auto &last_region = this->l1_regions.back();
    if (address < last_region.second) {
        TT_THROW(
            "Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address",
            address,
            last_region.first,
            last_region.second);
    }
    if (address == last_region.second) {
        last_region.second += size;
    } else {
        this->l1_regions.emplace_back(address, address + size);
    }
}

CBHandle detail::Program_::add_circular_buffer_(
    const CoreRangeSet& core_range_set, const std::shared_ptr<CircularBuffer>& circular_buffer) {
    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory
    // allocator
    if (not circular_buffer->globally_allocated()) {
        this->invalidate_circular_buffer_allocation();
    } else {
        circular_buffer->assign_global_address();
    }
    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange &core_range : core_range_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS> &cb_indices = this->per_core_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& local_cb_indices = this->per_core_local_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& remote_cb_indices = this->per_core_remote_cb_indices_[logical_core];
                auto add_buffer_indices = [&cb_indices](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
                        // TT_ASSERT since we validate when constructing the config that it's within range
                        TT_ASSERT(
                            buffer_index < NUM_CIRCULAR_BUFFERS,
                            "Invalid circular buffer index: {} should be between 0 and {}",
                            buffer_index,
                            NUM_CIRCULAR_BUFFERS);
                        if (cb_indices[buffer_index]) {
                            TT_THROW(
                                "Invalid circular buffer index: Cannot add circular buffer at index {}, another "
                                "circular "
                                "buffer already exists",
                                buffer_index);
                        }
                        cb_indices[buffer_index] = 1;
                        target_cb_indices[buffer_index] = 1;
                    }
                };
                add_buffer_indices(circular_buffer->config().local_buffer_indices(), local_cb_indices);
                add_buffer_indices(circular_buffer->config().remote_buffer_indices(), remote_cb_indices);
            }
        }

        // There is one CircularBufferAllocator per unique core range, create one if it does not already exist for
        // current core range
        auto val = std::find_if(
            cb_allocators_.begin(), cb_allocators_.end(), [&core_range](const CircularBufferAllocator &cb_allocator) {
                return cb_allocator.core_range == core_range;
            });
        if (val == cb_allocators_.end()) {
            this->cb_allocators_.emplace_back(core_range);
        }
    }

    this->circular_buffers_.push_back(circular_buffer);
    this->circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

CBHandle detail::Program_::add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config) {
    TT_FATAL(this->compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", this->id);
    std::shared_ptr<CircularBuffer> circular_buffer = std::make_shared<CircularBuffer>(core_range_set, config);
    return add_circular_buffer_(core_range_set, circular_buffer);
}

CBHandle detail::Program_::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const v1::experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(this->compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", this->id);
    std::shared_ptr<CircularBuffer> circular_buffer =
        std::make_shared<CircularBuffer>(core_range_set, config, global_circular_buffer);
    return add_circular_buffer_(core_range_set, circular_buffer);
}

CBHandle Program::add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config) {
    return pimpl_->add_circular_buffer(core_range_set, config);
}

CBHandle Program::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const v1::experimental::GlobalCircularBuffer& global_circular_buffer) {
    return pimpl_->add_circular_buffer(core_range_set, config, global_circular_buffer);
}

std::shared_ptr<CircularBuffer> detail::Program_::get_circular_buffer(CBHandle cb_id) const {
    if (this->circular_buffer_by_id_.find(cb_id) == this->circular_buffer_by_id_.end()) {
        TT_THROW("No circular buffer with id {} exists in Program {}", cb_id, this->id);
    }
    return this->circular_buffer_by_id_.at(cb_id);
}

std::vector<std::shared_ptr<CircularBuffer>> detail::Program_::circular_buffers_on_core(const CoreCoord &core) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (const auto& circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_core(const CoreCoord &core) const {
    return pimpl_->circular_buffers_on_core(core);
}

std::vector<std::shared_ptr<CircularBuffer>> detail::Program_::circular_buffers_on_corerange(const CoreRange &cr) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (const auto& circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_corerange(cr)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_corerange(const CoreRange &cr) const {
    return pimpl_->circular_buffers_on_corerange(cr);
}

std::vector<CoreRange> detail::Program_::circular_buffers_unique_coreranges() const {
    std::vector<CoreRange> core_ranges;
    for (const auto& circular_buffer : circular_buffers_) {
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            if (std::find(core_ranges.begin(), core_ranges.end(), core_range) == core_ranges.end()) {
                core_ranges.push_back(core_range);
            }
        }
    }
    return core_ranges;
}

std::vector<CoreRange> Program::circular_buffers_unique_coreranges() const {
    return pimpl_->circular_buffers_unique_coreranges();
}

void detail::Program_::invalidate_circular_buffer_allocation() {
    if (this->local_circular_buffer_allocation_needed_) {
        return;
    }
    for (CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        cb_allocator.reset_available_addresses();
    }
    this->local_circular_buffer_allocation_needed_ = true;
}

void Program::invalidate_circular_buffer_allocation() { pimpl_->invalidate_circular_buffer_allocation(); }

void detail::Program_::allocate_circular_buffers(const Device *device) {
    //ZoneScoped;
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_cb_address = device->get_base_allocator_addr(HalMemType::L1);
    for (const auto& circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
            continue;
        }

        uint64_t computed_addr = base_cb_address;
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            // Need the max available address across all cores circular buffer is placed on
            for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
                if (cb_allocator.core_range == core_range) {
                    computed_addr = std::max(computed_addr, cb_allocator.get_cb_region_end());
                    break;
                }
            }
        }
        computed_addr = align(computed_addr, device->get_allocator_alignment());
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            for (CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
                if (cb_allocator.core_range.intersects(core_range)) {
                    if (cb_allocator.core_range != core_range and computed_addr < cb_allocator.get_cb_region_end()) {
                        // Intersecting core range has already been marked to have allocation at this address. This
                        // could have been marked by a circular buffer on a core range disjoint from current
                        // `core_range` but also intersecting `cb_allocator.core_range`
                        continue;
                    }
                    cb_allocator.mark_address(computed_addr, circular_buffer->size(), base_cb_address);
                }
            }
        }
        tt::tt_metal::GraphTracker::instance().track_allocate_cb(circular_buffer->core_ranges(), computed_addr, circular_buffer->size(), circular_buffer->globally_allocated());
        circular_buffer->set_locally_allocated_address(computed_addr);
    }
    this->local_circular_buffer_allocation_needed_ = false;
}

void Program::allocate_circular_buffers(const Device *device) { pimpl_->allocate_circular_buffers(device); }

void detail::Program_::validate_circular_buffer_region(const Device *device) {
    //ZoneScoped;

    // TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device
    std::optional<DeviceAddr> lowest_address = device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        if (cb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t cb_region_end = cb_allocator.l1_regions.back().second; //cb_allocator.get_cb_region_end();
        if (cb_region_end > max_l1_size) {
            TT_THROW(
                "Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} "
                "B",
                cb_allocator.core_range.str(),
                cb_region_end,
                max_l1_size);
        }
        if (lowest_address.has_value() and lowest_address.value() < cb_region_end) {
            TT_THROW(
                "Statically allocated circular buffers in program {} clash with L1 buffers on core range {}. L1 buffer "
                "allocated at {} and static circular buffer region ends at {}",
                this->id,
                cb_allocator.core_range.str(),
                lowest_address.value(),
                cb_region_end);
        }
    }
}

size_t Program::num_semaphores(const CoreCoord &core, CoreType core_type) const { return semaphores_on_core(core, core_type).size(); }

size_t detail::Program_::num_semaphores() const { return semaphores_.size(); }

size_t Program::num_semaphores() const { return pimpl_->num_semaphores(); }

void detail::Program_::init_semaphores(const Device &device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const {

    uint64_t kernel_config_base = hal.get_dev_addr(programmable_core_type_index, HalL1MemAddrType::KERNEL_CONFIG);
    uint64_t addr = kernel_config_base + this->program_configs_[programmable_core_type_index].sem_offset;
    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    auto semaphores_on_core = this->semaphores_on_core(logical_core, core_type);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(
            device.id(),
            device.virtual_core_from_logical_core(logical_core, core_type),
            std::vector{semaphore.get().initial_value()},
            addr + semaphore.get().offset());
    }
}

void Program::init_semaphores(const Device &device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const {
    pimpl_->init_semaphores(device, logical_core, programmable_core_type_index);
}

void detail::Program_::add_semaphore(const CoreRangeSet &crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add semaphore to an already compiled program {}", this->id);
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, core_type));
}

void Program::add_semaphore(const CoreRangeSet &crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    pimpl_->add_semaphore(crs, semaphore_id, init_value, core_type);
}

void detail::Program_::add_config_buffer(const std::shared_ptr<Buffer>& config_buffer) { config_buffers_.emplace_back(config_buffer); }

std::vector<std::vector<CoreCoord>> detail::Program_::logical_cores() const {
    std::vector<std::vector<CoreCoord>> cores_in_program;
    std::vector<std::set<CoreCoord>> unique_cores;
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < kernels_.size(); programmable_core_type_index++) {
        auto &kernels = this->kernels_[programmable_core_type_index];
        cores_in_program.push_back({});
        unique_cores.push_back({});
        for (auto [id, kernel] : kernels) {
            for (auto core : kernel->logical_cores()) {
                if (unique_cores[programmable_core_type_index].find(core) != unique_cores[programmable_core_type_index].end()) {
                    continue;
                }
                unique_cores[programmable_core_type_index].insert(core);
                cores_in_program[programmable_core_type_index].push_back(core);
            }
        }
    }
    return cores_in_program;
}

std::vector<std::vector<CoreCoord>> Program::logical_cores() const { return pimpl_->logical_cores(); }

void detail::Program_::set_remote_circular_buffer_init(const std::shared_ptr<Kernel>& kernel) const {
    const auto& kernel_defines = kernel->defines();
    const std::string reserved_defines[] = {"ALIGN_LOCAL_CBS_TO_REMOTE_CBS"};
    for (const auto& str : reserved_defines) {
        TT_FATAL(
            kernel_defines.find(str) == kernel_defines.end(), "{} is a reserved define and can't be manually set", str);
    }
    std::string align_code = "";
    std::unordered_set<CBHandle> initialized_cbs;
    std::unordered_set<uint8_t> remote_cb_indices;
    for (auto logical_cr : kernel->logical_coreranges()) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            if (circular_buffer->remote_buffer_indices().empty() || initialized_cbs.contains(circular_buffer->id())) {
                continue;
            }
            initialized_cbs.insert(circular_buffer->id());
            auto remote_cb_index = *circular_buffer->remote_buffer_indices().begin();
            remote_cb_indices.insert(remote_cb_index);

            // We only need the first remote buffer index
            if (!circular_buffer->local_buffer_indices().empty()) {
                align_code += fmt::format(
                    "experimental::align_local_cbs_to_remote_cb<{}>({},{{",
                    circular_buffer->local_buffer_indices().size(),
                    remote_cb_index);
                for (auto buffer_index : circular_buffer->local_buffer_indices()) {
                    align_code += fmt::format("{},", buffer_index);
                }
                align_code.back() = '}';
                align_code.append(");");
            }
        }
    }
    if (!remote_cb_indices.empty()) {
        std::map<std::string, std::string> defines;
        if (!align_code.empty()) {
            defines["ALIGN_LOCAL_CBS_TO_REMOTE_CBS"] = align_code;
        }
        kernel->add_defines(defines);
    }
}

void detail::Program_::set_cb_data_fmt(const std::vector<CoreRange> &crs, JitBuildOptions &build_options) const {
    //ZoneScoped;
    for (const auto& logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(
                    static_cast<CBIndex>(buffer_index), circular_buffer->data_format(buffer_index));
            }
        }
    }
}

void detail::Program_::set_cb_tile_dims(const std::vector<CoreRange> &crs, JitBuildOptions &build_options) const {
    //ZoneScoped;
    for (const auto &logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto &circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                auto tile = circular_buffer->tile(buffer_index);
                if (tile.has_value()) {
                    build_options.set_cb_tile_dims_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        tile->get_num_faces(),
                        tile->get_partial_face(),
                        tile->get_face_shape()[0],
                        tile->get_narrow_tile(),
                        tile->get_tile_shape()[0],
                        tile->get_tile_shape()[1]);
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        tile->get_tile_size(circular_buffer->data_format(buffer_index)));
                } else {
                    Tile t;
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        t.get_tile_size(circular_buffer->data_format(buffer_index)));
                }

            }
        }
    }
}

void detail::Program_::populate_dispatch_data(Device *device) {
    auto extract_dst_noc_unicast_info =
        [&device](const auto &ranges, const CoreType core_type) -> std::vector<std::pair<transfer_info_cores, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info;
        for (const CoreRange &core_range : ranges) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord virtual_coord = device->virtual_core_from_logical_core(CoreCoord({x, y}), core_type);
                    dst_noc_unicast_info.push_back(std::make_pair(virtual_coord, /*num_mcast_dests=*/0));
                }
            }
        }
        return dst_noc_unicast_info;
    };

    // Unicast/Multicast Semaphores
    for (const Semaphore &semaphore : this->semaphores()) {
        std::vector<uint32_t> semaphore_data(1);
        semaphore_data[0] = semaphore.initial_value();

        // TODO: use semaphore.core_type from main
        if (semaphore.core_type() == CoreType::WORKER) {
            uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                device->extract_dst_noc_multicast_info(
                    semaphore.core_range_set().ranges(), CoreType::WORKER);
            transfer_info transfer_info = {
                .dst_base_addr = semaphore.offset(),
                .dst_noc_info = dst_noc_multicast_info,
                .linked = false,
                .data = semaphore_data};
            this->program_transfer_info.multicast_semaphores[semaphore.offset()].push_back(transfer_info);
        } else if (semaphore.core_type() == CoreType::ETH) {
            // TODO: we only fast dispatch to active eth...
            uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
            std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                extract_dst_noc_unicast_info(semaphore.core_range_set().ranges(), CoreType::ETH);
            transfer_info transfer_info = {
                .dst_base_addr = semaphore.offset(),
                .dst_noc_info = dst_noc_unicast_info,
                .linked = false,
                .data = semaphore_data};
            this->program_transfer_info.unicast_semaphores[semaphore.offset()].push_back(transfer_info);
        }
    }

    // Circular Buffer Configs handled in EnqueueProgram

    // Assume here and in command queue that kg_buffers is populated with multicast buffers first then unicast buffers
    // Program Binaries and Go Signals
    // TODO: cleanup put the WORKERS and ETH logic together..

    // All program binaries will be packed into a single buffer in memory
    std::vector<uint32_t> binaries_data;
    // Map is used to look up transfer info by kernel id when we populate data ordered by core groups
    std::unordered_map<KernelHandle, kernel_bins_transfer_info> kernel_transfer_info;
    // This is generic for workers and eth cores
    for (const auto &kernels : this->kernels_) {
        for (const auto &[kernel_id, kernel] : kernels) {
            std::vector<RISCV> sub_kernels;
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }
            const auto &binaries = kernel->binaries(device->build_key());
            std::vector<uint32_t> dst_base_addrs;
            std::vector<uint32_t> page_offsets;
            std::vector<uint32_t> lengths;
            std::vector<RISCV> riscvs;
            uint32_t transfer_info_index = 0;

            for (size_t sub_kernel_index = 0; sub_kernel_index < binaries.size(); ++sub_kernel_index) {
                const ll_api::memory& kernel_bin = *binaries[sub_kernel_index];

                // TODO: Pack erisc spans too, and then everthing is
                // one span
                uint32_t num_spans = kernel_bin.num_spans();
                dst_base_addrs.resize(dst_base_addrs.size() + num_spans);
                page_offsets.resize(page_offsets.size() + num_spans);
                lengths.resize(lengths.size() + num_spans);
                riscvs.resize(riscvs.size() + num_spans);

                kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {

                    // Set dst for eth kernels until they move to ring buffer
                    dst_base_addrs[transfer_info_index] = dst;
                    page_offsets[transfer_info_index] =
                        binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    lengths[transfer_info_index] = len * sizeof(uint32_t);
                    riscvs[transfer_info_index] = sub_kernels[sub_kernel_index];

                    binaries_data.insert(binaries_data.end(), mem_ptr, mem_ptr + len);
                    binaries_data.resize(
                        align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)), 0);
                    transfer_info_index++;
                });
            }

            kernel_bins_transfer_info kb_transfer_info = {
                .dst_base_addrs = dst_base_addrs, .page_offsets = page_offsets, .lengths = lengths, .riscvs = riscvs};
            kernel_transfer_info.insert({kernel_id, kb_transfer_info});
        }
    }

    if (binaries_data.size() > 0) {
        this->program_transfer_info.binary_data = binaries_data;
    }

    std::uint32_t num_active_cores = 0;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        for (const auto& kernel_group : this->get_kernel_groups(index)) {
            // TODO: add a bit in the hal that says if this core type is unicast/multicast
            if (core_type == CoreType::WORKER) {
                std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                    device->extract_dst_noc_multicast_info(
                       kernel_group->core_ranges.ranges(), core_type);
                std::vector<KernelHandle> kernel_ids;
                for (int dispatch_class = 0; dispatch_class < kernel_group->kernel_ids.size(); dispatch_class++) {
                    auto &optional_id = kernel_group->kernel_ids[dispatch_class];
                    if (optional_id) {
                        KernelHandle device_local_kernel_id = program_utils::get_device_local_kernel_handle(optional_id.value());
                        kernel_ids.push_back(device_local_kernel_id);
                        int proc_sub_class = 0;
                        for (uint32_t& dst_addr : kernel_transfer_info.at(device_local_kernel_id).dst_base_addrs) {
                            // TODO: ditch this w/ linear writes based on program config kernel_text_offset and size
                            dst_addr = kernel_group->kernel_text_offsets[dispatch_class + proc_sub_class];
                            proc_sub_class++;
                        }
                    }
                }

                for (const auto &[cores, num_mcast_dsts] : dst_noc_multicast_info) {
                    for (const auto &kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            cores, num_mcast_dsts, kernel_transfer_info.at(kernel_id));
                    }
                }
            } else {
                TT_ASSERT(core_type == CoreType::ETH);
                std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                    extract_dst_noc_unicast_info(kernel_group->core_ranges.ranges(), core_type);

                std::vector<KernelHandle> kernel_ids;
                if (kernel_group->kernel_ids[DISPATCH_CLASS_ETH_DM0]) {
                    KernelHandle device_local_kernel_id = program_utils::get_device_local_kernel_handle(kernel_group->kernel_ids[DISPATCH_CLASS_ETH_DM0].value());
                    kernel_ids.push_back(device_local_kernel_id);
                }

                for (const auto &[cores, num_mcast_dsts] : dst_noc_unicast_info) {
                    for (const auto &kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            cores, num_mcast_dsts, kernel_transfer_info.at(kernel_id));
                    }
                }
            }
        }
        num_active_cores += this->logical_cores()[index].size();
    }

    this->program_transfer_info.num_active_cores = num_active_cores;

    return;
}

uint32_t detail::Program_::finalize_rt_args(uint32_t programmable_core_type_index, uint32_t base_offset) {
    // Iterate over kernels in the program and "level" the number of RTAs based on the max
    // Unique RTAs are packed across dispatch classes
    // Common RTAs come after unique RTAs
    return program_utils::finalize_rt_args(
        this->kernels_[programmable_core_type_index],
        this->get_kernel_groups(programmable_core_type_index),
        base_offset,
        programmable_core_type_index,
        this->get_program_config(programmable_core_type_index).rta_offset,
        this->get_program_config(programmable_core_type_index).crta_offsets,
        this->get_program_config(programmable_core_type_index).crta_sizes
    );
}

ProgramConfig& detail::Program_::get_program_config(uint32_t programmable_core_type_index) {
    return this->program_configs_[programmable_core_type_index];
}

ProgramConfig& Program::get_program_config(uint32_t programmable_core_type_index) {
    return pimpl_->get_program_config(programmable_core_type_index);
}

uint32_t detail::Program_::finalize_sems(uint32_t programmable_core_type_index, uint32_t base_offset) {
    return program_utils::finalize_sems(programmable_core_type_index, base_offset, this->semaphores_, this->program_configs_[programmable_core_type_index].sem_offset, this->program_configs_[programmable_core_type_index].sem_size);
}

void detail::Program_::set_launch_msg_sem_offsets() {

    for (uint32_t kg_type_index = 0; kg_type_index < hal.get_programmable_core_type_count(); kg_type_index++) {
        for (auto& kg : this->get_kernel_groups(kg_type_index)) {
            for (uint32_t sem_type_index = 0; sem_type_index < hal.get_programmable_core_type_count(); sem_type_index++) {
                kg->launch_msg.kernel_config.sem_offset[sem_type_index] =
                    this->program_configs_[sem_type_index].sem_offset;
            }
        }
    }
}

uint32_t detail::Program_::finalize_cbs(uint32_t programmable_core_type_index, uint32_t base_offset) {
     return program_utils::finalize_cbs(programmable_core_type_index,  this->get_kernel_groups(programmable_core_type_index), base_offset, this->program_configs_[programmable_core_type_index].cb_offset, this->program_configs_[programmable_core_type_index].cb_size, this->program_configs_[programmable_core_type_index].local_cb_size);
}

uint32_t detail::Program_::finalize_kernel_bins(Device *device, uint32_t programmable_core_type_index, uint32_t base_offset) {
    return program_utils::finalize_kernel_bins(device, programmable_core_type_index, this->kernels_[programmable_core_type_index], this->get_kernel_groups(programmable_core_type_index), base_offset, this->program_configs_[programmable_core_type_index].kernel_text_offset, this->program_configs_[programmable_core_type_index].kernel_text_size);
}

uint32_t& detail::Program_::get_program_config_size(uint32_t programmable_core_type_index) {
    return this->program_config_sizes_[programmable_core_type_index];
}

const std::vector<SubDeviceId> &detail::Program_::determine_sub_device_ids(const Device *device) {
    // We need to calculate the sub_device_id when we haven't compiled the program yet, or this is the first time we
    // are getting the sub_device_ids after compilation
    auto sub_device_manager_id = device->get_active_sub_device_manager_id();
    auto sub_device_ids = this->sub_device_ids_.find(sub_device_manager_id);
    if (this->compiled_.empty() || sub_device_ids == this->sub_device_ids_.end()) {
        if (!this->compiled_.empty()) {
            TT_FATAL(this->sub_device_ids_.empty(), "Multiple sub device managers are not currently supported for a single program");
        }
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr || sub_device_manager_id == device->get_default_sub_device_manager_id()) {
            // No sub device manager, nothing to validate
            auto [sub_device_ids, _] = this->sub_device_ids_.insert_or_assign(sub_device_manager_id, std::vector<SubDeviceId>{SubDeviceId{0}});
            return sub_device_ids->second;
        } else {
            std::unordered_set<SubDeviceId> used_sub_device_ids;
            auto find_sub_device_ids = [&] (HalProgrammableCoreType core_type) {
                auto core_type_index = hal.get_programmable_core_type_index(core_type);
                if (core_type_index == -1) {
                    return;
                }
                const auto& program_kgs = this->get_kernel_groups(hal.get_programmable_core_type_index(core_type));
                uint32_t num_intersections = 0;
                uint32_t num_cores = 0;
                for (const auto& kg : program_kgs) {
                    for (size_t i = 0; i < device->num_sub_devices(); ++i) {
                        const auto& sub_device_cores = device->worker_cores(core_type, SubDeviceId{i});
                        auto intersection = sub_device_cores.intersection(kg->core_ranges);
                        if (intersection.size() > 0) {
                            used_sub_device_ids.insert(SubDeviceId{i});
                            num_intersections += intersection.num_cores();
                        }
                    }
                    num_cores += kg->core_ranges.num_cores();
                }
                TT_FATAL(num_intersections == num_cores,
                         "Kernel group cores do not match sub device cores for programmable core type {}",
                         magic_enum::enum_name(core_type));
            };
            find_sub_device_ids(HalProgrammableCoreType::TENSIX);
            find_sub_device_ids(HalProgrammableCoreType::ACTIVE_ETH);
            auto [sub_device_ids, _] = this->sub_device_ids_.insert_or_assign(sub_device_manager_id, std::vector<SubDeviceId>(used_sub_device_ids.begin(), used_sub_device_ids.end()));
            return sub_device_ids->second;
        }
    }
    return sub_device_ids->second;
}

void detail::Program_::allocate_kernel_bin_buf_on_device(Device *device) {
    // Allocate the DRAM kernel binary buffer for this program on the specified device, if not previously allocated.
    // We allocate program binaries top down to minimize fragmentation with other buffers in DRAM, which are typically allocated bottom up
    std::size_t binary_data_size_bytes = this->program_transfer_info.binary_data.size() * sizeof(uint32_t);
    if (this->kernels_buffer_.find(device->id()) == this->kernels_buffer_.end() and binary_data_size_bytes) {
        std::shared_ptr<Buffer> kernel_bin_buf = Buffer::create(device, binary_data_size_bytes, HostMemDeviceCommand::PROGRAM_PAGE_SIZE, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, std::nullopt, false);
        this->kernels_buffer_[device->id()] = kernel_bin_buf;
    }
}

void detail::Program_::finalize(Device *device) {
    // Store the number of tensix "go signals" for use by CQ
    // CQ iterates over these to update runtime addresses, needs to know when eth begins (after tensix)
    // TODO: should store all the counts

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        HalProgrammableCoreType programmable_core_type = static_cast<HalProgrammableCoreType>(index);
        uint32_t offset = 0;

        offset = finalize_rt_args(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));

        offset = finalize_sems(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));

        offset = finalize_cbs(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));

        offset = finalize_kernel_bins(device, index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));

        this->get_program_config_size(index) = offset;

        auto max_size = hal.get_dev_size(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);
        TT_FATAL(offset < max_size,
                 "Program size ({}) too large for kernel config buffer ({}) on {}",
                 offset, max_size, magic_enum::enum_name(programmable_core_type));
    }

    this->get_program_config_size(hal.get_programmable_core_type_count()) = runs_on_noc_multicast_only_cores();
    this->get_program_config_size(hal.get_programmable_core_type_count() + 1) = runs_on_noc_unicast_only_cores();

    // The sem offsets cross programmable_core_types so must be set after the loop above
    this->set_launch_msg_sem_offsets();

    // TODO: This check is wrong - it populates dispatch data for dispatch kernels
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        this->populate_dispatch_data(device);  // TODO: maybe rename
    }

    finalized_ = true;
}

void Program::set_launch_msg_sem_offsets() { pimpl_->set_launch_msg_sem_offsets(); }
void Program::populate_dispatch_data(Device* device) { pimpl_->populate_dispatch_data(device); }

void Program::generate_dispatch_commands(Device* device) {
    bool is_cached = this->is_cached();
    uint64_t command_hash = device->build_key();
    if (not hal.is_coordinate_virtualization_enabled()) {
        // When coordinate virtualization is not enabled, explicitly encode the device
        // id into the command hash, to always assert on programs being reused across devices.
        command_hash = (command_hash << 32) | (device->id());
    }
    auto& cached_program_command_sequences = this->get_cached_program_command_sequences();
    if (!is_cached) {
        auto sub_device_id = this->determine_sub_device_ids(device)[0];
        ProgramCommandSequence program_command_sequence;
        program_utils::insert_empty_program_dispatch_preamble_cmd(program_command_sequence);
        program_utils::insert_stall_cmds(program_command_sequence, sub_device_id, device);
        program_utils::assemble_runtime_args_commands(program_command_sequence, *this, device);
        program_utils::assemble_device_commands(program_command_sequence, *this, device, sub_device_id);
        cached_program_command_sequences.insert({command_hash, std::move(program_command_sequence)});
        this->set_cached();
    } else {
        auto cached_cmd_iter = cached_program_command_sequences.find(command_hash);
        TT_FATAL(cached_cmd_iter != cached_program_command_sequences.end(), "Enqueueing a Program across devices with different cores harvested is not supported, unless coordinate virtualization is enabled (only enabled on Wormhole and above).");
    }
}

void Program::allocate_kernel_bin_buf_on_device(Device* device) { pimpl_->allocate_kernel_bin_buf_on_device(device); }

void Program::finalize(Device *device) { pimpl_->finalize(device); }

void detail::Program_::compile(Device *device, bool fd_bootloader_mode) {
    //ZoneScoped;
    if (compiled_.contains(device->build_key())) {
        return;
    }
    // Clear the determined sub_device_ids when we compile the program for the first time
    // This way, determine_sub_device_ids is forced to recalculate with the finalized information on the used cores
    if (compiled_.empty()) {
        this->sub_device_ids_.erase(device->get_active_sub_device_manager_id());
    }

    TT_FATAL(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        this->get_id());

    bool profile_kernel = getDeviceProfilerState();
    std::vector<std::shared_future<void>> events;
    DprintServerSetProfilerState(profile_kernel);

    auto validate_kernel_placement = [&device, &fd_bootloader_mode](std::shared_ptr<Kernel> kernel) {
        // Placement rules:
        //  Slow dispatch:
        //      - kernels cannot be on storage only cores
        //  Fast dispatch (tensix):
        //      - kernels cannot be on storage only cores an
        //      - tensix kernels cannot be on dispatch cores
        //  Fast dispatch (ethernet):
        //      - kernels cannot be on storage only cores
        //      - eth kernels cannot be on idle eth cores
        bool slow_dispatch = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

        const auto &dispatch_core_config = dispatch_core_manager::instance().get_dispatch_core_config(device->id());
        CoreType dispatch_core_type = dispatch_core_config.get_core_type();
        const std::vector<CoreCoord> &storage_cores = tt::get_logical_storage_cores(device->id(), device->num_hw_cqs(), dispatch_core_config);
        bool on_storage_only_core =  std::any_of(storage_cores.begin(), storage_cores.end(), [&kernel](const CoreCoord& storage_core) {
            return kernel->is_on_logical_core(storage_core);
        });
        TT_FATAL(not on_storage_only_core, "Illegal kernel placement for {}. Kernels cannot be placed on storage only cores!", kernel->name());

        // Kernels used to implement fast dispatch can be placed on dispatch cores
        if (not slow_dispatch and not fd_bootloader_mode) {
            const std::vector<CoreCoord> &dispatch_cores = tt::get_logical_dispatch_cores(device->id(), device->num_hw_cqs(), dispatch_core_config);

            bool on_dispatch_core = std::any_of(dispatch_cores.begin(), dispatch_cores.end(), [&kernel, &dispatch_core_type](const CoreCoord &dispatch_core) {
                if (kernel->get_kernel_core_type() != dispatch_core_type) {
                    return false;
                }
                return kernel->is_on_logical_core(dispatch_core);
            });

            TT_FATAL(not on_dispatch_core, "Illegal kernel placement for {}, Kernels cannot be placed on dispatch cores!", kernel->name());
        }
    };
    for (auto & kernels : kernels_) {
        for (auto &[id, kernel] : kernels) {
            validate_kernel_placement(kernel);
            launch_build_step(
                [kernel, device, this] {
                    JitBuildOptions build_options(device->build_env());
                    kernel->set_build_options(build_options);
                    if (this->compiled_.empty()) {
                        this->set_remote_circular_buffer_init(kernel);
                    }
                    this->set_cb_data_fmt(kernel->logical_coreranges(), build_options);
                    this->set_cb_tile_dims(kernel->logical_coreranges(), build_options);

                    auto kernel_hash = KernelCompileHash(kernel, build_options, device->build_key(), device->get_device_kernel_defines_hash());
                    std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
                    kernel->set_full_name(kernel_path_suffix);
                    build_options.set_name(kernel_path_suffix);
                    bool cache_hit = true;
                    bool path_exists = std::filesystem::exists(build_options.path);
                    if (enable_persistent_kernel_cache && path_exists) {
                        if (not detail::HashLookup::inst().exists(kernel_hash)) {
                            detail::HashLookup::inst().add(kernel_hash);
                            detail::HashLookup::inst().add_generated_bin(kernel_hash);
                        }
                    } else if (detail::HashLookup::inst().add(kernel_hash)) {
                        GenerateBinaries(device, build_options, kernel);
                        cache_hit = false;
                        detail::HashLookup::inst().add_generated_bin(kernel_hash);
                    }
                    while (not detail::HashLookup::inst().is_bin_generated(kernel_hash)) {
                    }
                    if (detail::CompilationReporter::enabled()) {
                        detail::CompilationReporter::inst().add_kernel_compile_stats(
                            get_id(), kernel, cache_hit, kernel_hash);
                    }
                    kernel->set_binary_path(build_options.path);
                },
                events);
        }
    }
    sync_build_step(events);

    for (auto &kernels : kernels_) {
        for (auto &[id, kernel] : kernels) {
            launch_build_step([kernel, device] { kernel->read_binaries(device); }, events);
        }
    }

    sync_build_step(events);

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(get_id(), num_kernels(), [this](size_t kernel_id) {
            return get_kernel(kernel_id);
        }, enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(get_id(), device);
    }
    compiled_.insert(device->build_key());
}

void Program::compile(Device *device, bool fd_bootloader_mode) { pimpl_->compile(device, fd_bootloader_mode); }

void detail::Program_::set_runtime_id(uint64_t id) { this->runtime_id = id; }

void Program::set_runtime_id(uint64_t id) { pimpl_->set_runtime_id(id); }

uint32_t detail::Program_::get_sem_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) {

    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);
    const auto &sub_device_ids = this->determine_sub_device_ids(device);
    // TODO: This restriction can be lifted once we have support for programs spanning multiple sub-devices
    // Semaphores across sub-devices are expected to have the same address
    TT_FATAL(sub_device_ids.size() == 1, "get_sem_base_addr currently only supports programs spanning a single sub-device");
    auto sub_device_index = sub_device_ids[0].to_index();
    uint32_t base_addr = device->using_fast_dispatch()
                             ? this->last_used_command_queue_for_testing->get_config_buffer_mgr(sub_device_index).get_last_slot_addr(
                                   programmable_core_type)
                             : hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);

    return base_addr + this->program_configs_[index].sem_offset;
}

uint32_t Program::get_sem_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_sem_base_addr(device, logical_core, core_type);
}

uint32_t detail::Program_::get_cb_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) {

    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);
    const auto &sub_device_ids = this->determine_sub_device_ids(device);
    // TODO: This restriction can be lifted once this function is changed to return a vector of addresses
    // Addresses are not the same across sub-devices
    TT_FATAL(sub_device_ids.size() == 1, "get_sem_base_addr currently only supports programs spanning a single sub-device");
    auto sub_device_index = sub_device_ids[0].to_index();
    uint32_t base_addr = device->using_fast_dispatch()
                             ? this->last_used_command_queue_for_testing->get_config_buffer_mgr(sub_device_index).get_last_slot_addr(
                                   programmable_core_type)
                             : hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);

    return base_addr + this->program_configs_[index].cb_offset;
}

uint32_t Program::get_cb_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) {
    return pimpl_->get_cb_base_addr(device, logical_core, core_type);
}

void detail::Program_::set_last_used_command_queue_for_testing(HWCommandQueue *queue) {
    this->last_used_command_queue_for_testing = queue;
}

void Program::set_last_used_command_queue_for_testing(HWCommandQueue *queue) {
    pimpl_->set_last_used_command_queue_for_testing(queue);
}

uint32_t detail::Program_::get_sem_size(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].sem_size;
}

uint32_t Program::get_sem_size(Device *device, CoreCoord logical_core, CoreType core_type) const {
    return pimpl_->get_sem_size(device, logical_core, core_type);
}

uint32_t detail::Program_::get_cb_size(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].cb_size;
}

uint32_t Program::get_cb_size(Device *device, CoreCoord logical_core, CoreType core_type) const {
    return pimpl_->get_cb_size(device, logical_core, core_type);
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::Program_::runs_on_noc_unicast_only_cores() {
    return (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
            not this->get_kernel_groups(hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)).empty());
}

bool Program::runs_on_noc_unicast_only_cores() { return pimpl_->runs_on_noc_unicast_only_cores(); }

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::Program_::runs_on_noc_multicast_only_cores() {
    return (hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX) != -1 and
            not this->get_kernel_groups(hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)).empty());
}

bool Program::runs_on_noc_multicast_only_cores() { return pimpl_->runs_on_noc_multicast_only_cores(); }

bool detail::Program_::kernel_binary_always_stored_in_ringbuffer() {
    // Active ethernet cores use a fixed address for the kernel binary, because they don't have enough memory to have
    // that big of a ringbuffer.
    return !(
        hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
        not this->get_kernel_groups(hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)).empty());
}

bool Program::kernel_binary_always_stored_in_ringbuffer() {
    return pimpl_->kernel_binary_always_stored_in_ringbuffer();
}

Program::Program(Program &&other) noexcept = default;

Program& Program::operator=(Program &&other) noexcept = default;

Program::~Program() noexcept = default;

uint64_t detail::Program_::get_id() const { return this->id; }

uint64_t Program::get_id() const { return pimpl_->get_id(); }

uint64_t detail::Program_::get_runtime_id() const { return this->runtime_id; }

uint64_t Program::get_runtime_id() const { return pimpl_->get_runtime_id(); }

size_t detail::Program_::num_kernels() const {
    size_t count = 0;
    for (const auto& kernels : kernels_) {
        count += kernels.size();
    }
    return count;
}

size_t Program::num_kernels() const { return pimpl_->num_kernels(); }

const std::vector<std::shared_ptr<CircularBuffer>> &detail::Program_::circular_buffers() const { return circular_buffers_; }

const std::vector<std::shared_ptr<CircularBuffer>> &Program::circular_buffers() const { return pimpl_->circular_buffers(); }

const std::vector< Semaphore > & detail::Program_::semaphores() const { return semaphores_; }

const std::vector< Semaphore > & Program::semaphores() const { return pimpl_->semaphores(); }

void detail::Program_::add_buffer(std::shared_ptr<Buffer> buf) { owned_buffer_pool.push_back(std::move(buf)); }

void Program::add_buffer(std::shared_ptr<Buffer> buf) { pimpl_->add_buffer(std::move(buf)); }

void detail::Program_::release_buffers() { owned_buffer_pool = {}; }

void Program::release_buffers() { pimpl_->release_buffers(); }

std::vector<std::reference_wrapper<const Semaphore>> detail::Program_::semaphores_on_core(const CoreCoord &core, CoreType core_type) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for (const Semaphore &s : this->semaphores_) {
        if (s.initialized_on_logical_core(core) && s.core_type() == core_type) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

std::vector<std::reference_wrapper<const Semaphore>> Program::semaphores_on_core(const CoreCoord &core, CoreType core_type) const {
    return pimpl_->semaphores_on_core(core, core_type);
}

bool detail::Program_::is_finalized() const { return this->finalized_; }

bool Program::is_finalized() const { return pimpl_->is_finalized(); }
bool Program::is_cached() const { return pimpl_->is_cached(); }
void Program::set_cached() { pimpl_->set_cached(); }

ProgramBinaryStatus Program::get_program_binary_status(std::size_t device_id) const { return pimpl_->get_program_binary_status(device_id); }
void Program::set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status) { pimpl_->set_program_binary_status(device_id, status); }

const std::vector<SubDeviceId> &Program::determine_sub_device_ids(const Device *device) { return pimpl_->determine_sub_device_ids(device); }

const ProgramTransferInfo &Program::get_program_transfer_info() const noexcept { return pimpl_->program_transfer_info; }

std::shared_ptr<Buffer> Program::get_kernels_buffer(Device* device) const noexcept {
    if (auto it = pimpl_->kernels_buffer_.find(device->id()); it != pimpl_->kernels_buffer_.end()) {
        return it->second;
    }
    return nullptr;
}

void Program::set_kernels_bin_buffer(const std::shared_ptr<Buffer>& buffer) {
    pimpl_->kernels_buffer_.insert({buffer->device()->id(), buffer});
}

std::vector<uint32_t> &Program::get_program_config_sizes() const noexcept { return pimpl_->program_config_sizes_; }

std::unordered_map<uint64_t, ProgramCommandSequence> &Program::get_cached_program_command_sequences() noexcept {
    return pimpl_->cached_program_command_sequences_;
}

v1::ProgramHandle v1::CreateProgram() { return {}; }

v1::KernelHandle v1::CreateKernel(
    v1::ProgramHandle &program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const DataMovementConfig &config) {
    return v1::KernelHandle{v0::CreateKernel(program, std::string{file_name}, core_spec, config)};
}

v1::KernelHandle v1::CreateKernel(
    v1::ProgramHandle &program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const ComputeConfig &config) {
    return v1::KernelHandle{v0::CreateKernel(program, std::string{file_name}, core_spec, config)};
}

v1::KernelHandle v1::CreateKernel(
    v1::ProgramHandle &program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const EthernetConfig &config) {
    return v1::KernelHandle{v0::CreateKernel(program, std::string{file_name}, core_spec, config)};
}

uint32_t v1::CreateSemaphore(
    v1::ProgramHandle &program, const CoreRangeSet &core_spec, uint32_t initial_value, CoreType core_type) {
    return v0::CreateSemaphore(program, core_spec, initial_value, core_type);
}

v1::CircularBufferHandle v1::CreateCircularBuffer(
    v1::ProgramHandle &program, const CoreRangeSet &core_spec, const CircularBufferConfig &config) {
    return v1::CircularBufferHandle{v0::CreateCircularBuffer(program, core_spec, config)};
}

const CircularBufferConfig &v1::GetCircularBufferConfig(
    v1::ProgramHandle &program, v1::CircularBufferHandle cb_handle) {
    return v0::GetCircularBufferConfig(program, static_cast<v0::CBHandle>(cb_handle));
}

constexpr auto to_handle() {
    return [](const detail::Internal_::map_type::value_type &pair) {
        return v1::CircularBufferHandle{pair.first};
    };
}

v1::SizedCircularBufferRange v1::GetCircularBuffers(v1::ProgramHandle &program) {
    return detail::Internal_::get_circular_buffers_by_id(program) |
           ranges::views::transform(to_handle());
}

inline auto is_on_logical_corerange(CoreRange cr) {
    return [=](const detail::Internal_::map_type::value_type &pair) {
        return pair.second->is_on_logical_corerange(cr);
    };
}

v1::CircularBufferRange v1::GetCircularBuffersOnCoreRange(v1::ProgramHandle &program, CoreRange cr) {
    return detail::Internal_::get_circular_buffers_by_id(program) |
           ranges::views::filter(is_on_logical_corerange(cr)) |
           ranges::views::transform(to_handle());
}

void v1::UpdateCircularBufferTotalSize(
    v1::ProgramHandle &program, v1::CircularBufferHandle cb_handle, std::uint32_t total_size) {
    v0::UpdateCircularBufferTotalSize(program, static_cast<v0::CBHandle>(cb_handle), total_size);
}

void v1::UpdateDynamicCircularBufferAddress(
    v1::ProgramHandle &program, v1::CircularBufferHandle cb_handle, const v1::BufferHandle& buffer) {
    v0::UpdateDynamicCircularBufferAddress(program, static_cast<v0::CBHandle>(cb_handle), *buffer);
}

}  // namespace tt::tt_metal
