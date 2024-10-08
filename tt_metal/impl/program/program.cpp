// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/program.hpp"

#include "common/executor.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/graph/graph_tracking.hpp"

namespace tt::tt_metal {

namespace {
std::atomic<bool> enable_persistent_kernel_cache = false;

void GenerateBinaries(Device *device, JitBuildOptions &build_options, std::shared_ptr<Kernel> kernel) {
    ZoneScoped;
    const std::string tracyPrefix = "GenerateBinaries_";
    ZoneName((tracyPrefix + build_options.name).c_str(), build_options.name.length() + tracyPrefix.length());
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

size_t KernelCompileHash(const std::shared_ptr<Kernel> kernel, JitBuildOptions &build_options, uint32_t build_key) {
    // Account for device id in hash because generated headers are dependent on harvesting config, which can differ per
    // device This can be removed with https://github.com/tenstorrent/tt-metal/issues/3381

    // Also account for watcher/dprint enabled in hash because they enable additional code to
    // be compiled into the kernel.
    string compile_hash_str = fmt::format(
        "{}_{}_{}_{}",
        build_key,
        std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc)),
        kernel->compute_hash(),
        tt::llrt::OptionsG.get_watcher_enabled());

    for (int i = 0; i < llrt::RunTimeDebugFeatureCount; i++) {
        compile_hash_str += "_";
        compile_hash_str += tt::llrt::OptionsG.get_feature_hash_string((llrt::RunTimeDebugFeatures)i);
    }
    size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

#ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        unique_lock<mutex> lock;
        f << kernel->name() << " :: " << build_key << "::" << std::hash<tt_hlk_desc>{}(build_options.hlk_desc)
          << " :: " << kernel->compute_hash() << " :: " << compile_hash_str << " " << compile_hash << std::endl
          << std::flush;
    }
#endif
    return compile_hash;
}
}  // namespace
namespace detail {

KernelHandle AddKernel (Program &program, std::shared_ptr<Kernel> kernel, const HalProgrammableCoreType core_type) {
    return program.add_kernel(kernel, core_type);
}

std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id) {
    return program.get_kernel(kernel_id);
}

std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id) {
    return program.get_circular_buffer(id);
}

// Checks that circular buffers do not grow into L1 buffer space
void ValidateCircularBufferRegion(const Program &program, const Device *device) {
    program.validate_circular_buffer_region(device);
}

void AddConfigBuffer(Program &program, std::shared_ptr<Buffer> config_buffer) {
    program.add_config_buffer(config_buffer);
}

void EnablePersistentKernelCache() { enable_persistent_kernel_cache = true; }

void DisablePersistentKernelCache() { enable_persistent_kernel_cache = false; }
}  // namespace detail

std::atomic<uint64_t> Program::program_counter = 0;

Program::Program() :
    id(program_counter++), runtime_id(0), worker_crs_({}), local_circular_buffer_allocation_needed_(false), finalized_(false) {

    uint32_t programmable_core_count = hal.get_programmable_core_type_count();
    for (uint32_t i = 0; i < programmable_core_count; i++) {
        kernels_.push_back({});
        grid_extent_.push_back({});
        kernel_groups_.push_back({});
        core_to_kernel_group_index_table_.push_back({});
    }

    program_configs_.resize(programmable_core_count);
    program_config_sizes_.resize(programmable_core_count);
}

KernelHandle Program::add_kernel(std::shared_ptr<Kernel> kernel, const HalProgrammableCoreType &programmable_core_type) {
    this->invalidate_compile();
    // Id is unique across all kernels on all core types
    KernelHandle id = this->num_kernels();
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);
    kernels_[index].insert({id, kernel});
    kernel_groups_[index].resize(0);
    core_to_kernel_group_index_table_[index].clear();
    return id;
}

std::shared_ptr<Kernel> Program::get_kernel(KernelHandle kernel_id) const {
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

KernelGroup::KernelGroup() : core_ranges({}) {}

KernelGroup::KernelGroup(
    const Program &program,
    uint32_t programmable_core_type_index,
    kernel_id_array_t kernel_ids,
    bool erisc_is_idle,
    int last_cb_index,
    const CoreRangeSet &new_ranges) :
    core_ranges({}) {

    this->programmable_core_type_index = programmable_core_type_index;
    this->core_ranges = this->core_ranges.merge(new_ranges);
    this->kernel_ids = kernel_ids;

    std::memset(&this->launch_msg, 0, sizeof(launch_msg_t));

    // Slow dispatch uses fixed addresses for the kernel config, configured here statically
    // Fast dispatch kernel config mangement happens under the CQ and will re-program the base
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        this->launch_msg.kernel_config.kernel_config_base[index] =
            hal.get_dev_addr(index, HalMemAddrType::KERNEL_CONFIG);
    }

    for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
        auto& optional_id = kernel_ids[class_id];
        if (optional_id) {
            const auto kernel = program.get_kernel(optional_id.value());
            this->launch_msg.kernel_config.watcher_kernel_ids[class_id] = kernel->get_watcher_kernel_id();
            this->launch_msg.kernel_config.enables |= 1 << class_id;

            if (programmable_core_type_index == hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)) {
                // The code below sets the brisc_noc_id for use by the device firmware
                // Use 0 if neither brisc nor ncrisc specify a noc
                if (class_id == DISPATCH_CLASS_TENSIX_DM0) {
                    // Use brisc's noc if brisc specifies a noc
                    this->launch_msg.kernel_config.brisc_noc_id = std::get<DataMovementConfig>(kernel->config()).noc;
                } else if (class_id == DISPATCH_CLASS_TENSIX_DM1) {
                    // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
                    // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
                    this->launch_msg.kernel_config.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
                    this->launch_msg.kernel_config.ncrisc_kernel_size16 = kernel->get_binary_size16();
                }
            }
        }
    }

    this->launch_msg.kernel_config.exit_erisc_kernel = false;
    this->launch_msg.kernel_config.max_cb_index = last_cb_index + 1;
    this->go_msg.signal = RUN_MSG_GO;
}

CoreType KernelGroup::get_core_type() const {
    return hal.get_core_type(this->programmable_core_type_index);
};

std::vector<KernelGroup> &Program::get_kernel_groups(uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    return kernel_groups_[programmable_core_type_index];
}

KernelGroup *Program::kernels_on_core(const CoreCoord &core, uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    if (core.x >= grid_extent_[programmable_core_type_index].x || core.y >= grid_extent_[programmable_core_type_index].y)
        return nullptr;
    uint8_t index = core_to_kernel_group_index_table_[programmable_core_type_index].at(core.y * grid_extent_[programmable_core_type_index].x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : &kernel_groups_[programmable_core_type_index].at(index);
}

struct KernelGroupInt {
    bool valid;
    kernel_id_array_t kernel_ids;

    bool operator==(const KernelGroupInt &b) const;
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

void Program::update_kernel_groups(uint32_t programmable_core_type_index) {
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
                grid[core_index].update(kernel->dispatch_class(), id);
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
            int last_cb_index = -1;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : kg_to_cores.second) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                        core_to_kernel_group_index_table_[programmable_core_type_index][y * grid_extent_[programmable_core_type_index].x + x] = index;

                        if (not hal.get_supports_cbs(programmable_core_type_index)) continue;
                        auto val = per_core_cb_indices_.find(CoreCoord({x, y}));
                        if (val != per_core_cb_indices_.end()) {
                            int i;
                            for (i = NUM_CIRCULAR_BUFFERS - 1; i >= 0; i--) {
                                if (val->second[i]) {
                                    break;
                                }
                            }
                            last_cb_index = (i > last_cb_index) ? i : last_cb_index;
                        }
                    }
                }
            }

            kernel_groups_[programmable_core_type_index].push_back(KernelGroup(
                *this,
                programmable_core_type_index,
                kg_to_cores.first.kernel_ids,
                erisc_is_idle,
                last_cb_index,
                kg_to_cores.second));
            index++;
        }
    }
}

void Program::CircularBufferAllocator::mark_address(uint64_t address, uint64_t size, uint64_t base_address) {
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

CBHandle Program::add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config) {
    this->invalidate_compile();
    std::shared_ptr<CircularBuffer> circular_buffer = std::make_shared<CircularBuffer>(core_range_set, config);
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

                for (uint32_t buffer_index : circular_buffer->buffer_indices()) {
                    if (buffer_index > NUM_CIRCULAR_BUFFERS) {
                        TT_THROW(
                            "Invalid circular buffer index: {} should be between 0 and {}",
                            buffer_index,
                            NUM_CIRCULAR_BUFFERS);
                    }
                    if (cb_indices.to_ulong() & (1 << buffer_index)) {
                        TT_THROW(
                            "Invalid circular buffer index: Cannot add circular buffer at index {}, another circular "
                            "buffer already exists",
                            buffer_index);
                    }
                    cb_indices[buffer_index] = 1;
                }
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

std::shared_ptr<CircularBuffer> Program::get_circular_buffer(CBHandle cb_id) const {
    if (this->circular_buffer_by_id_.find(cb_id) == this->circular_buffer_by_id_.end()) {
        TT_THROW("No circular buffer with id {} exists in Program {}", cb_id, this->id);
    }
    return this->circular_buffer_by_id_.at(cb_id);
}

const std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_core(const CoreCoord &core) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

const std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_corerange(const CoreRange &cr) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_corerange(cr)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

const std::vector<CoreRange> Program::circular_buffers_unique_coreranges() const {
    std::vector<CoreRange> core_ranges;
    for (auto circular_buffer : circular_buffers_) {
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            if (std::find(core_ranges.begin(), core_ranges.end(), core_range) == core_ranges.end()) {
                core_ranges.push_back(core_range);
            }
        }
    }
    return core_ranges;
}

void Program::invalidate_circular_buffer_allocation() {
    if (this->local_circular_buffer_allocation_needed_) {
        return;
    }
    for (CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        cb_allocator.reset_available_addresses();
    }
    this->local_circular_buffer_allocation_needed_ = true;
}

void Program::allocate_circular_buffers(const Device *device) {
    ZoneScoped;
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_cb_address = device->get_base_allocator_addr(HalMemType::L1);
    for (std::shared_ptr<CircularBuffer> circular_buffer : this->circular_buffers_) {
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
        tt::tt_metal::GraphTracker::instance().track_allocate_cb(circular_buffer->core_ranges(), computed_addr, circular_buffer->size());
        circular_buffer->set_locally_allocated_address(computed_addr);
    }
    this->local_circular_buffer_allocation_needed_ = false;
}

void Program::validate_circular_buffer_region(const Device *device) const {
    ZoneScoped;

    // Banks are in lockstep so we only need to get lowest L1 address of one compute and storage core
    // Only compute with storage cores can have CBs and all compute with storage cores will have the same bank offset
    const std::vector<uint32_t> &bank_ids =
        device->bank_ids_from_logical_core(BufferType::L1, *device->compute_cores_.begin());
    std::optional<DeviceAddr> lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids[0]);
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

size_t Program::num_semaphores(const CoreCoord &core) const { return semaphores_on_core(core).size(); }

size_t Program::num_semaphores() const { return semaphores_.size(); }

void Program::init_semaphores(const Device &device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const {
    auto semaphores_on_core = this->semaphores_on_core(logical_core);

    uint64_t kernel_config_base = hal.get_dev_addr(programmable_core_type_index, HalMemAddrType::KERNEL_CONFIG);
    uint64_t addr = kernel_config_base + this->program_configs_[programmable_core_type_index].sem_offset;
    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(
            device.id(),
            device.physical_core_from_logical_core(logical_core, core_type),
            {semaphore.get().initial_value()},
            addr + semaphore.get().offset());
    }
}

void Program::add_semaphore(const CoreRangeSet &crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    this->invalidate_compile();
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, core_type));
}

void Program::add_config_buffer(std::shared_ptr<Buffer> config_buffer) { config_buffers_.emplace_back(config_buffer); }

std::vector<std::vector<CoreCoord>> Program::logical_cores() const {
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

void Program::construct_core_range_set_for_worker_cores() {
    bool found_kernels = false;
    uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    for (auto [id, kernel] : kernels_[index]) {
        this->worker_crs_ = this->worker_crs_.merge(kernel->core_range_set());
        found_kernels = true;
    }
    TT_ASSERT(!found_kernels || this->worker_crs_.ranges().size() >= 1, "Invalid core range set");
}

void Program::set_cb_data_fmt(Device *device, const std::vector<CoreRange> &crs, JitBuildOptions &build_options) const {
    ZoneScoped;
    for (auto logical_cr : crs) {
        auto cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(
                    static_cast<CB>(buffer_index), circular_buffer->data_format(buffer_index));
            }
        }
    }
}

void Program::set_cb_tile_dims(Device *device, const std::vector<CoreRange> &crs, JitBuildOptions &build_options) const {
    ZoneScoped;
    for (const auto &logical_cr : crs) {
        auto cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto &circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                auto tile = circular_buffer->tile(buffer_index);
                if (tile.has_value()) {
                    build_options.set_cb_tile_dims_all_cores(
                        static_cast<CB>(buffer_index),
                        tile->get_num_faces(),
                        tile->get_partial_face(),
                        tile->get_face_shape()[0],
                        tile->get_narrow_tile(),
                        tile->get_tile_shape()[0],
                        tile->get_tile_shape()[1]);
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CB>(buffer_index),
                        tile->get_tile_size(circular_buffer->data_format(buffer_index)));
                } else {
                    Tile t;
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CB>(buffer_index),
                        t.get_tile_size(circular_buffer->data_format(buffer_index)));
                }

            }
        }
    }
}

void Program::invalidate_compile() {
    for (auto &[device_id, compile_needed] : compile_needed_) {
        compile_needed = true;
    }
}

void Program::populate_dispatch_data(Device *device) {
    static const uint32_t processor_to_firmware_base[] = {
        MEM_BRISC_FIRMWARE_BASE,
        MEM_NCRISC_FIRMWARE_BASE,
        MEM_TRISC0_FIRMWARE_BASE,
        MEM_TRISC1_FIRMWARE_BASE,
        MEM_TRISC2_FIRMWARE_BASE,
        eth_l1_mem::address_map::FIRMWARE_BASE
    };
    static const uint32_t processor_to_firmware_size[] = {
        MEM_BRISC_FIRMWARE_SIZE,
        MEM_NCRISC_INIT_IRAM_L1_SIZE,
        MEM_TRISC0_FIRMWARE_SIZE,
        MEM_TRISC1_FIRMWARE_SIZE,
        MEM_TRISC2_FIRMWARE_SIZE,
        eth_l1_mem::address_map::FIRMWARE_SIZE
    };

    auto extract_dst_noc_unicast_info =
        [&device](const std::set<CoreRange> &ranges, const CoreType core_type) -> std::vector<pair<transfer_info_cores, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info;
        for (const CoreRange &core_range : ranges) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord physical_coord = device->physical_core_from_logical_core(CoreCoord({x, y}), core_type);
                    dst_noc_unicast_info.push_back(std::make_pair(physical_coord, /*num_mcast_dests=*/0));
                }
            }
        }
        return dst_noc_unicast_info;
    };

    // Unicast/Multicast Semaphores
    for (const Semaphore &semaphore : this->semaphores()) {
        vector<uint32_t> semaphore_data(1);
        semaphore_data[0] = semaphore.initial_value();

        // TODO: use semaphore.core_type from main
        if (semaphore.core_type() == CoreType::WORKER) {
            uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                device->extract_dst_noc_multicast_info<std::set<CoreRange>>(
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
            vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
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
                const ll_api::memory &kernel_bin = binaries[sub_kernel_index];

                // Spans are now packed into one
                // TODO: code below can be simplified w/ a single span
                uint32_t num_spans = kernel_bin.num_spans();
                dst_base_addrs.resize(dst_base_addrs.size() + num_spans);
                page_offsets.resize(page_offsets.size() + num_spans);
                lengths.resize(lengths.size() + num_spans);
                riscvs.resize(riscvs.size() + num_spans);

                TT_ASSERT(kernel_bin.num_spans() == 1);

                uint32_t max_kernel_bin_size = processor_to_firmware_size[sub_kernels[sub_kernel_index]];

                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {

                    max_kernel_bin_size -= dst - processor_to_firmware_base[sub_kernels[sub_kernel_index]];

                    uint64_t relo_addr =
                        tt::llrt::relocate_dev_addr(dst);

                    dst_base_addrs[transfer_info_index] = (uint32_t)relo_addr;
                    page_offsets[transfer_info_index] =
                        binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    lengths[transfer_info_index] = len * sizeof(uint32_t);
                    riscvs[transfer_info_index] = sub_kernels[sub_kernel_index];

                    binaries_data.insert(binaries_data.end(), mem_ptr, mem_ptr + len);
                    binaries_data.resize(
                        align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)), 0);
                    transfer_info_index++;
                });

                uint32_t bin_size = kernel_bin.size() * sizeof(uint32_t);
                // TODO: remove this check when the ring buffer is in place (checked there)
                TT_FATAL(bin_size <= max_kernel_bin_size,
                    "Kernel binary size, {}, overflowed kernel binary storage size, {}",
                     bin_size, max_kernel_bin_size);
            }

            kernel_bins_transfer_info kb_transfer_info = {
                .dst_base_addrs = dst_base_addrs, .page_offsets = page_offsets, .lengths = lengths, .riscvs = riscvs};
            kernel_transfer_info.insert({kernel_id, kb_transfer_info});
        }
    }

    if (binaries_data.size() > 0) {
        // We allocate program binaries top down to minimize fragmentation with other buffers in DRAM, which are typically allocated bottom up
        this->kernels_buffer = std::make_shared<Buffer>(
            device, binaries_data.size() * sizeof(uint32_t), HostMemDeviceCommand::PROGRAM_PAGE_SIZE, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, std::nullopt, false);

        this->program_transfer_info.binary_data = binaries_data;
    }

    std::uint32_t num_active_cores = 0;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        for (KernelGroup &kernel_group : this->get_kernel_groups(index)) {
            // TODO: add a bit in the hal that says if this core type is unicast/multicast
            if (core_type == CoreType::WORKER) {
                std::vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                    device->extract_dst_noc_multicast_info<std::set<CoreRange>>(
                        kernel_group.core_ranges.ranges(), core_type);

                vector<KernelHandle> kernel_ids;
                for (auto &optional_id : kernel_group.kernel_ids) {
                    if (optional_id) {
                        kernel_ids.push_back(optional_id.value());
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
                vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                    extract_dst_noc_unicast_info(kernel_group.core_ranges.ranges(), core_type);

                vector<KernelHandle> kernel_ids;
                if (kernel_group.kernel_ids[DISPATCH_CLASS_ETH_DM0]) {
                    kernel_ids.push_back(kernel_group.kernel_ids[DISPATCH_CLASS_ETH_DM0].value());
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

uint32_t Program::finalize_rt_args(uint32_t programmable_core_type_index, uint32_t base_offset) {

    // Iterate over kernels in the program and "level" the number of RTAs based on the max
    // Unique RTAs are packed across dispatch classes
    // Common RTAs come after unique RTAs
    vector<uint32_t> max_rtas(DISPATCH_CLASS_MAX);
    vector<uint32_t> max_crtas(DISPATCH_CLASS_MAX);
    uint32_t max_unique_rta_size = 0;
    uint32_t total_crta_size = 0;

    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);

    this->get_program_config(programmable_core_type_index).rta_offset = base_offset;

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    for (auto& kg : this->get_kernel_groups(programmable_core_type_index)) {
        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            max_rtas[dispatch_class] = 0;
            auto& optional_id = kg.kernel_ids[dispatch_class];
            if (optional_id) {
                auto kernel = detail::GetKernel(*this, optional_id.value());
                for (const CoreRange &core_range : kg.core_ranges.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            CoreCoord core_coord(x, y);
                            max_rtas[dispatch_class] =
                                std::max(max_rtas[dispatch_class], (uint32_t)kernel->runtime_args(core_coord).size());
                        }
                    }
                }
            }
        }

        uint32_t offset = 0;
        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            auto& optional_id = kg.kernel_ids[dispatch_class];
            kg.rta_sizes[dispatch_class] = max_rtas[dispatch_class] * sizeof(uint32_t);
            if (optional_id) {
                auto kernel = detail::GetKernel(*this, optional_id.value());
                kernel->set_runtime_args_count(kg.core_ranges, max_rtas[dispatch_class]);
                kg.launch_msg.kernel_config.mem_map[dispatch_class].rta_offset = base_offset + offset;
                offset += max_rtas[dispatch_class] * sizeof(uint32_t);
            } else {
                kg.launch_msg.kernel_config.mem_map[dispatch_class].rta_offset = 0;
            }
        }

        kg.total_rta_size = offset;
        offset = align(offset, l1_alignment);
        max_unique_rta_size = std::max(offset, max_unique_rta_size);
    }

    for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
        max_crtas[dispatch_class] = 0;
    }
    // Find the max # common RTAs across all kernels for each dispatch class
    for (size_t kernel_id = 0; kernel_id < this->num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(*this, kernel_id);
        // TODO: kernels should be stored by programmable core type
        if (core_type == kernel->get_kernel_core_type() &&
            (programmable_core_type == HalProgrammableCoreType::IDLE_ETH) == kernel->is_idle_eth()) {
            uint32_t dispatch_class = kernel->dispatch_class();
            max_crtas[dispatch_class] =
                std::max(max_crtas[dispatch_class], (uint32_t)kernel->common_runtime_args().size());
        }
    }

    // Calculate the address offset and size for common RTAs for each dispatch class
    uint32_t offset = 0;
    for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
        uint32_t size = max_crtas[dispatch_class] * sizeof(uint32_t);
        this->get_program_config(programmable_core_type_index).crta_offsets[dispatch_class] = base_offset + max_unique_rta_size + offset;
        this->get_program_config(programmable_core_type_index).crta_sizes[dispatch_class] = size;
        offset += size;
        offset = align(offset, l1_alignment);
    }
    total_crta_size = offset;

    // Set the runtime_args_data sizing info based on the shared max
    for (size_t kernel_id = 0; kernel_id < this->num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(*this, kernel_id);
        // TODO: as above, fix when kernels are stored by programmable core type
        if (core_type == kernel->get_kernel_core_type() &&
            (programmable_core_type == HalProgrammableCoreType::IDLE_ETH) == kernel->is_idle_eth()) {
            uint32_t dispatch_class = kernel->dispatch_class();
            kernel->set_common_runtime_args_count(max_crtas[dispatch_class]);
        }
    }

    // Set the kernel group common runtime arg offsets use in the launch message
    for (auto& kg : this->get_kernel_groups(programmable_core_type_index)) {
        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            kg.launch_msg.kernel_config.mem_map[dispatch_class].crta_offset = this->get_program_config(programmable_core_type_index).crta_offsets[dispatch_class];
        }
    }

    // TODO: this is asserted here as the leveling above can break the limits enforced by the API
    // Once we use a ring buffer, memory space will be dynamic and this assert won't matter
    TT_FATAL(offset <= L1_KERNEL_CONFIG_SIZE, "offset {} cannot exceed config size {}", offset, L1_KERNEL_CONFIG_SIZE);

    return max_unique_rta_size + total_crta_size;
}

ProgramConfig& Program::get_program_config(uint32_t programmable_core_type_index) {
    return this->program_configs_[programmable_core_type_index];
}

uint32_t Program::finalize_sems(uint32_t programmable_core_type_index, uint32_t base_offset) {

    int max_id = -1;
    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    for (const auto & sem : this->semaphores_) {
        if (sem.core_type() == core_type && (int)sem.id() > max_id) {
            max_id = sem.id();
        }
    }

    uint32_t sem_size = (max_id + 1) * hal.get_alignment(HalMemType::L1);

    this->program_configs_[programmable_core_type_index].sem_offset = base_offset;
    this->program_configs_[programmable_core_type_index].sem_size = sem_size;

    return base_offset + sem_size;
}

void Program::set_launch_msg_sem_offsets() {

    for (uint32_t kg_type_index = 0; kg_type_index < hal.get_programmable_core_type_count(); kg_type_index++) {
        for (auto& kg : this->get_kernel_groups(kg_type_index)) {
            for (uint32_t sem_type_index = 0; sem_type_index < hal.get_programmable_core_type_count(); sem_type_index++) {
                kg.launch_msg.kernel_config.sem_offset[sem_type_index] =
                    this->program_configs_[sem_type_index].sem_offset;
            }
        }
    }
}

uint32_t Program::finalize_cbs(uint32_t programmable_core_type_index, uint32_t base_offset) {

    int count = 0;

    // TODO: has to be better way to do this and don't read from volatile
    for (auto& kg : this->get_kernel_groups(programmable_core_type_index)) {
        // TODO "max_cb_index" is misnamed, it is really the count of indices
        int32_t id = kg.launch_msg.kernel_config.max_cb_index;
        if (id > count) {
            count = id;
        }

        kg.launch_msg.kernel_config.cb_offset = base_offset;
    }

    uint32_t cb_size = count * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    this->program_configs_[programmable_core_type_index].cb_offset = base_offset;
    this->program_configs_[programmable_core_type_index].cb_size = cb_size;

    return base_offset + cb_size;
}

uint32_t& Program::get_program_config_size(uint32_t programmable_core_type_index) {
    return this->program_config_sizes_[programmable_core_type_index];
}

void Program::finalize() {
    // Store the number of tensix "go signals" for use by CQ
    // CQ iterates over these to update runtime addresses, needs to know when eth begins (after tensix)
    // TODO: should store all the counts
    this->tensix_go_signal_count_ = 0;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        if (core_type == CoreType::WORKER) {
            for (auto& kg : this->get_kernel_groups(index)) {
                this->tensix_go_signal_count_ += kg.core_ranges.size();
            }
        }
    }

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        uint32_t offset = 0;
        offset = finalize_rt_args(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));
        offset = finalize_sems(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));
        offset = finalize_cbs(index, offset);
        TT_ASSERT(offset == align(offset, hal.get_alignment(HalMemType::L1)));
        this->get_program_config_size(index) = offset;
    }

    // The sem offsets cross programmable_core_types so must be set after the loop above
    this->set_launch_msg_sem_offsets();

    finalized_ = true;
}

void Program::compile(Device *device, bool fd_bootloader_mode) {
    ZoneScoped;
    bool first_compile_on_device = compile_needed_.find(device->id()) == compile_needed_.end();
    if (not first_compile_on_device and (not compile_needed_.at(device->id()))) {
        return;
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

        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        const std::vector<CoreCoord> &storage_cores = tt::get_logical_storage_cores(device->id(), device->num_hw_cqs(), dispatch_core_type);
        bool on_storage_only_core =  std::any_of(storage_cores.begin(), storage_cores.end(), [&kernel](const CoreCoord& storage_core) {
            return kernel->is_on_logical_core(storage_core);
        });
        TT_FATAL(not on_storage_only_core, "Illegal kernel placement for {}. Kernels cannot be placed on storage only cores!", kernel->name());

        // Kernels used to implement fast dispatch can be placed on dispatch cores
        if (not slow_dispatch and not fd_bootloader_mode) {
            const std::vector<CoreCoord> &dispatch_cores = tt::get_logical_dispatch_cores(device->id(), device->num_hw_cqs(), dispatch_core_type);

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
                    this->set_cb_data_fmt(device, kernel->logical_coreranges(), build_options);
                    this->set_cb_tile_dims(device, kernel->logical_coreranges(), build_options);

                    auto kernel_hash = KernelCompileHash(kernel, build_options, device->build_key());
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
                            *this, kernel, cache_hit, kernel_hash);
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

    this->construct_core_range_set_for_worker_cores();
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        this->populate_dispatch_data(device);  // TODO: maybe rename
    }

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(*this, enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(*this, device);
    }
    compile_needed_[device->id()] = false;
}

void Program::set_runtime_id(uint64_t id) { this->runtime_id = id; }

uint32_t Program::get_sem_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord phys_core = device->physical_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(phys_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    uint32_t base_addr = device->using_fast_dispatch ?
        device->sysmem_manager().get_config_buffer_mgr().get_last_slot_addr(programmable_core_type) :
        hal.get_dev_addr(programmable_core_type, HalMemAddrType::KERNEL_CONFIG);

    return base_addr + this->program_configs_[index].sem_offset;
}

uint32_t Program::get_cb_base_addr(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord phys_core = device->physical_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(phys_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    uint32_t base_addr = device->using_fast_dispatch ?
        device->sysmem_manager().get_config_buffer_mgr().get_last_slot_addr(programmable_core_type) :
        hal.get_dev_addr(programmable_core_type, HalMemAddrType::KERNEL_CONFIG);

    return base_addr + this->program_configs_[index].cb_offset;
}

uint32_t Program::get_sem_size(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord phys_core = device->physical_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(phys_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].sem_size;
}

uint32_t Program::get_cb_size(Device *device, CoreCoord logical_core, CoreType core_type) const {

    CoreCoord phys_core = device->physical_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(phys_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].cb_size;
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool Program::runs_on_noc_unicast_only_cores() {
    return (hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
            this->get_kernel_groups(hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)).size());
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool Program::runs_on_noc_multicast_only_cores() {
    return (hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX) != -1 and
            this->get_kernel_groups(hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)).size());
}

Program::~Program() {}
}  // namespace tt::tt_metal
