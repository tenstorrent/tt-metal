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
void EnablePersistentKernelCache() { enable_persistent_kernel_cache = true; }

void DisablePersistentKernelCache() { enable_persistent_kernel_cache = false; }
}  // namespace detail

std::atomic<uint64_t> Program::program_counter = 0;

Program::Program() :
    id(program_counter++), worker_crs_({}), local_circular_buffer_allocation_needed_(false), loaded_onto_device(false) {
    std::set<CoreType> supported_core_types = {CoreType::WORKER, CoreType::ETH};
    for (const auto &core_type : supported_core_types) {
        kernels_.insert({core_type, {}});
        grid_extent_.insert({core_type, {}});
        kernel_groups_.insert({core_type, {}});
        core_to_kernel_group_index_table_.insert({core_type, {}});
    }
}

KernelHandle Program::add_kernel(std::shared_ptr<Kernel> kernel, const CoreType &core_type) {
    this->invalidate_compile();
    // Id is unique across all kernels on all core types
    KernelHandle id = this->num_kernels();
    kernels_[core_type].insert({id, kernel});
    kernel_groups_[core_type].resize(0);
    core_to_kernel_group_index_table_[core_type].clear();
    return id;
}

std::shared_ptr<Kernel> Program::get_kernel(KernelHandle kernel_id) const {
    // TT_ASSERT(kernel_id < this->kernels_.size(), "Expected Kernel with ID {} to be in Program {}", kernel_id,
    // this->id);
    //  find coretype based on kernel_id
    for (const auto &[core_type, kernels] : this->kernels_) {
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
    CoreType core_type,
    std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX> kernel_ids,
    bool erisc_is_idle,
    int last_cb_index,
    const CoreRangeSet &new_ranges) :
    core_ranges({}) {

    this->core_type = core_type;
    this->core_ranges = this->core_ranges.merge(new_ranges);
    this->kernel_ids = kernel_ids;

    std::memset(&this->launch_msg, 0, sizeof(launch_msg_t));

    // The code below sets the brisc_noc_id for use by the device firmware
    // Use 0 if neither brisc nor ncrisc specify a noc
    if (core_type == CoreType::WORKER) {
        // Dynamic address map
        this->launch_msg.kernel_config_base = L1_KERNEL_CONFIG_BASE;
    } else {
        this->launch_msg.kernel_config_base =
            erisc_is_idle ? IDLE_ERISC_L1_KERNEL_CONFIG_BASE : eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    }

    for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
        auto& optional_id = kernel_ids[class_id];
        if (optional_id) {
            const auto kernel = program.get_kernel(optional_id.value());
            this->launch_msg.watcher_kernel_ids[class_id] = kernel->get_watcher_kernel_id();
            this->launch_msg.enables |= 1 << class_id;

            if (core_type == CoreType::WORKER) {
                if (class_id == DISPATCH_CLASS_TENSIX_DM0) {
                    // Use brisc's noc if brisc specifies a noc
                    this->launch_msg.brisc_noc_id = std::get<DataMovementConfig>(kernel->config()).noc;
                } else if (class_id == DISPATCH_CLASS_TENSIX_DM1) {
                    // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
                    // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
                    this->launch_msg.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
                    this->launch_msg.ncrisc_kernel_size16 = kernel->get_binary_size16();
                }
            }
        }
    }

    this->launch_msg.exit_erisc_kernel = false;
    this->launch_msg.max_cb_index = last_cb_index + 1;
    this->launch_msg.run = RUN_MSG_GO;
}

CoreType KernelGroup::get_core_type() const {
    return core_type;
};

std::vector<KernelGroup> &Program::get_kernel_groups(const CoreType &core_type) {
    update_kernel_groups(core_type);
    return kernel_groups_[core_type];
}

KernelGroup *Program::kernels_on_core(const CoreCoord &core, const CoreType &core_type) {
    update_kernel_groups(core_type);
    if (core.x >= grid_extent_[core_type].x || core.y >= grid_extent_[core_type].y)
        return nullptr;
    uint8_t index = core_to_kernel_group_index_table_[core_type].at(core.y * grid_extent_[core_type].x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : &kernel_groups_[core_type].at(index);
}

struct KernelGroupInt {
    bool valid;
    std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX> kernel_ids;

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

void Program::update_kernel_groups(const CoreType &core_type) {
    if (core_to_kernel_group_index_table_[core_type].size() == 0) {
        bool erisc_is_idle = false;

        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(), std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_[core_type] = {0, 0};
        for (auto [id, kernel] : kernels_[core_type]) {
            for (auto core : kernel->logical_cores()) {
                if (core.x > grid_extent_[core_type].x)
                    grid_extent_[core_type].x = core.x;
                if (core.y > grid_extent_[core_type].y)
                    grid_extent_[core_type].y = core.y;
                if (core.x < base.x)
                    base.x = core.x;
                if (core.y < base.y)
                    base.y = core.y;
            }
            erisc_is_idle = kernel->is_idle_eth();
        }
        grid_extent_[core_type].x++;
        grid_extent_[core_type].y++;

        // grid maps cores to sets-of-kernels running on that core
        std::vector<KernelGroupInt> grid;
        grid.resize(grid_extent_[core_type].x * grid_extent_[core_type].y);
        for (auto [id, kernel] : kernels_[core_type]) {
            for (auto core : kernel->logical_cores()) {
                int core_index = core.y * grid_extent_[core_type].x + core.x;
                grid[core_index].valid = true;
                grid[core_index].update(kernel->dispatch_class(), id);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::unordered_map<KernelGroupInt, std::set<CoreRange>, KernelGroupIntHasher> map;
        for (auto y = base.y; y < grid_extent_[core_type].y; y++) {
            for (auto x = base.x; x < grid_extent_[core_type].x; x++) {
                int index = y * grid_extent_[core_type].x + x;
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
        core_to_kernel_group_index_table_[core_type].resize(
            grid_extent_[core_type].x * grid_extent_[core_type].y, core_to_kernel_group_invalid_index);
        for (auto &kg_to_cores : map) {
            int last_cb_index = -1;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : kg_to_cores.second) {
                for (auto y = range.start.y; y <= range.end.y; y++) {
                    for (auto x = range.start.x; x <= range.end.x; x++) {
                        core_to_kernel_group_index_table_[core_type][y * grid_extent_[core_type].x + x] = index;

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

            kernel_groups_[core_type].push_back(KernelGroup(
                *this,
                core_type,
                kg_to_cores.first.kernel_ids,
                erisc_is_idle,
                last_cb_index,
                kg_to_cores.second));
            index++;
        }
    }
}

void Program::CircularBufferAllocator::mark_address(uint64_t address, uint64_t size) {
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
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
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

void Program::allocate_circular_buffers() {
    ZoneScoped;
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    for (std::shared_ptr<CircularBuffer> circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
            continue;
        }

        uint64_t computed_addr = L1_UNRESERVED_BASE;
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
                    cb_allocator.mark_address(computed_addr, circular_buffer->size());
                }
            }
        }

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
    std::optional<uint64_t> lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids[0]);
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        uint64_t cb_region_end = cb_allocator.get_cb_region_end();
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

void Program::init_semaphores(const Device &device, const CoreCoord &logical_core, const CoreType core_type) const {
    auto semaphores_on_core = this->semaphores_on_core(logical_core);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(
            device.id(),
            device.physical_core_from_logical_core(logical_core, core_type),
            {semaphore.get().initial_value()},
            semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet &crs, uint32_t address, uint32_t init_value, CoreType core_type) {
    this->invalidate_compile();
    semaphores_.emplace_back(Semaphore(crs, address, init_value, core_type));
}

void Program::add_config_buffer(std::shared_ptr<Buffer> config_buffer) { config_buffers_.emplace_back(config_buffer); }

std::unordered_map<CoreType, std::vector<CoreCoord>> Program::logical_cores() const {
    std::unordered_map<CoreType, std::vector<CoreCoord>> cores_in_program;
    std::unordered_map<CoreType, std::set<CoreCoord>> unique_cores;
    for (auto [core_type, kernels] : kernels_) {
        if (cores_in_program.find(core_type) == cores_in_program.end()) {
            cores_in_program.insert({core_type, {}});
        }
        if (unique_cores.find(core_type) == unique_cores.end()) {
            unique_cores.insert({core_type, {}});
        }
        for (auto [id, kernel] : kernels) {
            for (auto core : kernel->logical_cores()) {
                if (unique_cores.at(core_type).find(core) != unique_cores.at(core_type).end()) {
                    continue;
                }
                unique_cores.at(core_type).insert(core);
                cores_in_program.at(core_type).push_back(core);
            }
        }
    }
    return cores_in_program;
}

void Program::construct_core_range_set_for_worker_cores() {
    bool found_kernels = false;
    for (auto [id, kernel] : kernels_[CoreType::WORKER]) {
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

void Program::invalidate_compile() {
    for (auto &[device_id, compile_needed] : compile_needed_) {
        compile_needed = true;
    }
}

void Program::populate_dispatch_data(Device *device) {
    static const map<RISCV, uint32_t> processor_to_local_mem_addr = {
        {RISCV::BRISC, MEM_BRISC_INIT_LOCAL_L1_BASE},
        {RISCV::NCRISC, MEM_NCRISC_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC0, MEM_TRISC0_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC1, MEM_TRISC1_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC2, MEM_TRISC2_INIT_LOCAL_L1_BASE},
        {RISCV::ERISC, eth_l1_mem::address_map::FIRMWARE_BASE}};

    auto extract_dst_noc_unicast_info =
        [&device](const set<CoreRange> &ranges, const CoreType core_type) -> vector<pair<transfer_info_cores, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info;
        for (const CoreRange &core_range : ranges) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
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
            vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                extract_dst_noc_multicast_info<std::set<CoreRange>>(
                    device, semaphore.core_range_set().ranges(), semaphore.core_type());
            transfer_info_2 transfer_info = {
                .dst_base_addr = semaphore.address(),
                .dst_noc_info = dst_noc_multicast_info,
                .linked = false,
                .data = semaphore_data};
            this->program_transfer_info.multicast_semaphores[semaphore.address()].push_back(transfer_info);
        } else if (semaphore.core_type() == CoreType::ETH) {
            vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                extract_dst_noc_unicast_info(semaphore.core_range_set().ranges(), semaphore.core_type());
            transfer_info_2 transfer_info = {
                .dst_base_addr = semaphore.address(),
                .dst_noc_info = dst_noc_unicast_info,
                .linked = false,
                .data = semaphore_data};
            this->program_transfer_info.unicast_semaphores[semaphore.address()].push_back(transfer_info);
        }
    }

    // Circular Buffer Configs handled in EnqueueProgram

    // Assume here and in command queue that kg_buffers is populated with multicast buffers first then unicast buffers
    // Program Binaries and Go Signals
    // TODO: cleanup put the WORKERS and ETH logic together..
    for (KernelGroup &kernel_group : this->get_kernel_groups(CoreType::WORKER)) {
        vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info = extract_dst_noc_multicast_info<std::set<CoreRange>>(
            device, kernel_group.core_ranges.ranges(), kernel_group.get_core_type());

        // So far, we don't support linking optimizations for kernel groups
        // which use multiple core ranges
        bool linked = dst_noc_multicast_info.size() == 1;
        vector<KernelHandle> kernel_ids;
        for (auto& optional_id : kernel_group.kernel_ids) {
            if (optional_id) {
                kernel_ids.push_back(optional_id.value());
            }
        }
        for (size_t i = 0; i < kernel_ids.size(); i++) {
            KernelHandle kernel_id = kernel_ids[i];
            vector<RISCV> sub_kernels;
            std::shared_ptr<Kernel> kernel = detail::GetKernel(*this, kernel_id);
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }

            uint32_t sub_kernel_index = 0;
            const auto &binaries = kernel->binaries(device->build_key());

            for (size_t j = 0; j < binaries.size(); j++) {
                const ll_api::memory &kernel_bin = binaries[j];
                uint32_t k = 0;
                uint32_t num_spans = kernel_bin.num_spans();

                std::vector<uint32_t> dst_base_addrs(num_spans);
                std::vector<uint32_t> page_offsets(num_spans);
                std::vector<uint32_t> lengths(num_spans);
                vector<uint32_t> binaries_data;

                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                    linked &= (i != kernel_ids.size() - 1) or (j != binaries.size() - 1) or (k != num_spans - 1);
                    uint64_t relo_addr =
                        tt::llrt::relocate_dev_addr(dst, processor_to_local_mem_addr.at(sub_kernels[sub_kernel_index]));

                    dst_base_addrs[k] = (uint32_t)relo_addr;
                    page_offsets[k] = binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    lengths[k] = len * sizeof(uint32_t);

                    binaries_data.resize(binaries_data.size() + len);
                    std::copy(mem_ptr, mem_ptr + len, binaries_data.end() - len);
                    binaries_data.resize(
                        align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)), 0);
                    k++;
                });
                kernel_bins_transfer_info kernel_bins_transfer_info = {
                    .dst_base_addrs = dst_base_addrs,
                    .page_offsets = page_offsets,
                    .lengths = lengths,
                    .dst_noc_info = dst_noc_multicast_info,
                    .linked = false,
                    .data = binaries_data};
                this->program_transfer_info.kernel_bins.push_back(kernel_bins_transfer_info);

                this->kg_buffers.push_back(std::make_unique<Buffer>(
                    device,
                    binaries_data.size() * sizeof(uint32_t),
                    HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                    BufferType::DRAM));
                sub_kernel_index++;
            }
        }
    }
    for (KernelGroup &kernel_group : this->get_kernel_groups(CoreType::ETH)) {
        vector<pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
            extract_dst_noc_unicast_info(kernel_group.core_ranges.ranges(), kernel_group.get_core_type());

        vector<KernelHandle> kernel_ids;
        if (kernel_group.core_type == CoreType::ETH && kernel_group.kernel_ids[DISPATCH_CLASS_ETH_DM0]) {
            kernel_ids.push_back(kernel_group.kernel_ids[DISPATCH_CLASS_ETH_DM0].value());
        }
        for (size_t i = 0; i < kernel_ids.size(); i++) {
            KernelHandle kernel_id = kernel_ids[i];
            vector<RISCV> sub_kernels;
            std::shared_ptr<Kernel> kernel = detail::GetKernel(*this, kernel_id);
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }

            uint32_t sub_kernel_index = 0;
            const auto &binaries = kernel->binaries(device->build_key());
            for (size_t j = 0; j < binaries.size(); j++) {
                const ll_api::memory &kernel_bin = binaries[j];
                uint32_t k = 0;
                uint32_t num_spans = kernel_bin.num_spans();

                std::vector<uint32_t> dst_base_addrs(num_spans);
                std::vector<uint32_t> page_offsets(num_spans);
                std::vector<uint32_t> lengths(num_spans);
                vector<uint32_t> binaries_data;

                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                    uint64_t relo_addr =
                        tt::llrt::relocate_dev_addr(dst, processor_to_local_mem_addr.at(sub_kernels[sub_kernel_index]));

                    dst_base_addrs[k] = (uint32_t)relo_addr;
                    page_offsets[k] = binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    lengths[k] = len * sizeof(uint32_t);

                    binaries_data.resize(binaries_data.size() + len);
                    std::copy(mem_ptr, mem_ptr + len, binaries_data.end() - len);
                    binaries_data.resize(
                        align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)), 0);
                    k++;
                });
                kernel_bins_transfer_info kernel_bins_transfer_info = {
                    .dst_base_addrs = dst_base_addrs,
                    .page_offsets = page_offsets,
                    .lengths = lengths,
                    .dst_noc_info = dst_noc_unicast_info,
                    .linked = false,
                    .data = binaries_data};
                this->program_transfer_info.kernel_bins.push_back(kernel_bins_transfer_info);

                this->kg_buffers.push_back(std::make_unique<Buffer>(
                    device,
                    binaries_data.size() * sizeof(uint32_t),
                    HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                    BufferType::DRAM));
                sub_kernel_index++;
            }
        }
    }
    std::uint32_t num_active_cores = 0;
    num_active_cores += this->logical_cores().at(CoreType::WORKER).size();
    num_active_cores += this->logical_cores().at(CoreType::ETH).size();
    this->program_transfer_info.num_active_cores = num_active_cores;

    return;
}

template <typename T, std::size_t dim2, std::size_t dim1, std::size_t dim0>
using Array3D = std::array<std::array<std::array<T, dim0>, dim1>, dim2>;

void Program::finalize_rt_args() {

    // Iterate over kernels in the program and "levels" the number of RTAs based on the max
    // Unique RTAs are packed across dispatch classes
    // Common RTAs come after unique RTAs and are also packed
    static vector<CoreType>core_types = { CoreType::WORKER, CoreType::ETH }; // TODO: make this global
    vector<uint32_t> max_rtas(DISPATCH_CLASS_MAX);
    vector<uint32_t> max_crtas(DISPATCH_CLASS_MAX);

    for (CoreType core_type : core_types) {
        uint32_t unique_rta_size = 0;

        for (auto& kg : this->get_kernel_groups(core_type)) {
            for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
                max_rtas[dispatch_class] = 0;
                auto& optional_id = kg.kernel_ids[dispatch_class];
                if (optional_id) {
                    auto kernel = detail::GetKernel(*this, optional_id.value());
                    for (const CoreRange &core_range : kg.core_ranges.ranges()) {
                        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
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
                    kg.launch_msg.mem_map[dispatch_class].rta_offset = offset;
                    offset += max_rtas[dispatch_class] * sizeof(uint32_t);
                } else {
                    kg.launch_msg.mem_map[dispatch_class].rta_offset = 0;
                }
            }

            kg.total_rta_size = offset;
            offset = align(offset, L1_ALIGNMENT);
            unique_rta_size = offset;
        }

        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            max_crtas[dispatch_class] = 0;
        }
        // Find the max # common RTAs across all kernels for each dispatch class
        for (size_t kernel_id = 0; kernel_id < this->num_kernels(); kernel_id++) {
            auto kernel = detail::GetKernel(*this, kernel_id);
            if (core_type == kernel->get_kernel_core_type()) {
                uint32_t dispatch_class = kernel->dispatch_class();
                max_crtas[dispatch_class] =
                    std::max(max_crtas[dispatch_class], (uint32_t)kernel->common_runtime_args().size());
            }
        }

        // Calculate the address offset and size for common RTAs for each dispatch class
        uint32_t offset = 0;
        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            uint32_t size = max_crtas[dispatch_class] * sizeof(uint32_t);
            this->crta_offsets[core_type == CoreType::WORKER][dispatch_class] = unique_rta_size + offset;
            this->crta_sizes[core_type == CoreType::WORKER][dispatch_class] = size;
            offset += size;
            offset = align(offset, L1_ALIGNMENT);
        }

        // Set the runtime_args_data sizing info based on the shared max
        for (size_t kernel_id = 0; kernel_id < this->num_kernels(); kernel_id++) {
            auto kernel = detail::GetKernel(*this, kernel_id);
            if (core_type == kernel->get_kernel_core_type()) {
                uint32_t dispatch_class = kernel->dispatch_class();
                kernel->set_common_runtime_args_count(max_crtas[dispatch_class]);
            }
        }

        // Set the kernel group common runtime arg offsets use in the launch message
        for (auto& kg : this->get_kernel_groups(core_type)) {
            for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
                kg.launch_msg.mem_map[dispatch_class].crta_offset = this->crta_offsets[core_type == CoreType::WORKER][dispatch_class];
            }
        }

        // TODO: this is asserted here as the leveling above can break the limits enforced by the API
        // Once we use a ring buffer, memory space will be dynamic and this assert won't matter
        TT_FATAL(offset <= L1_KERNEL_CONFIG_SIZE);
    }
}

void Program::compile(Device *device) {
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

    for (auto &[core_type, kernels] : kernels_) {
        for (auto &[id, kernel] : kernels) {
            launch_build_step(
                [kernel, device, this] {
                    JitBuildOptions build_options(device->build_env());
                    kernel->set_build_options(build_options);
                    this->set_cb_data_fmt(device, kernel->logical_coreranges(), build_options);

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

    for (auto &[core_type, kernels] : kernels_) {
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
    this->loaded_onto_device = false;
}

Program::~Program() {}
}  // namespace tt::tt_metal
