// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "common/executor.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/jit_build/genfiles.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"

namespace tt::tt_metal {


namespace{
    std::atomic<bool> enable_persistent_kernel_cache = false;

    void GenerateBinaries(Device *device, JitBuildOptions& build_options, std::shared_ptr<Kernel> kernel) {
        ZoneScoped;
        const std::string tracyPrefix = "GenerateBinaries_";
        ZoneName( (tracyPrefix + build_options.name).c_str(), build_options.name.length() + tracyPrefix.length());
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

    size_t KernelCompileHash(
        const std::shared_ptr<Kernel> kernel, JitBuildOptions &build_options, const chip_id_t &device_id) {
        // Account for device id in hash because generated headers are dependent on harvesting config, which can differ per device
        // This can be removed with https://github.com/tenstorrent-metal/tt-metal/issues/3381

        // Also account for watcher/dprint enabled in hash because they enable additional code to
        // be compiled into the kernel.
        string compile_hash_str = fmt::format(
            "{}_{}_{}_{}_{}",
            device_id,
            std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc)),
            kernel->compute_hash(),
            tt::llrt::OptionsG.get_watcher_enabled(),
            tt::llrt::OptionsG.get_dprint_enabled()
        );
        size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

    #ifdef GENERATE_HASH_LOG
        static std::ofstream f("/tmp/hashlog.txt");
        static std::mutex mutex_;
        {
            unique_lock<mutex> lock;
            f << kernel->name() << " :: "
            << device_id << "::"
            << std::hash<tt_hlk_desc>{}(build_options.hlk_desc) << " :: "
            << kernel->compute_hash() << " :: "
            << compile_hash_str << " "
            << compile_hash << std::endl << std::flush;
        }
    #endif
        return compile_hash;
    }

}
namespace detail{
    void EnablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = true;
    }

    void DisablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = false;
    }
}


std::atomic<uint64_t> Program::program_counter = 0;

Program::Program(): id(program_counter++),worker_crs_({}), local_circular_buffer_allocation_needed_(false), loaded_onto_device(false) {
    std::set<CoreType> supported_core_types = {CoreType::WORKER, CoreType::ETH};
    for (const auto& core_type : supported_core_types) {
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
    //TT_ASSERT(kernel_id < this->kernels_.size(), "Expected Kernel with ID {} to be in Program {}", kernel_id, this->id);
    // find coretype based on kernel_id
    for (const auto &[core_type, kernels] : this->kernels_) {
        if (kernels.find(kernel_id) != kernels.end()) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

KernelGroup::KernelGroup() : core_ranges({}) {
}

KernelGroup::KernelGroup(
    const Program &program,
    std::optional<KernelHandle> brisc_id,
    std::optional<KernelHandle> ncrisc_id,
    std::optional<KernelHandle> trisc_id,
    std::optional<KernelHandle> erisc_id,
    int last_cb_index,
    const CoreRangeSet &new_ranges) :
    core_ranges({}) {
    this->core_ranges = this->core_ranges.merge(new_ranges);

    this->riscv0_id = brisc_id;
    this->riscv1_id = ncrisc_id;
    this->compute_id = trisc_id;
    this->erisc_id = erisc_id;

    // The code below sets the brisc_noc_id for use by the device firmware
    // Use 0 if neither brisc nor trisc specify a noc
    this->launch_msg.brisc_noc_id = 0;
    if (brisc_id) {
        // Use brisc's noc if brisc specifies a noc
        this->launch_msg.enable_brisc = true;
        this->launch_msg.brisc_noc_id = std::get<DataMovementConfig>(program.get_kernel(brisc_id.value())->config()).noc;
        this->launch_msg.brisc_watcher_kernel_id = program.get_kernel(brisc_id.value())->get_watcher_kernel_id();
    } else {
        this->launch_msg.brisc_watcher_kernel_id = 0;
        this->launch_msg.enable_brisc = false;
    }

    if (ncrisc_id) {
        const auto kernel = program.get_kernel(ncrisc_id.value());
        // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
        // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
        this->launch_msg.enable_ncrisc = true;
        this->launch_msg.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
        this->launch_msg.ncrisc_kernel_size16 = kernel->get_binary_size16();
        this->launch_msg.ncrisc_watcher_kernel_id = kernel->get_watcher_kernel_id();
    } else {
        this->launch_msg.ncrisc_watcher_kernel_id = 0;
        this->launch_msg.enable_ncrisc = false;
        this->launch_msg.ncrisc_kernel_size16 = 0;
    }

    if (trisc_id) {
        this->launch_msg.enable_triscs = true;
        this->launch_msg.triscs_watcher_kernel_id = program.get_kernel(trisc_id.value())->get_watcher_kernel_id();
    } else {
        this->launch_msg.triscs_watcher_kernel_id = 0;
        this->launch_msg.enable_triscs = false;
    }

    if (erisc_id) {
        this->launch_msg.enable_erisc = true;
        // Ethernet cores use the brisc kernel id field
        this->launch_msg.brisc_watcher_kernel_id = program.get_kernel(erisc_id.value())->get_watcher_kernel_id();
    } else {
        this->launch_msg.enable_erisc = false;
    }

    this->launch_msg.max_cb_index = last_cb_index + 1;
    this->launch_msg.run = RUN_MSG_GO;
}

CoreType KernelGroup::get_core_type() const {
    if (this->erisc_id.has_value()) {
        return CoreType::ETH;
    } else {
        return CoreType::WORKER;
    }
};

std::vector<KernelGroup>& Program::get_kernel_groups(const CoreType &core_type) {
    update_kernel_groups(core_type);
    return kernel_groups_[core_type];
}

KernelGroup * Program::kernels_on_core(const CoreCoord &core, const CoreType &core_type) {
    update_kernel_groups(core_type);
    if (core.x >= grid_extent_[core_type].x || core.y >= grid_extent_[core_type].y) return nullptr;
    uint8_t index = core_to_kernel_group_index_table_[core_type].at(core.y * grid_extent_[core_type].x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : &kernel_groups_[core_type].at(index);
}

struct KernelGroupInt {
    bool valid;
    std::optional<KernelHandle> trisc_id = std::nullopt;
    std::optional<KernelHandle> brisc_id = std::nullopt;
    std::optional<KernelHandle> ncrisc_id = std::nullopt;
    std::optional<KernelHandle> erisc_id = std::nullopt;

    bool operator==(const KernelGroupInt& b) const;
    void update(RISCV riscv_processor, size_t kernel_idx) {
        switch (riscv_processor) {
        case RISCV::BRISC:
            this->brisc_id = static_cast<KernelHandle>(kernel_idx);
            break;
        case RISCV::NCRISC:
            this->ncrisc_id = static_cast<KernelHandle>(kernel_idx);
            break;
        case RISCV::COMPUTE:
            this->trisc_id = static_cast<KernelHandle>(kernel_idx);
            break;
        case RISCV::ERISC:
            this->erisc_id = static_cast<KernelHandle>(kernel_idx);
            break;
        default:
            TT_ASSERT(false, "Unsupported kernel processor!");
        }
    }
};

bool KernelGroupInt::operator==(const KernelGroupInt& b) const {
    return trisc_id == b.trisc_id && brisc_id == b.brisc_id && ncrisc_id == b.ncrisc_id && erisc_id == b.erisc_id;
}

struct KernelGroupIntHasher {
    std::size_t operator()(const KernelGroupInt& x) const {
        return static_cast<size_t>(x.erisc_id.value_or(0)) | static_cast<size_t>(x.trisc_id.value_or(0)) |
               static_cast<size_t>(x.brisc_id.value_or(0)) << 16 | static_cast<size_t>(x.ncrisc_id.value_or(0)) << 32;
    }
};

void Program::update_kernel_groups(const CoreType &core_type) {
    if (core_to_kernel_group_index_table_[core_type].size() == 0) {
        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(),
                          std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_[core_type] = {0, 0};
        for (auto [id, kernel] : kernels_[core_type]) {
            for (auto core : kernel->logical_cores()) {
                if (core.x > grid_extent_[core_type].x) grid_extent_[core_type].x = core.x;
                if (core.y > grid_extent_[core_type].y) grid_extent_[core_type].y = core.y;
                if (core.x < base.x) base.x = core.x;
                if (core.y < base.y) base.y = core.y;
            }
        }
        grid_extent_[core_type].x++;
        grid_extent_[core_type].y++;

        // grid maps cores to sets-of-kernels running on that core
        std::vector<KernelGroupInt> grid;
        grid.resize(grid_extent_[core_type].x * grid_extent_[core_type].y);
        for (auto [id, kernel]: kernels_[core_type]) {
            for (auto core : kernel->logical_cores()) {
                int core_index = core.y * grid_extent_[core_type].x + core.x;
                grid[core_index].valid = true;
                grid[core_index].update(kernel->processor(), id);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::unordered_map<KernelGroupInt, std::set<CoreRange>, KernelGroupIntHasher> map;
        for (auto y = base.y; y < grid_extent_[core_type].y; y++) {
            for (auto x = base.x; x < grid_extent_[core_type].x; x++) {
                int index = y * grid_extent_[core_type].x + x;
                if (grid[index].valid) {
                    std::set<CoreRange>& set = map[grid[index]];
                    set.insert(CoreRange({x, y}, {x, y}));
                }
            }
        }

        // Build the list of KernelGroups with merged core range sets from the
        // mapping of sets-of-kernels to cores
        TT_ASSERT(map.size() < core_to_kernel_group_invalid_index);
        kernel_groups_.reserve(map.size());
        int index = 0;
        core_to_kernel_group_index_table_[core_type].resize(grid_extent_[core_type].x * grid_extent_[core_type].y, core_to_kernel_group_invalid_index);
        for (auto& kg_to_cores : map) {

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
                kg_to_cores.first.brisc_id,
                kg_to_cores.first.ncrisc_id,
                kg_to_cores.first.trisc_id,
                kg_to_cores.first.erisc_id,
                last_cb_index,
                kg_to_cores.second));
            index++;
        }
    }
}

void Program::CircularBufferAllocator::mark_address(uint64_t address, uint64_t size) {
    auto &last_region = this->l1_regions.back();
    if (address < last_region.second) {
        TT_THROW("Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address", address, last_region.first, last_region.second);
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
    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory allocator
    if (not circular_buffer->globally_allocated()) {
        this->invalidate_circular_buffer_allocation();
    }
    else {
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
                        TT_THROW("Invalid circular buffer index: {} should be between 0 and {}", buffer_index, NUM_CIRCULAR_BUFFERS);
                    }
                    if (cb_indices.to_ulong() & (1 << buffer_index)) {
                        TT_THROW("Invalid circular buffer index: Cannot add circular buffer at index {}, another circular buffer already exists", buffer_index);
                    }
                    cb_indices[buffer_index] = 1;
                }
            }
        }

        // There is one CircularBufferAllocator per unique core range, create one if it does not already exist for current core range
        auto val = std::find_if(cb_allocators_.begin(), cb_allocators_.end(), [&core_range](const CircularBufferAllocator &cb_allocator) {
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

const std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_corerange(const CoreRange & cr) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_corerange(cr)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
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
                        // Intersecting core range has already been marked to have allocation at this address. This could have been marked by a circular buffer on a core range disjoint from
                        // current `core_range` but also intersecting `cb_allocator.core_range`
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
    const std::vector<uint32_t> &bank_ids = device->bank_ids_from_logical_core(*device->compute_cores_.begin());
    std::optional<uint64_t> lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids[0]);
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        uint64_t cb_region_end = cb_allocator.get_cb_region_end();
        if (cb_region_end > max_l1_size) {
            TT_THROW("Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} B", cb_allocator.core_range.str(), cb_region_end, max_l1_size);
        }
        if (lowest_address.has_value() and lowest_address.value() < cb_region_end) {
            TT_THROW("Statically allocated circular buffers in program {} clash with L1 buffers on core range {}. L1 buffer allocated at {} and static circular buffer region ends at {}", this->id, cb_allocator.core_range.str(), lowest_address.value(), cb_region_end);
        }
    }
}

size_t Program::num_semaphores(const CoreCoord &core) const {
    return semaphores_on_core(core).size();
}

size_t Program::num_semaphores() const {
    return semaphores_.size();
}

void Program::init_semaphores( const Device & device, const CoreCoord &logical_core, const CoreType core_type) const{
    auto semaphores_on_core = this->semaphores_on_core(logical_core);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(device.id(), device.physical_core_from_logical_core(logical_core, core_type), {semaphore.get().initial_value()}, semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value, CoreType core_type) {
    this->invalidate_compile();
    semaphores_.emplace_back(Semaphore( crs, address, init_value, core_type));
}

std::unordered_map<CoreType, std::vector<CoreCoord>> Program::logical_cores() const {
    std::unordered_map<CoreType, std::vector<CoreCoord>> cores_in_program;
    std::unordered_map<CoreType, std::set<CoreCoord>> unique_cores;
    for (auto [core_type, kernels] : kernels_){
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
    for (auto [id, kernel] : kernels_[CoreType::WORKER]){
        this->worker_crs_ = this->worker_crs_.merge ( kernel->core_range_set() );
        found_kernels = true;
    }
    TT_ASSERT(!found_kernels || this->worker_crs_.ranges().size() >= 1, "Invalid core range set");
}

void Program::set_cb_data_fmt(
    Device *device, const std::vector<CoreRange> & crs, JitBuildOptions &build_options) const {
    ZoneScoped;
    for (auto logical_cr : crs) {
        auto cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(static_cast<CB>(buffer_index), circular_buffer->data_format(buffer_index));
            }
        }
    }
}

void Program::invalidate_compile() {
    for (auto &[device_id, compile_needed] : compile_needed_) {
        compile_needed = true;
    }
}

ProgramDeviceMap ConstructProgramDeviceMap(const Device* device, Program& program) {
    std::unordered_map<PageTransferType, vector<transfer_info>> program_page_transfers = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<transfer_info>> runtime_arg_page_transfers = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<transfer_info>> cb_config_page_transfers = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<transfer_info>> go_signal_page_transfers = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<uint32_t>> num_transfers_in_program_pages = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<uint32_t>> num_transfers_in_runtime_arg_pages = {
        {PageTransferType::MULTICAST, {}},
        {PageTransferType::UNICAST,
         {}}};  // Corresponds to the number of transfers within host data pages across all host data pages
    std::unordered_map<PageTransferType, vector<uint32_t>> num_transfers_in_cb_config_pages = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};
    std::unordered_map<PageTransferType, vector<uint32_t>> num_transfers_in_go_signal_pages = {
        {PageTransferType::MULTICAST, {}}, {PageTransferType::UNICAST, {}}};

    static const map<RISCV, uint32_t> processor_to_local_mem_addr = {
        {RISCV::BRISC, MEM_BRISC_INIT_LOCAL_L1_BASE},
        {RISCV::NCRISC, MEM_NCRISC_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC0, MEM_TRISC0_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC1, MEM_TRISC1_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC2, MEM_TRISC2_INIT_LOCAL_L1_BASE},
        {RISCV::ERISC, eth_l1_mem::address_map::FIRMWARE_BASE}};

    static const map<RISCV, uint32_t> processor_to_l1_arg_base_addr = {
        {RISCV::BRISC, BRISC_L1_ARG_BASE},
        {RISCV::NCRISC, NCRISC_L1_ARG_BASE},
        {RISCV::COMPUTE, TRISC_L1_ARG_BASE},
        {RISCV::ERISC, eth_l1_mem::address_map::ERISC_L1_ARG_BASE},
    };

    uint32_t num_transfers_within_page = 0;

    uint32_t src = 0;
    vector<uint32_t> program_pages;
    uint32_t program_page_idx = 0;
    uint32_t program_new_page_tracker = 0;
    constexpr static uint32_t noc_transfer_alignment_in_bytes = 16;

    auto update_program_page_transfers = [&num_transfers_within_page](
                                             uint32_t src,
                                             uint32_t num_bytes,
                                             uint32_t dst,
                                             vector<transfer_info>& transfers,
                                             vector<uint32_t>& num_transfers_per_page,
                                             const vector<pair<uint32_t, uint32_t>>& dst_noc_transfer_info,
                                             bool linked = false) -> uint32_t {
        while (num_bytes) {
            uint32_t num_bytes_left_in_page = DeviceCommand::PROGRAM_PAGE_SIZE - (src % DeviceCommand::PROGRAM_PAGE_SIZE);
            uint32_t num_bytes_in_transfer = std::min(num_bytes_left_in_page, num_bytes);
            src = align(src + num_bytes_in_transfer, noc_transfer_alignment_in_bytes);

            uint32_t transfer_instruction_idx = 1;
            for (const auto& [dst_noc_encoding, num_receivers] : dst_noc_transfer_info) {
                bool last = transfer_instruction_idx == dst_noc_transfer_info.size();
                transfer_info transfer_instruction = {.size_in_bytes = num_bytes_in_transfer, .dst = dst, .dst_noc_encoding = dst_noc_encoding, .num_receivers = num_receivers, .last_transfer_in_group = last, .linked = linked};
                transfers.push_back(transfer_instruction);
                num_transfers_within_page++;
                transfer_instruction_idx++;
            }

            dst += num_bytes_in_transfer;
            num_bytes -= num_bytes_in_transfer;

            if ((src % DeviceCommand::PROGRAM_PAGE_SIZE) == 0) {
                num_transfers_per_page.push_back(num_transfers_within_page);
                num_transfers_within_page = 0;
            }
        }

        return src;
    };

    auto extract_dst_noc_multicast_info =
        [&device](const set<CoreRange>& ranges, const CoreType core_type) -> vector<pair<uint32_t, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info;
        for (const CoreRange& core_range : ranges) {
            CoreCoord physical_start = device->physical_core_from_logical_core(core_range.start, core_type);
            CoreCoord physical_end = device->physical_core_from_logical_core(core_range.end, core_type);

            uint32_t dst_noc_multicast_encoding = NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);

            uint32_t num_receivers = core_range.size();
            dst_noc_multicast_info.push_back(std::make_pair(dst_noc_multicast_encoding, num_receivers));
        }
        return dst_noc_multicast_info;
    };

    auto update_program_pages_with_new_page = [&program_pages, &src, &program_new_page_tracker]() {
        program_pages.resize(program_pages.size() + align(src, DeviceCommand::PROGRAM_PAGE_SIZE) / sizeof(uint32_t), 0);
        src = 0;
        program_new_page_tracker++;
    };
    auto align_program_page_idx_to_new_page = [&program_page_idx, &program_new_page_tracker]() {
        program_page_idx = align(program_page_idx, DeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t));
        program_new_page_tracker--;
    };

    auto get_noc_unicast_encoding = [](const CoreCoord& coord) {
        return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y));
    };

    auto update_program_page_for_kernel_group =
        [&program_page_transfers,
         &num_transfers_in_program_pages,
         &update_program_page_transfers,
         &extract_dst_noc_multicast_info,
         &device,
         &program,
         &get_noc_unicast_encoding](uint32_t src, const KernelGroup& kernel_group, PageTransferType page_transfer_type) -> uint32_t {
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(kernel_group.core_ranges.ranges(), kernel_group.get_core_type());

        // So far, we don't support linking optimizations for kernel groups
        // which use multiple core ranges
        bool linked = dst_noc_multicast_info.size() == 1;

        vector<KernelHandle> kernel_ids;
        if (kernel_group.riscv0_id)
            kernel_ids.push_back(kernel_group.riscv0_id.value());
        if (kernel_group.riscv1_id)
            kernel_ids.push_back(kernel_group.riscv1_id.value());
        if (kernel_group.compute_id)
            kernel_ids.push_back(kernel_group.compute_id.value());
        if (kernel_group.erisc_id)
            kernel_ids.push_back(kernel_group.erisc_id.value());

        for (size_t i = 0; i < kernel_ids.size(); i++) {
            KernelHandle kernel_id = kernel_ids[i];
            vector<RISCV> sub_kernels;
            std::shared_ptr<Kernel> kernel = detail::GetKernel(program, kernel_id);
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }

            uint32_t sub_kernel_index = 0;
            const auto& binaries = kernel->binaries(device->id());
            for (size_t j = 0; j < binaries.size(); j++) {
                const ll_api::memory& kernel_bin = binaries[j];

                uint32_t k = 0;
                uint32_t num_spans = kernel_bin.num_spans();
                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                    linked &= (i != kernel_ids.size() - 1) or (j != binaries.size() - 1) or (k != num_spans - 1);
                    uint64_t relo_addr =
                        tt::llrt::relocate_dev_addr(dst, processor_to_local_mem_addr.at(sub_kernels[sub_kernel_index]));

                    uint32_t num_bytes = len * sizeof(uint32_t);

                    if (page_transfer_type == PageTransferType::UNICAST) {
                        for (const auto& logical_core : kernel->logical_cores()) {
                            uint32_t dst_noc = get_noc_unicast_encoding(
                                device->physical_core_from_logical_core(logical_core, kernel_group.get_core_type()));
                            src = update_program_page_transfers(
                                src,
                                num_bytes,
                                relo_addr,
                                program_page_transfers.at(PageTransferType::UNICAST),
                                num_transfers_in_program_pages.at(PageTransferType::UNICAST),
                                {{dst_noc, 1}});
                        }
                    } else if (page_transfer_type == PageTransferType::MULTICAST) {
                        src = update_program_page_transfers(
                            src,
                            num_bytes,
                            relo_addr,
                            program_page_transfers.at(PageTransferType::MULTICAST),
                            num_transfers_in_program_pages.at(PageTransferType::MULTICAST),
                            dst_noc_multicast_info,
                            linked);
                    }
                    k++;
                });
                sub_kernel_index++;
            }
        }
        return src;
    };
    auto populate_program_binaries_pages =
        [&program_pages, &program_page_idx, &device, &program](const KernelGroup& kernel_group) {
            vector<KernelHandle> kernel_ids;
            if (kernel_group.riscv0_id)
                kernel_ids.push_back(kernel_group.riscv0_id.value());
            if (kernel_group.riscv1_id)
                kernel_ids.push_back(kernel_group.riscv1_id.value());
            if (kernel_group.compute_id)
                kernel_ids.push_back(kernel_group.compute_id.value());
            if (kernel_group.erisc_id)
                kernel_ids.push_back(kernel_group.erisc_id.value());
            for (KernelHandle kernel_id : kernel_ids) {
                auto kernel = detail::GetKernel(program, kernel_id);

                for (const ll_api::memory& kernel_bin : kernel->binaries(device->id())) {
                    kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                        std::copy(mem_ptr, mem_ptr + len, program_pages.begin() + program_page_idx);

                        program_page_idx =
                            align(program_page_idx + len, noc_transfer_alignment_in_bytes / sizeof(uint32_t));
                    });
                }
            }
        };

    // Step 1: Get transfer info for runtime args (soon to just be host data). We
    // want to send host data first because of the higher latency to pull
    // in host data.
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        uint32_t dst = processor_to_l1_arg_base_addr.at(kernel->processor());
        const auto& kernel_core_type = kernel->get_kernel_core_type();
        for (const auto& core_coord : kernel->cores_with_runtime_args()) {
            CoreCoord physical_core =
                device->physical_core_from_logical_core(core_coord, kernel->get_kernel_core_type());
            const auto& runtime_args = kernel->runtime_args(core_coord);
            uint32_t num_bytes = runtime_args.size() * sizeof(uint32_t);
            uint32_t dst_noc = get_noc_unicast_encoding(physical_core);

            // Only one receiver per set of runtime arguments
            src = update_program_page_transfers(
                src,
                num_bytes,
                dst,
                runtime_arg_page_transfers.at(PageTransferType::MULTICAST),
                num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST),
                {{dst_noc, 1}});
        }
    }

    // Cleanup step of separating runtime arg pages from program pages
    if (num_transfers_within_page) {
        num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    src = 0;  // Resetting since in a new page
    // Step 2: Continue constructing pages for circular buffer configs
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        // No CB support for ethernet cores
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(cb->core_ranges().ranges(), CoreType::WORKER);
        constexpr static uint32_t num_bytes = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
        for (const auto buffer_index : cb->buffer_indices()) {
            src = update_program_page_transfers(
                src,
                num_bytes,
                CIRCULAR_BUFFER_CONFIG_BASE + buffer_index * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t),
                cb_config_page_transfers.at(PageTransferType::MULTICAST),
                num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST),
                dst_noc_multicast_info);
        }
    }

    // Cleanup step of separating runtime arg pages from program pages
    if (num_transfers_within_page) {
        num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    // Split kernel groups by multicast/unicast, program multicast transfers first then unicast
    std::vector<KernelGroup> kernel_group_multicast;
    std::vector<KernelGroup> kernel_group_unicast;
    for (const KernelGroup& kernel_group : program.get_kernel_groups(CoreType::WORKER)) {
        kernel_group_multicast.emplace_back(kernel_group);
    }
    for (const KernelGroup& kernel_group : program.get_kernel_groups(CoreType::ETH)) {
        kernel_group_unicast.emplace_back(kernel_group);
    }
    // Enqueue program binaries and go siggals in this order:
    // - Multicast Program Binaries
    // - Unicast Program Binaries
    // - Multicast Go Signals
    // - Unicast Go Signals
    // This probably has better perf than sending binaries and go signals together:
    // - Multicast Program Binaries
    // - Multicast Go Signals
    // - Unicast Program Binaries
    // - Unicast Go Signals
    // Step 3a (Multicast): Determine the transfer information for each program binary
    src = 0;  // Restart src since multicast program binaries begins in a new page
    for (const KernelGroup& kernel_group : kernel_group_multicast) {
        src = update_program_page_for_kernel_group(src, kernel_group, PageTransferType::MULTICAST);
    }
    // Step 4 (Multicast): Continue constructing pages for semaphore configs, only multicast/worker cores supported
    for (const Semaphore& semaphore : program.semaphores()) {
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(semaphore.core_range_set().ranges(), semaphore.core_type());

        src = update_program_page_transfers(
            src,
            L1_ALIGNMENT,
            semaphore.address(),
            program_page_transfers.at(PageTransferType::MULTICAST),
            num_transfers_in_program_pages.at(PageTransferType::MULTICAST),
            dst_noc_multicast_info);
    }

    if (num_transfers_within_page) {
        num_transfers_in_program_pages.at(PageTransferType::MULTICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    // Step 3b (Unicast)
    // skipping step 4 since no semaphore support
    update_program_pages_with_new_page();  // sets src to 0 since unicast program binaries begins in new page
    for (const KernelGroup& kernel_group : kernel_group_unicast) {
        src = update_program_page_for_kernel_group(src, kernel_group, PageTransferType::UNICAST);
    }
    if (num_transfers_within_page) {
        num_transfers_in_program_pages.at(PageTransferType::UNICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    // Step 5a (Multicast): Continue constructing pages for GO signals, multicast first then unicast
    update_program_pages_with_new_page();  // sets src to 0 since multicast signals begins in new page
    for (KernelGroup& kernel_group : kernel_group_multicast) {
        kernel_group.launch_msg.mode = DISPATCH_MODE_DEV;
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(kernel_group.core_ranges.ranges(), kernel_group.get_core_type());
        src = update_program_page_transfers(
            src,
            sizeof(launch_msg_t),
            GET_MAILBOX_ADDRESS_HOST(launch),
            go_signal_page_transfers.at(PageTransferType::MULTICAST),
            num_transfers_in_go_signal_pages.at(PageTransferType::MULTICAST),
            dst_noc_multicast_info);
    }
    if (num_transfers_within_page) {
        num_transfers_in_go_signal_pages.at(PageTransferType::MULTICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    // Step 5b (Unicast)
    update_program_pages_with_new_page();  // sets src to 0 since unicast signals begins in new page
    for (KernelGroup& kernel_group : kernel_group_unicast) {
        kernel_group.launch_msg.mode = DISPATCH_MODE_DEV;
        if (kernel_group.get_core_type() == CoreType::ETH) {
            auto kernel = detail::GetKernel(program, kernel_group.erisc_id.value());
            for (const auto& logical_eth_core : kernel->logical_cores()) {
                uint32_t dst_noc =
                    get_noc_unicast_encoding(device->physical_core_from_logical_core(logical_eth_core, CoreType::ETH));
                src = update_program_page_transfers(
                    src,
                    sizeof(launch_msg_t),
                    GET_ETH_MAILBOX_ADDRESS_HOST(launch),
                    go_signal_page_transfers.at(PageTransferType::UNICAST),
                    num_transfers_in_go_signal_pages.at(PageTransferType::UNICAST),
                    {{dst_noc, 1}});
            }
        } else {
            TT_ASSERT(false, "All non-ethernet core go signals should be muticasted");
        }
    }
    if (num_transfers_within_page) {
        num_transfers_in_go_signal_pages.at(PageTransferType::UNICAST).push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    // Allocate some more space for GO signal
    update_program_pages_with_new_page();  // sets src to 0, but not needed

    // Create a vector of all program binaries/cbs/semaphores
    align_program_page_idx_to_new_page();
    for (const KernelGroup& kernel_group : kernel_group_multicast) {
        populate_program_binaries_pages(kernel_group);
    }

    for (const Semaphore& semaphore : program.semaphores()) {
        program_pages[program_page_idx] = semaphore.initial_value();
        program_page_idx += 4;
    }

    align_program_page_idx_to_new_page();
    for (const KernelGroup& kernel_group : kernel_group_unicast) {
        if (kernel_group.get_core_type() == CoreType::ETH) {
            auto kernel = detail::GetKernel(program, kernel_group.erisc_id.value());
            for (const auto& logical_core : kernel->logical_cores()) {
                populate_program_binaries_pages(kernel_group);
            }
        }
    }

    // Since GO signal begin in a new page, I need to advance my idx
    align_program_page_idx_to_new_page();
    // uint32_t dispatch_core_word = ((uint32_t)dispatch_core.y << 16) | dispatch_core.x;
    for (KernelGroup& kernel_group : kernel_group_multicast) {
        // TODO(agrebenisan): Hanging when we extend the launch msg. Needs to be investigated. For now,
        // only supporting enqueue program for cq 0 on a device.
        // kernel_group.launch_msg.dispatch_core_x = dispatch_core.x;
        // kernel_group.launch_msg.dispatch_core_y = dispatch_core.y;
        static_assert(sizeof(launch_msg_t) % sizeof(uint32_t) == 0);
        uint32_t* launch_message_data = (uint32_t*)&kernel_group.launch_msg;
        for (int i = 0; i < sizeof(launch_msg_t) / sizeof(uint32_t); i++) {
            program_pages[program_page_idx + i] = launch_message_data[i];
        }
        program_page_idx += sizeof(launch_msg_t) / sizeof(uint32_t);
    }

    align_program_page_idx_to_new_page();
    for (KernelGroup& kernel_group : kernel_group_unicast) {
        if (kernel_group.get_core_type() == CoreType::ETH) {
            auto kernel = detail::GetKernel(program, kernel_group.erisc_id.value());
            uint32_t* launch_message_data = (uint32_t*)&kernel_group.launch_msg;
            for (int i = 0; i < sizeof(launch_msg_t) / sizeof(uint32_t); i++) {
                program_pages[program_page_idx + i] = launch_message_data[i];
            }
            program_page_idx += sizeof(launch_msg_t) / sizeof(uint32_t);
        } else {
            TT_ASSERT(false, "All non-ethernet core go signals should be multicasted");
        }
    }

    TT_ASSERT(
        program_new_page_tracker == 0, "Number of new program pages not aligned between sizing and populating data.");

    uint32_t num_workers = 0;
    // Explicitly sum the worker and eth cores, since we don't have support for all core types
    num_workers += program.logical_cores().at(CoreType::WORKER).size();
    num_workers += program.logical_cores().at(CoreType::ETH).size();
    return {
        .num_workers = num_workers,
        .program_pages = std::move(program_pages),
        .program_page_transfers = std::move(program_page_transfers),
        .runtime_arg_page_transfers = std::move(runtime_arg_page_transfers),
        .cb_config_page_transfers = std::move(cb_config_page_transfers),
        .go_signal_page_transfers = std::move(go_signal_page_transfers),
        .num_transfers_in_program_pages = std::move(num_transfers_in_program_pages),
        .num_transfers_in_runtime_arg_pages = std::move(num_transfers_in_runtime_arg_pages),
        .num_transfers_in_cb_config_pages = std::move(num_transfers_in_cb_config_pages),
        .num_transfers_in_go_signal_pages = std::move(num_transfers_in_go_signal_pages),
    };
}

void Program::compile( Device * device )
{
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

    // compile all kernels in parallel
    for (auto &[core_type, kernels] : kernels_) {
       for (auto &[id, kernel]: kernels) {
            events.emplace_back ( detail::async ( [kernel, device, this] {

                JitBuildOptions build_options(device->build_env());
                kernel->set_build_options(build_options);
                this->set_cb_data_fmt(device, kernel->logical_coreranges(), build_options);

                auto kernel_hash = KernelCompileHash(kernel, build_options, device->id());
                std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
                kernel->set_full_name(kernel_path_suffix);
                build_options.set_name(kernel_path_suffix);

                bool cache_hit = true;
                bool path_exists = std::filesystem::exists(build_options.path);
                if ( enable_persistent_kernel_cache && path_exists ) {
                    if ( not detail::HashLookup::inst().exists(kernel_hash) ) detail::HashLookup::inst().add(kernel_hash);
                } else if ( detail::HashLookup::inst().add(kernel_hash) ) {
                    cache_hit = false;
                    GenerateBinaries(device, build_options, kernel);
                }
                if (detail::CompilationReporter::enabled()) {
                    detail::CompilationReporter::inst().add_kernel_compile_stats(*this, kernel, cache_hit, kernel_hash);
                }

                kernel->set_binary_path(build_options.path);
        } ) );
       }
    }

    for (auto & f : events)
        f.get();

    for (auto &[core_type, kernels] : kernels_) {
        for (auto &[id, kernel] : kernels) {
            events.emplace_back ( detail::async ( [kernel, device] { kernel->read_binaries(device); }));
        }
    }

    for (auto & f : events)
        f.get();

    this->construct_core_range_set_for_worker_cores();

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        this->program_device_map = ConstructProgramDeviceMap(device, *this);
        this->buffer = std::make_unique<Buffer>(device, this->program_device_map.program_pages.size() * sizeof(uint32_t),  DeviceCommand::PROGRAM_PAGE_SIZE, BufferType::DRAM);
    }

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(*this, enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(*this, device);
    }
    compile_needed_[device->id()] = false;
}

Program::~Program() {
}
}  // namespace tt::tt_metal
