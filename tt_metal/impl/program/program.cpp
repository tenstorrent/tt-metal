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

    void GenerateBinaries(Device *device, JitBuildOptions& build_options, Kernel *kernel) {
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
        Kernel *kernel, JitBuildOptions &build_options, const chip_id_t &device_id) {
        // Account for device id in hash because generated headers are dependent on harvesting config, which can differ per device
        // This can be removed with https://github.com/tenstorrent-metal/tt-metal/issues/3381

        string compile_hash_str = fmt::format(
            "{}_{}_{}",
            device_id,
            std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc)),
            kernel->compute_hash()
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
auto Program::semaphores_on_core(const CoreCoord &core) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for ( const Semaphore & s : this->semaphores_) {
        if (s.initialized_on_logical_core(core)) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

std::atomic<uint64_t> Program::program_counter = 0;

Program::Program(): id(program_counter++),worker_crs_({}), local_circular_buffer_allocation_needed_(false) {}

KernelHandle Program::add_kernel(Kernel *kernel) {
    this->invalidate_compile();
    KernelHandle id = kernels_.size();
    kernels_.push_back(kernel);
    kernel_groups_.resize(0);
    core_to_kernel_group_index_table_.clear();
    return id;
}

Kernel *Program::get_kernel(KernelHandle kernel_id) const {
    //TT_ASSERT(kernel_id < this->kernels_.size(), "Expected Kernel with ID {} to be in Program {}", kernel_id, this->id);
    return this->kernels_.at(kernel_id);
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
        const Kernel *kernel = program.get_kernel(ncrisc_id.value());
        // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
        // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
        this->launch_msg.enable_ncrisc = true;
        this->launch_msg.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
        this->launch_msg.ncrisc_kernel_size16 = kernel->get_binary_size16();
        this->launch_msg.ncrisc_watcher_kernel_id = program.get_kernel(ncrisc_id.value())->get_watcher_kernel_id();
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
    } else {
        this->launch_msg.enable_erisc = false;
    }

    this->launch_msg.max_cb_index = last_cb_index + 1;
    this->launch_msg.run = RUN_MSG_GO;
}

std::vector<KernelGroup>& Program::get_kernel_groups() {
    update_kernel_groups();
    return kernel_groups_;
}

KernelGroup * Program::kernels_on_core(const CoreCoord &core) {
    update_kernel_groups();
    if (core.x >= grid_extent_.x || core.y >= grid_extent_.y) return nullptr;
    uint8_t index = core_to_kernel_group_index_table_.at(core.y * grid_extent_.x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : &kernel_groups_.at(index);
}

struct KernelGroupInt {
    bool valid;
    std::optional<KernelHandle> trisc_id = std::nullopt;
    std::optional<KernelHandle> brisc_id = std::nullopt;
    std::optional<KernelHandle> ncrisc_id = std::nullopt;
    std::optional<KernelHandle> erisc_id = std::nullopt;

    bool operator==(const KernelGroupInt& b) const;
    void update(Kernel* kernel, size_t kernel_idx) {
        RISCV riscv_processor = kernel->processor();
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

void Program::update_kernel_groups() {
    if (core_to_kernel_group_index_table_.size() == 0) {
        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(),
                          std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_ = {0, 0};
        for (Kernel * kernel : kernels_) {
            for (auto core : kernel->logical_cores()) {
                if (core.x > grid_extent_.x) grid_extent_.x = core.x;
                if (core.y > grid_extent_.y) grid_extent_.y = core.y;
                if (core.x < base.x) base.x = core.x;
                if (core.y < base.y) base.y = core.y;
            }
        }
        grid_extent_.x++;
        grid_extent_.y++;

        // grid maps cores to sets-of-kernels running on that core
        std::vector<KernelGroupInt> grid;
        grid.resize(grid_extent_.x * grid_extent_.y);
        for (size_t kidx = 0; kidx < this->num_kernels(); kidx++) {
            Kernel * kernel = kernels_[kidx];
            for (auto core : kernel->logical_cores()) {
                int core_index = core.y * grid_extent_.x + core.x;
                grid[core_index].valid = true;
                grid[core_index].update(kernel, kidx);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::unordered_map<KernelGroupInt, std::set<CoreRange>, KernelGroupIntHasher> map;
        for (auto y = base.y; y < grid_extent_.y; y++) {
            for (auto x = base.x; x < grid_extent_.x; x++) {
                int index = y * grid_extent_.x + x;
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
        core_to_kernel_group_index_table_.resize(grid_extent_.x * grid_extent_.y, core_to_kernel_group_invalid_index);
        for (auto& kg_to_cores : map) {

            int last_cb_index = -1;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : kg_to_cores.second) {
                for (auto y = range.start.y; y <= range.end.y; y++) {
                    for (auto x = range.start.x; x <= range.end.x; x++) {
                        core_to_kernel_group_index_table_[y * grid_extent_.x + x] = index;

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

            kernel_groups_.push_back(KernelGroup(
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

std::vector<std::string> Program::cores_to_ops() const {
    std::vector<std::string> ops;
    for (const auto &[core_type, cores_of_type] : this->logical_cores()) {
        for (const auto &core : cores_of_type) {
            for (Kernel * kernel : kernels_) {
                if ( kernel->get_kernel_core_type() !=  core_type ) continue;
                auto cores = kernel->logical_cores();
                if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
                    ops.push_back(kernel->name());
                }
            }
        }
    }
    return ops;
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
    const std::vector<uint32_t> &bank_ids = device->bank_ids_from_logical_core(*device->compute_cores.begin());
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

uint32_t Program::semaphore_address ( uint32_t sem_idx ) const {
    return semaphores_.at(sem_idx).address();
}

void Program::init_semaphores( const Device & device, const CoreCoord &logical_core ) const{
    auto semaphores_on_core = this->semaphores_on_core(logical_core);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(device.id(), device.worker_core_from_logical_core(logical_core), {semaphore.get().initial_value()}, semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value) {
    this->invalidate_compile();
    semaphores_.emplace_back(Semaphore( crs, address, init_value));
}

std::unordered_map<CoreType, std::vector<CoreCoord>> Program::logical_cores() const {
    std::unordered_map<CoreType, std::vector<CoreCoord>> cores_in_program;
    std::unordered_map<CoreType, std::set<CoreCoord>> unique_cores;
    for (Kernel * kernel : kernels_){
        const auto &core_type = kernel->get_kernel_core_type();
        if (cores_in_program.find(core_type) == cores_in_program.end()) {
            cores_in_program.insert({core_type, {}});
        }
        if (unique_cores.find(core_type) == unique_cores.end()) {
            unique_cores.insert({core_type, {}});
        }
        for (auto core : kernel->logical_cores()) {
            if (unique_cores.at(core_type).find(core) != unique_cores.at(core_type).end()) {
                continue;
            }
            unique_cores.at(core_type).insert(core);
            cores_in_program.at(core_type).push_back(core);
        }
    }
    return cores_in_program;
}

void Program::construct_core_range_set_for_worker_cores() {
    bool found_kernels = false;
    for (Kernel * kernel : kernels_){
        this->worker_crs_ = this->worker_crs_.merge ( kernel->core_range_set() );
        found_kernels = true;
    }
    TT_ASSERT(!found_kernels || this->worker_crs_.ranges().size() >= 1, "Invalid core range set");
}

void Program::set_cb_data_fmt(
    Device *device, Kernel *kernel, JitBuildOptions &build_options) const {
    ZoneScoped;
    for (auto logical_cr : kernel->logical_coreranges()) {
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

void Program::compile( Device * device )
{
    bool first_compile_on_device = compile_needed_.find(device->id()) == compile_needed_.end();
    if (not first_compile_on_device and (not compile_needed_.at(device->id()))) {
        return;
    }

    TT_FATAL(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        this->get_id());

    detail::ProfileTTMetalScope profile_this =
        detail::ProfileTTMetalScope(std::string("CompileProgram ") + std::to_string(device->id()));
    bool profile_kernel = getDeviceProfilerState();
    std::vector<std::future<void>> events;
    DprintServerSetProfilerState(profile_kernel);

    // compile all kernels in parallel
    for (Kernel * kernel : kernels_) {
        events.emplace_back ( detail::async ( [kernel, device, this] {
            ZoneScoped;

            JitBuildOptions build_options(device->build_env());
            kernel->set_build_options(build_options);
            this->set_cb_data_fmt(device, kernel, build_options);

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

    for (auto & f : events)
        f.wait();

    for (Kernel * kernel : kernels_) {
        events.emplace_back ( detail::async ( [kernel, device] { kernel->read_binaries(device); }));
    }

    for (auto & f : events)
        f.wait();

    this->construct_core_range_set_for_worker_cores();

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(*this, enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(*this, device);
    }
    compile_needed_[device->id()] = false;
}

Program::~Program() {
    for (Kernel * kernel : kernels_) {
        delete kernel;
    }
}
}  // namespace tt::tt_metal
