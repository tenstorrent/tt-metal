#include "tt_metal/impl/program.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt::tt_metal {

auto Program::semaphores_on_core(const CoreCoord &core) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for ( const Semaphore & s : this->semaphores_) {
        if (s.initialized_on_logical_core(core)) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

std::atomic<u64> Program::program_counter = 0;

Program::Program(): id(program_counter++),worker_crs_({}) {}

std::vector<ComputeKernel *> Program::compute_kernels() const {
    std::vector<ComputeKernel *> compute_kernels;
    for (auto kernel : kernels_) {
        if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
            compute_kernels.push_back(compute_kernel);
        }
    }
    return compute_kernels;
}

std::vector<DataMovementKernel *> Program::data_movement_kernels() const {
    std::vector<DataMovementKernel *> data_movement_kernels;
    for (auto kernel : kernels_) {
        if (auto data_movement_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
            data_movement_kernels.push_back(data_movement_kernel);
        }
    }
    return data_movement_kernels;
}

void populate_kernel_group(KernelGroup &kernel_group, Kernel *kernel) {
    if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
        kernel_group.compute = compute_kernel;
    } else if (auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
        if (dm_kernel->data_movement_processor() == DataMovementProcessor::RISCV_0) {
            kernel_group.riscv_0 = dm_kernel;
        } else {
            kernel_group.riscv_1 = dm_kernel;
        }
    }
}

KernelGroup Program::kernels_on_core(const CoreCoord &core) const {
    KernelGroup kernel_group;
    for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
        if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
            populate_kernel_group(kernel_group, kernel);
        }
    }
    return kernel_group;
}

std::map<CoreCoord, KernelGroup> Program::core_to_kernel_group() const {
    std::map<CoreCoord, KernelGroup> core_to_kernel_group;

    for (auto kernel : kernels_) {
        for (auto core : kernel->logical_cores()) {
            KernelGroup &kernel_group = core_to_kernel_group[core];
            populate_kernel_group(kernel_group, kernel);
        }
    }

    return core_to_kernel_group;
}

std::vector<std::string> Program::cores_to_ops() const {
    std::vector<std::string> ops;

    for (const auto &core : this->logical_cores()) {
        for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
            if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
                ops.push_back(kernel->name());
            }
        }
    }
    return ops;
}

void Program::CircularBufferConfig::add_index(u32 index) {
    log_assert(not this->indices[index], "Cannot add circular buffer at index {}, another circular buffer already exists", index);
    this->indices[index] = 1;
}

// CBs on a core are sequential so the next available address for a local buffer is the end of the last
u64 Program::CircularBufferConfig::get_address_candidate() const {
    return this->l1_regions.back().second;
}

void Program::CircularBufferConfig::mark_address(u64 address, u64 size) {
    auto &last_region = this->l1_regions.back();
    log_assert(address >= last_region.second, "Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address", address, last_region.first, last_region.second);
    if (address == last_region.second) {
        last_region.second += size;
    } else {
        this->l1_regions.push_back({address, address + size});
    }
}

const CircularBuffer &Program::add_circular_buffer(const CoreRangeSet &core_range_set, const std::set<u32> &indices, u32 num_tiles, u32 size_bytes, const DataFormat &data_format, std::optional<u32> address) {
    std::optional<u64> computed_addr = std::nullopt;
    std::vector<std::reference_wrapper<CircularBufferConfig>> cb_configs;
    for (const auto &core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord logical_core(x, y);
                auto &cb_config = this->per_core_cb_config_[logical_core];

                for (auto buffer_index : indices) {
                    cb_config.add_index(buffer_index);
                }

                auto candidate_addr = cb_config.get_address_candidate();
                if (not computed_addr.has_value()) {
                    computed_addr = candidate_addr;
                } else {
                    computed_addr = std::max(computed_addr.value(), candidate_addr);
                }

                cb_configs.push_back(cb_config);
            }
        }
    }

    if (address.has_value()) {
        log_assert(address.value() >= computed_addr.value(), "Specified address {} should be at max local buffer region for core range set, try {} instead", address.value(), computed_addr.value());
        computed_addr = address;
    }

    for (auto &cb_config : cb_configs) {
        cb_config.get().mark_address(computed_addr.value(), size_bytes);
    }

    this->circular_buffers_.emplace_back(CircularBuffer(core_range_set, indices, num_tiles, size_bytes, computed_addr.value(), data_format));
    return this->circular_buffers_.back();
}

const std::vector<CircularBuffer> Program::circular_buffers_on_core(const CoreCoord &core) const {
    std::vector<CircularBuffer> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer.is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

void Program::validate_circular_buffer_region(const Device *device, const CoreCoord &logical_core) const {
    auto highest_cb_l1_region = [&]() {
        if (this->per_core_cb_config_.find(logical_core) == this->per_core_cb_config_.end()) {
            return std::make_pair((u64)UNRESERVED_BASE, (u64)UNRESERVED_BASE);
        }
        return this->per_core_cb_config_.at(logical_core).l1_regions.back();
    };

    const auto &cb_space = highest_cb_l1_region();

    log_assert(cb_space.second <= device->l1_size(), "Local buffers on core {} grow to {} KB which is beyond max L1 size of {} KB", logical_core.str(), cb_space.second/1024, device->l1_size()/1024);

    auto bank_ids = device->bank_ids_from_logical_core(logical_core);
    log_assert(bank_ids.size() == 1, "Expected one bank on core that holds local and L1 buffers");

    auto lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids.at(0));
    if (lowest_address.has_value()) {
        log_assert(
            lowest_address.value() >= cb_space.second,
            "Circular buffers in program {} clash with L1 buffers on core {}. L1 buffer allocated at {} and local buffers end at {}", this->id, logical_core.str(), lowest_address.value(), cb_space.second
        );
    }
}

void Program::validate_circular_buffer_region(const Device *device) const {
    for (const auto &[logical_core, cb_config] : this->per_core_cb_config_) {
        this->validate_circular_buffer_region(device, logical_core);
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
        llrt::write_hex_vec_to_core(device.cluster(), device.pcie_slot(), device.worker_core_from_logical_core(logical_core), {semaphore.get().initial_value()}, semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value) {
    semaphores_.emplace_back(Semaphore( crs, address, init_value));
}

std::vector<CoreCoord> Program::logical_cores() const {
    std::vector<CoreCoord> cores_in_program;
    std::set<CoreCoord> unique_cores;
    for (auto kernel : kernels_) {
        for (auto core : kernel->logical_cores()) {
            if (unique_cores.find(core) != unique_cores.end()) {
                continue;
            }
            unique_cores.insert(core);
            cores_in_program.push_back(core);
        }
    }
    return cores_in_program;
}

void Program::construct_core_range_set_for_worker_cores() {
    for (auto kernel : kernels_ )
    {
        this->worker_crs_.merge ( kernel->core_range_set());
    }
}

Program::~Program() {
    for (auto kernel : kernels_) {
        delete kernel;
    }
}

}  // namespace tt::tt_metal
