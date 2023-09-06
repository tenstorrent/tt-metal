// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

void Program::add_kernel(Kernel *kernel) {
    kernel_ids_.push_back(kernel->id());
    kernel_by_id_[kernel->id()] = kernel;
}

Kernel *Program::get_kernel(KernelID kernel_id) const {
    TT_ASSERT(this->kernel_by_id_.find(kernel_id) != this->kernel_by_id_.end(), "Expected Kernel with ID {} to be in Program {}", kernel_id, this->id);
    return this->kernel_by_id_.at(kernel_id);
}

void populate_kernel_group(KernelGroup &kernel_group, Kernel *kernel) {
    RISCV riscv_processor = kernel->processor();
    switch (riscv_processor) {
        case RISCV::BRISC: kernel_group.riscv0_id = kernel->id(); break;
        case RISCV::NCRISC: kernel_group.riscv1_id = kernel->id(); break;
        case RISCV::COMPUTE: kernel_group.compute_id = kernel->id(); break;
        default:
            TT_ASSERT(false, "Unsupported kernel processor!");
    }
}

KernelGroup Program::kernels_on_core(const CoreCoord &core) const {
    KernelGroup kernel_group;
    for (auto &[kernel_id, kernel] : this->kernel_by_id_) {
        auto cores = kernel->logical_cores();
        if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
            populate_kernel_group(kernel_group, kernel);
        }
    }
    return kernel_group;
}

std::map<CoreCoord, KernelGroup> Program::core_to_kernel_group() const {
    std::map<CoreCoord, KernelGroup> core_to_kernel_group;

    for (auto &[kernel_id, kernel] : this->kernel_by_id_) {
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
        for (auto kernel_id : this->kernel_ids_) {
        auto kernel = this->get_kernel(kernel_id);
        auto cores = kernel->logical_cores();
            if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
                ops.push_back(kernel->name());
            }
        }
    }
    return ops;
}

void Program::CircularBufferConfig::add_index(u32 index) {
    log_assert(0 <= index < NUM_CIRCULAR_BUFFERS, "Invalid circular buffer index: {} should be between 0 and {}", 0, NUM_CIRCULAR_BUFFERS);
    log_assert(not (this->indices.to_ulong() & (1 << index)), "Invalid circular buffer index: Cannot add circular buffer at index {}, another circular buffer already exists", index);
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
    log_assert(
        indices.size() <= NUM_CIRCULAR_BUFFERS,
        "Invalid number of circular buffers: Requested number of circular buffers ({}) exceeds max number of circular buffers per core ({})", indices.size(), NUM_CIRCULAR_BUFFERS
    );
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

void Program::validate_circular_buffer_region(const Device *device, std::optional<CoreCoord> logical_core) const {
    auto highest_cb_l1_region = [&](const CoreCoord &core) {
        if (this->per_core_cb_config_.find(core) == this->per_core_cb_config_.end()) {
            return std::make_pair((u64)UNRESERVED_BASE, (u64)UNRESERVED_BASE);
        }
        return this->per_core_cb_config_.at(core).l1_regions.back();
    };

    auto validate_cb_space_and_l1_buffer_space_disjoint = [&](const CoreCoord &core, const std::pair<u64, u64> &cb_space) {
        if (cb_space.second > device->l1_size()) {
            log_assert(cb_space.second <= device->l1_size(), "Local buffers on core {} grow to {} KB which is beyond max L1 size of {} KB", core.str(), cb_space.second/1024, device->l1_size()/1024);
        }

        auto bank_ids = device->bank_ids_from_logical_core(core);
        if (bank_ids.size() != 1) {
            log_assert(bank_ids.size() == 1, "Expected one bank on core that holds local and L1 buffers");
        }

        auto lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids.at(0));
        if (lowest_address.has_value()) {
            if (lowest_address.value() < cb_space.second) {
                log_assert(lowest_address.value() >= cb_space.second, "Circular buffers in program {} clash with L1 buffers on core {}. L1 buffer allocated at {} and local buffers end at {}", this->id, core.str(), lowest_address.value(), cb_space.second);
            }
        }
    };

    if (logical_core.has_value()) {
        const auto &cb_space = highest_cb_l1_region(logical_core.value());
        validate_cb_space_and_l1_buffer_space_disjoint(logical_core.value(), cb_space);
    } else {
        for (const auto &[core, cb_config] : this->per_core_cb_config_) {
            const auto &cb_space = highest_cb_l1_region(core);
            validate_cb_space_and_l1_buffer_space_disjoint(core, cb_space);
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
        llrt::write_hex_vec_to_core(device.cluster(), device.pcie_slot(), device.worker_core_from_logical_core(logical_core), {semaphore.get().initial_value()}, semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value) {
    semaphores_.emplace_back(Semaphore( crs, address, init_value));
}

std::vector<CoreCoord> Program::logical_cores() const {
    std::vector<CoreCoord> cores_in_program;
    std::set<CoreCoord> unique_cores;
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
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
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
        this->worker_crs_.merge ( kernel->core_range_set());
    }
}

Program::~Program() {
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
        delete kernel;
    }
}

}  // namespace tt::tt_metal
