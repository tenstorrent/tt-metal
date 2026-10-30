// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_builder.hpp"

#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal::detail {

namespace {

std::bitset<MAX_PROCESSOR_TYPES_COUNT> get_kernel_processor_set(const Kernel& kernel) {
    std::bitset<MAX_PROCESSOR_TYPES_COUNT> set;
    for (int i = 0; i < kernel.expected_num_binaries(); i++) {
        int processor_id = kernel.get_kernel_processor_type(i);
        TT_ASSERT(0 <= processor_id && processor_id < MAX_PROCESSOR_TYPES_COUNT);
        set.set(processor_id);
    }
    return set;
}

}  // namespace

Metal1ProgramBuilder::Metal1ProgramBuilder(uint32_t programmable_core_count, uint32_t max_circular_buffers) :
    max_circular_buffers_(max_circular_buffers), kernels_(programmable_core_count) {}

KernelHandle Metal1ProgramBuilder::add_kernel(
    const std::shared_ptr<Kernel>& kernel, uint32_t programmable_core_type_index) {
    // Id is unique across all kernels on all core types
    KernelHandle id = num_kernels();

    auto new_kernel_core_type = kernel->get_kernel_programmable_core_type();
    auto new_kernel_processor_class = kernel->get_kernel_processor_class();

    std::set<CoreCoord> kernel_logical_cores = kernel->logical_cores();
    auto new_kernel_processor_set = get_kernel_processor_set(*kernel);
    for (size_t i = 0; i < num_kernels(); i++) {
        // Note, looks like id is program specific, and increments naturally as kernels are added.
        //  add_kernel -> id = num_kernels -> kernel is inserted -> next num_kernels() increments.
        std::shared_ptr<Kernel> check_kernel = get_kernel(i);
        auto check_kernel_core_type = check_kernel->get_kernel_programmable_core_type();
        auto check_kernel_processor_class = check_kernel->get_kernel_processor_class();
        if (check_kernel_core_type == new_kernel_core_type &&
            check_kernel_processor_class == new_kernel_processor_class &&
            (new_kernel_processor_set & get_kernel_processor_set(*check_kernel)).any()) {
            // Two kernels are using the same processor, need to check core ranges.
            std::set<CoreCoord> check_kernel_logical_cores = check_kernel->logical_cores();
            for (CoreCoord coreCoord : kernel_logical_cores) {
                TT_FATAL(
                    !check_kernel_logical_cores.contains(coreCoord),
                    "Core Overlap Between (\"{}\") and new kernel (\"{}\") at {}",
                    check_kernel->name(),
                    kernel->name(),
                    coreCoord.str());
            }
        }
    }

    kernels_[programmable_core_type_index].insert({id, kernel});
    return id;
}

std::shared_ptr<Kernel> Metal1ProgramBuilder::get_kernel(KernelHandle kernel_id) const {
    // find coretype based on kernel_id
    for (const auto& kernels : kernels_) {
        if (kernels.contains(kernel_id)) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

size_t Metal1ProgramBuilder::num_kernels() const {
    size_t count = 0;
    for (const auto& kernels : kernels_) {
        count += kernels.size();
    }
    return count;
}

void Metal1ProgramBuilder::validate_circular_buffer_indices(const CircularBufferImpl& circular_buffer) const {
    // The deserialization constructor sets these sets independently, so local and remote are not
    // guaranteed to be subsets of buffer_indices; all three index the per-core bitsets.
    const CircularBufferConfig& config = circular_buffer.config();
    for (const auto* indices :
         {&config.buffer_indices(), &config.local_buffer_indices(), &config.remote_buffer_indices()}) {
        for (uint32_t buffer_index : *indices) {
            if (buffer_index >= max_circular_buffers_) {
                TT_THROW(
                    "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                    buffer_index,
                    max_circular_buffers_);
            }
        }
    }
}

CBHandle Metal1ProgramBuilder::add_circular_buffer(const std::shared_ptr<CircularBufferImpl>& circular_buffer) {
    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS>& cb_indices = per_core_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& local_cb_indices = per_core_local_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& remote_cb_indices = per_core_remote_cb_indices_[logical_core];
                auto add_buffer_indices = [&cb_indices](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
                        if (cb_indices[buffer_index]) {
                            TT_THROW(
                                "Invalid circular buffer index: Cannot add circular buffer at index {}, another "
                                "circular "
                                "buffer already exists",
                                buffer_index);
                        }
                        cb_indices[buffer_index] = true;
                        target_cb_indices[buffer_index] = true;
                    }
                };
                add_buffer_indices(circular_buffer->config().local_buffer_indices(), local_cb_indices);
                add_buffer_indices(circular_buffer->config().remote_buffer_indices(), remote_cb_indices);
            }
        }
    }

    circular_buffers_.push_back(circular_buffer);
    circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

void Metal1ProgramBuilder::validate_semaphore_id(
    const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const {
    TT_FATAL(semaphore_id < NUM_SEMAPHORES, "Semaphore id {} exceeds max value {}", semaphore_id, NUM_SEMAPHORES - 1);

    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing_semaphores = semaphores_on_core(logical_core, core_type);
                for (const auto& semaphore : existing_semaphores) {
                    TT_FATAL(
                        semaphore.get().id() != semaphore_id,
                        "Semaphore id {} already in use on core {}",
                        semaphore_id,
                        logical_core.str());
                }
            }
        }
    }
}

void Metal1ProgramBuilder::add_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, uint32_t l1_alignment, CoreType core_type) {
    validate_semaphore_id(crs, semaphore_id, core_type);
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, l1_alignment, core_type));
}

uint32_t Metal1ProgramBuilder::allocate_semaphore_id(const CoreRangeSet& crs, CoreType core_type) const {
    std::optional<uint32_t> semaphore_id;
    std::bitset<NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing = semaphores_on_core(logical_core, core_type);
                if (existing.size() == NUM_SEMAPHORES) {
                    TT_THROW(
                        "Cannot add semaphore on core {}. Max number of semaphores ({}) reached!",
                        logical_core.str(),
                        NUM_SEMAPHORES);
                }
                for (const auto& semaphore : existing) {
                    used_semaphore_ids.set(semaphore.get().id());
                }
            }
        }
    }
    for (uint32_t sem_id = 0; sem_id < NUM_SEMAPHORES; sem_id++) {
        if (!used_semaphore_ids.test(sem_id)) {
            semaphore_id = sem_id;
            break;
        }
    }
    TT_FATAL(
        semaphore_id.has_value(),
        "Unable to initialize semaphore on CoreRangeSet {}: all {} IDs are in use",
        crs.str(),
        NUM_SEMAPHORES);
    return *semaphore_id;
}

std::vector<std::reference_wrapper<const Semaphore>> Metal1ProgramBuilder::semaphores_on_core(
    const CoreCoord& core, CoreType core_type) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    semaphores.reserve(semaphores_.size());
    for (const Semaphore& s : semaphores_) {
        if (s.initialized_on_logical_core(core) && s.core_type() == core_type) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

}  // namespace tt::tt_metal::detail
