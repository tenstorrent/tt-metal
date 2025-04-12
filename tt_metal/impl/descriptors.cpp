// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/descriptors.hpp"

#include "tt_stl/overloaded.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
size_t hash_combine(size_t seed, size_t value) { return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2)); }
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

size_t ProgramDescriptor::calculate_program_hash() const {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    auto hash_kernel = [&](const KernelDescriptor& kernel) -> size_t {
        size_t hash = std::hash<std::string>()(kernel.kernel_source);
        hash = hash_combine(hash, static_cast<size_t>(kernel.source_type));

        hash = hash_combine(hash, kernel.core_ranges.size());
        for (const auto& core_range : kernel.core_ranges) {
            hash = hash_combine(hash, std::hash<CoreRange>()(core_range));
        }

        hash = hash_combine(hash, kernel.compile_time_args.size());
        for (const auto& compile_time_arg : kernel.compile_time_args) {
            hash = hash_combine(hash, compile_time_arg);
        }

        hash = hash_combine(hash, kernel.defines.size());
        for (const auto& [key, value] : kernel.defines) {
            hash = hash_combine(hash, std::hash<std::string>()(key));
            hash = hash_combine(hash, std::hash<std::string>()(value));
        }

        hash = hash_combine(hash, kernel.common_runtime_args.size());

        hash = hash_combine(hash, kernel.runtime_args.size());
        for (const auto& runtime_args_row : kernel.runtime_args) {
            hash = hash_combine(hash, runtime_args_row.size());
            for (const auto& core_runtime_args : runtime_args_row) {
                hash = hash_combine(hash, core_runtime_args.size());
            }
        }

        size_t hash_config = std::visit(
            tt::stl::overloaded{
                [&](const ReaderConfigDescriptor& reader_config) -> size_t { return 0; },
                [&](const WriterConfigDescriptor& writer_config) -> size_t { return 0; },
                [&](const DataMovementConfigDescriptor& data_movement_config) -> size_t {
                    size_t hash = static_cast<size_t>(data_movement_config.processor);
                    hash = hash_combine(hash, static_cast<size_t>(data_movement_config.noc));
                    hash = hash_combine(hash, static_cast<size_t>(data_movement_config.noc_mode));
                    return hash;
                },
                [&](const ComputeConfigDescriptor& compute_config) -> size_t {
                    size_t hash = static_cast<size_t>(compute_config.math_fidelity);
                    hash = hash_combine(hash, compute_config.fp32_dest_acc_en);
                    hash = hash_combine(hash, compute_config.dst_full_sync_en);
                    hash = hash_combine(hash, compute_config.bfp8_pack_precise);
                    hash = hash_combine(hash, compute_config.math_approx_mode);
                    hash = hash_combine(hash, compute_config.unpack_to_dest_mode.size());
                    for (auto unpack_to_dest_mode : compute_config.unpack_to_dest_mode) {
                        hash = hash_combine(hash, static_cast<size_t>(unpack_to_dest_mode));
                    }
                    return hash;
                },
                [&](const EthernetConfigDescriptor& ethernet_config) -> size_t {
                    size_t hash = static_cast<size_t>(ethernet_config.eth_mode);
                    hash = hash_combine(hash, static_cast<size_t>(ethernet_config.noc));
                    hash = hash_combine(hash, static_cast<size_t>(ethernet_config.processor));
                    return hash;
                }},
            kernel.config);
        hash = hash_combine(hash, kernel.config.index());
        hash = hash_combine(hash, hash_config);
        return hash;
    };

    auto hash_cb_format_descriptor = [&](const CBFormatDescriptor& format_descriptor) -> size_t {
        size_t hash = format_descriptor.buffer_index;
        hash = hash_combine(hash, static_cast<size_t>(format_descriptor.data_format));
        hash = hash_combine(hash, format_descriptor.page_size);
        if (format_descriptor.tile) {
            hash = hash_combine(hash, format_descriptor.tile->height);
            hash = hash_combine(hash, format_descriptor.tile->width);
            hash = hash_combine(hash, format_descriptor.tile->transpose);
        } else {
            hash = hash_combine(hash, 0);
            hash = hash_combine(hash, 0);
            hash = hash_combine(hash, 0);
        }
        return hash;
    };

    auto hash_circular_buffer = [&](const CBDescriptor& cb) -> size_t {
        size_t hash = cb.core_ranges.size();
        for (const auto& core_range : cb.core_ranges) {
            hash = hash_combine(hash, std::hash<CoreRange>()(core_range));
        }
        hash = hash_combine(hash, cb.format_descriptors.size());
        for (const auto& format_descriptor : cb.format_descriptors) {
            hash = hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        hash = hash_combine(hash, cb.remote_format_descriptors.size());
        for (const auto& format_descriptor : cb.remote_format_descriptors) {
            hash = hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        hash = hash_combine(hash, cb.buffer != nullptr);
        hash = hash_combine(hash, cb.global_circular_buffer != nullptr);
        return hash;
    };

    auto hash_semaphore = [&](const SemaphoreDescriptor& semaphore) -> size_t {
        size_t hash = semaphore.core_ranges.size();
        for (const auto& core_range : semaphore.core_ranges) {
            hash = hash_combine(hash, std::hash<CoreRange>()(core_range));
        }
        hash = hash_combine(hash, static_cast<size_t>(semaphore.core_type));
        hash = hash_combine(hash, semaphore.initial_value);
        return hash;
    };

    size_t hash = 0;
    for (const auto& kernel : kernels) {
        hash = hash_combine(hash, hash_kernel(kernel));
    }
    for (const auto& cb : cbs) {
        hash = hash_combine(hash, hash_circular_buffer(cb));
    }
    for (const auto& semaphore : semaphores) {
        hash = hash_combine(hash, hash_semaphore(semaphore));
    }
    return hash;
}

void KernelDescriptor::reserve_runtime_args() {
    size_t max_x = 0;
    size_t max_y = 0;
    for (const auto& core_range : core_ranges) {
        max_x = std::max(max_x, core_range.end_coord.x + 1);
        max_y = std::max(max_y, core_range.end_coord.y + 1);
    }
    runtime_args.resize(max_x);
    for (auto& runtime_args_row : runtime_args) {
        runtime_args_row.resize(max_y);
    }
}

size_t MeshWorkloadDescriptor::calculate_program_hash() const {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    size_t hash = 0;
    for (const auto& [coord_range, program] : programs) {
        hash = hash_combine(hash, std::hash<distributed::MeshCoordinateRange>()(coord_range));
        hash = hash_combine(hash, program.calculate_program_hash());
    }
    return hash;
}
}  // namespace tt::tt_metal
