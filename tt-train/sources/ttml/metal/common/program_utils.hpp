// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <string>
#include <tt-metalium/program_descriptors.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

inline uint32_t get_block_size(uint32_t num_inner, const uint32_t max_block_size = 4U) {
    for (uint32_t block_size = max_block_size; block_size > 1U; block_size--) {
        if (num_inner % block_size == 0) {  // if num_inner is divisible by block_size - choose this block_size
            return block_size;
        }
    }
    return 1U;
}

inline uint32_t pack_two_bfloat16_to_uint32(float value) {
    uint32_t uint32_data = std::bit_cast<uint32_t>(value);
    uint32_t casted_uint16_data = uint32_data >> 16U;
    return casted_uint16_data | (casted_uint16_data << 16U);
}

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

/**
 *   Byte-sized variant of create_circular_buffer for CBs whose total size and
 *   page size aren't naturally expressed as tile_size × num_tiles (control/scratch
 *   buffers, row-major staging, custom-aligned pages). If `page_size_bytes` is 0
 *   the whole buffer is one page.
 */
inline tt::tt_metal::CBHandle create_circular_buffer_bytes(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_bytes,
    uint32_t page_size_bytes = 0U) {
    const uint32_t page_size = page_size_bytes == 0U ? total_bytes : page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_bytes, {{cb_index, data_format}}).set_page_size(cb_index, page_size);
    return CreateCircularBuffer(program, core_ranges, cb_config);
}

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

/**
 * Create a compute kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path,
    const bool fp32_dest_acc_en) {
    return tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compile_time_args,
            .defines = defines});
}

namespace ttml::metal {

// ---------------------------------------------------------------------------------------------------------------------
// ProgramDescriptor factories.
//
// A descriptor program factory has a static create_descriptor() returning a tt::tt_metal::ProgramDescriptor; the
// framework builds the Program from it on a cache miss. On a cache hit it does one of two things:
//  - If the factory has no override_runtime_arguments(), it re-patches every buffer passed as a Buffer* to
//    KernelDescriptor::emplace_runtime_args(). That is only correct when buffer addresses are the only per-call state:
//    every other runtime arg must be a pure function of the program hash, and the hash must key which optional
//    tensors are present (bindings are positions in the list of present tensors).
//  - If the factory defines `static void override_runtime_arguments(tt::tt_metal::Program&, const
//    operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&, const
//    std::optional<ttnn::MeshCoordinate>&)`, only that runs: it must re-apply every buffer address and every
//    hash-excluded scalar on every core itself. Use it for ops with hash-excluded per-call values (lr, seeds, ...) or
//    that are called in place through tensor_args. TTML_ASSERT_DESCRIPTOR_OVERRIDE checks that the adapter really sees
//    the hook; a near-miss signature compiles and is silently never called.
//
// Kernel handles in the built Program are the push order of ProgramDescriptor::kernels. Factories with an override push
// fixed-index kernels first (reader, writer, ...) and the optional per-group compute kernel last, and name the indices.
// The helpers below mirror the Program-based create_* helpers above, with the same parameter order minus the Program.
// ---------------------------------------------------------------------------------------------------------------------

// CBs get merged core ranges, as CreateCircularBuffer gives them (Program::add_circular_buffer merges; building from a
// CBDescriptor does not), so a CB covers the same rectangles as in a Program-built factory.
inline tt::tt_metal::CBDescriptor make_cb_descriptor(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    return tt::tt_metal::CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_ranges.merge_ranges(),
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    };
}

// Descriptor counterpart of create_circular_buffer_bytes: a page size of 0 makes the whole buffer one page.
inline tt::tt_metal::CBDescriptor make_cb_descriptor_bytes(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_bytes,
    uint32_t page_size_bytes = 0U) {
    return tt::tt_metal::CBDescriptor{
        .total_size = total_bytes,
        .core_ranges = core_ranges.merge_ranges(),
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = page_size_bytes == 0U ? total_bytes : page_size_bytes,
        }}},
    };
}

inline tt::tt_metal::KernelDescriptor make_kernel_descriptor(
    const std::string& kernel_path,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = kernel_path;
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = core_ranges;
    kernel.compile_time_args = compile_time_args;
    kernel.defines.assign(defines.begin(), defines.end());
    kernel.config = std::move(config);
    return kernel;
}

inline tt::tt_metal::KernelDescriptor make_reader_kernel_descriptor(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return make_kernel_descriptor(
        kernel_path, core_ranges, compile_time_args, defines, tt::tt_metal::ReaderConfigDescriptor{});
}

inline tt::tt_metal::KernelDescriptor make_writer_kernel_descriptor(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return make_kernel_descriptor(
        kernel_path, core_ranges, compile_time_args, defines, tt::tt_metal::WriterConfigDescriptor{});
}

// Same settings as create_compute_kernel (HiFi4, no approximation) unless overridden. unpack_to_dest_mode is indexed
// by CB id, as in tt::tt_metal::ComputeConfig; leave it empty for the default.
inline tt::tt_metal::KernelDescriptor make_compute_kernel_descriptor(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path,
    const bool fp32_dest_acc_en,
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode = {},
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
    const bool math_approx_mode = false) {
    return make_kernel_descriptor(
        kernel_path,
        core_ranges,
        compile_time_args,
        defines,
        tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = math_approx_mode,
        });
}

// Program-local semaphore with an explicit id. Legacy CreateSemaphore hands out the lowest free id, so semaphores
// created on one core range in order get 0, 1, 2, ...; a descriptor must state the id, and the kernels' compile-time
// or runtime args must carry the same named constant.
inline tt::tt_metal::SemaphoreDescriptor make_semaphore_descriptor(
    uint32_t id, const tt::tt_metal::CoreRangeSet& core_ranges, uint32_t initial_value = 0U) {
    return tt::tt_metal::SemaphoreDescriptor{
        .id = id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges,
        .initial_value = initial_value,
    };
}

// Fails the build when a descriptor factory's override_runtime_arguments() does not match the signature the mesh
// adapter probes for (static, Program&, const attrs&, const tensor_args&, tensor_return_value_t&,
// const std::optional<ttnn::MeshCoordinate>&). A mismatch is not otherwise a compile error: the hook is never called
// and every hash-excluded value freezes at the first cache miss.
#define TTML_ASSERT_DESCRIPTOR_OVERRIDE(DeviceOperation, ProgramFactory)                                               \
    static_assert(                                                                                                     \
        ::ttnn::device_operation::MeshDeviceOperationAdapter<DeviceOperation>::template DescriptorMeshWorkloadAdapter< \
            ProgramFactory>::has_override_runtime_arguments(),                                                         \
        #ProgramFactory "::override_runtime_arguments() is not visible to the mesh adapter")

// One core's share of the work, as handed out by for_each_core_with_work.
struct CoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t index;      // position in the walk: core == {index / num_cores_y, index % num_cores_y}
    uint32_t num_units;  // rows, blocks or tiles this core processes
    uint32_t start;      // units handed to the cores before it
    bool in_group_1;     // which split_work_to_cores group the core is in; picks the per-group compute kernel
};

/**
 * Walk the cores that `tt::tt_metal::split_work_to_cores` handed work to, in the order tt-train
 * readers/writers assume (core i -> {i / num_cores_y, i % num_cores_y}), and call `fn(const CoreWork&)`
 * once per core. Ops that need the walk position (per-core seeds, reduction protocols) take it from
 * `CoreWork::index` rather than recomputing the walk; ops with one compute kernel per group pick it with
 * `CoreWork::in_group_1` rather than testing the groups again.
 */
template <typename Fn>
inline void for_each_core_with_work(
    uint32_t num_cores,
    uint32_t num_cores_y,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t num_units_per_core_group_1,
    uint32_t num_units_per_core_group_2,
    Fn&& fn) {
    uint32_t num_units_written = 0U;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const bool in_group_1 = core_group_1.contains(core);
        uint32_t num_units = 0U;
        if (in_group_1) {
            num_units = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units = num_units_per_core_group_2;
        } else {
            TT_FATAL(false, "Core {} is in neither work group", core.str());
        }
        fn(CoreWork{core, i, num_units, num_units_written, in_group_1});
        num_units_written += num_units;
    }
}

/**
 * The same core walk without the work lookup, for override_runtime_arguments where only buffer addresses
 * change and every core keeps the work it was given in create(). `fn` takes `(core)` or `(core, index)`.
 */
template <typename Fn>
inline void for_each_core(uint32_t num_cores, uint32_t num_cores_y, Fn&& fn) {
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core{i / num_cores_y, i % num_cores_y};
        if constexpr (std::is_invocable_v<Fn&, const tt::tt_metal::CoreCoord&, uint32_t>) {
            fn(core, i);
        } else {
            fn(core);
        }
    }
}

/**
 * Arithmetic form of `tt::tt_metal::split_work_to_cores(grid, units)` with the default column-major walk, for
 * override_runtime_arguments() of descriptor factories, which keep no shared variables and must not rerun
 * split_work_to_cores on every cache hit (it builds CoreRangeSets: about +3 us per hit on 64 cores). The first
 * `units % num_cores` cores of the walk get one extra unit (group 1); with no remainder every core is in group 1.
 * create_descriptor() pins it against split_work_to_cores with check_work_split().
 */
struct WorkSplit {
    uint32_t num_cores = 0U;          // cores with work: min(units, grid.x * grid.y)
    uint32_t num_cores_y = 0U;        // grid.y: core i is {i / num_cores_y, i % num_cores_y}
    uint32_t num_group_1_cores = 0U;  // a prefix of the walk; all of it when the split is even
    uint32_t units_per_core_group_1 = 0U;
    uint32_t units_per_core_group_2 = 0U;  // 0 when there is no group 2

    [[nodiscard]] bool has_group_2() const {
        return num_group_1_cores < num_cores;
    }
};

inline WorkSplit compute_work_split(const tt::tt_metal::CoreCoord& grid, uint32_t units) {
    WorkSplit split;
    split.num_cores_y = static_cast<uint32_t>(grid.y);
    if (units == 0U) {
        return split;
    }
    split.num_cores = std::min(units, static_cast<uint32_t>(grid.x * grid.y));
    const uint32_t remainder = units % split.num_cores;
    if (remainder == 0U) {
        split.num_group_1_cores = split.num_cores;
        split.units_per_core_group_1 = units / split.num_cores;
    } else {
        split.num_group_1_cores = remainder;
        split.units_per_core_group_1 = units / split.num_cores + 1U;
        split.units_per_core_group_2 = units / split.num_cores;
    }
    return split;
}

// TT_FATALs unless `split` describes the same cores and units as the split_work_to_cores result passed in.
inline void check_work_split(
    const WorkSplit& split,
    uint32_t num_cores,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t num_units_per_core_group_1,
    uint32_t num_units_per_core_group_2) {
    TT_FATAL(
        split.num_cores == num_cores && split.num_group_1_cores == core_group_1.num_cores() &&
            split.num_cores - split.num_group_1_cores == core_group_2.num_cores() &&
            split.units_per_core_group_1 == num_units_per_core_group_1 &&
            split.units_per_core_group_2 == num_units_per_core_group_2,
        "compute_work_split disagrees with split_work_to_cores: {} cores ({} in group 1, {}/{} units) vs {} cores "
        "({} in group 1, {}/{} units)",
        split.num_cores,
        split.num_group_1_cores,
        split.units_per_core_group_1,
        split.units_per_core_group_2,
        num_cores,
        core_group_1.num_cores(),
        num_units_per_core_group_1,
        num_units_per_core_group_2);
}

/**
 * for_each_core_with_work driven by a WorkSplit: the same walk, units and offsets, without CoreRangeSet lookups. Use
 * one walk in both create_descriptor() and override_runtime_arguments() so the two cannot drift apart.
 */
template <typename Fn>
inline void for_each_core_with_work(const WorkSplit& split, Fn&& fn) {
    uint32_t num_units_written = 0U;
    for (uint32_t i = 0; i < split.num_cores; ++i) {
        const bool in_group_1 = i < split.num_group_1_cores;
        const uint32_t num_units = in_group_1 ? split.units_per_core_group_1 : split.units_per_core_group_2;
        fn(CoreWork{{i / split.num_cores_y, i % split.num_cores_y}, i, num_units, num_units_written, in_group_1});
        num_units_written += num_units;
    }
}

}  // namespace ttml::metal
