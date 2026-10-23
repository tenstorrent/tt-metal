// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "frobenius_normalize_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "frobenius_normalize_device_operation.hpp"
#include "frobenius_normalize_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/"
    "reader_frobenius_normalize.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/"
    "writer_frobenius_normalize.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/compute/"
    "frobenius_normalize_compute.cpp";

constexpr auto kCbInput = tt::CBIndex::c_0;
constexpr auto kCbSqAcc = tt::CBIndex::c_1;

constexpr auto kCbRecv = tt::CBIndex::c_3;
constexpr auto kCbNorm = tt::CBIndex::c_4;
constexpr auto kCbOutput = tt::CBIndex::c_5;
constexpr auto kCbSqPartial = tt::CBIndex::c_7;

// Runtime arg slots that override_runtime_arguments() rewrites on every cache hit.
constexpr uint32_t kReaderInputAddrIdx = 0;

constexpr uint32_t kWriterOutputAddrIdx = 0;

constexpr uint32_t kComputeEpsilonIdx = 0;

// Semaphore ids, in the order legacy CreateSemaphore handed them out; the reader gets them as runtime args.
constexpr uint32_t kReductionSemId = 0U;
constexpr uint32_t kBcastSemId = 1U;

}  // namespace

namespace ttml::metal::ops::frobenius_normalize::device {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(FrobeniusNormalizeDeviceOperation, FrobeniusNormalizeProgramFactory);

namespace {

// The work unit is one tile of the input. Both functions walk the cores with this split; create_descriptor() pins it to
// split_work_to_cores.
WorkSplit compute_frobenius_work_split(const ttnn::Tensor& input) {
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        static_cast<uint32_t>(input.physical_volume() / tt::constants::TILE_HW));
}

// Compute runtime arg kComputeEpsilonIdx, the only value left out of the program hash.
uint32_t epsilon_bits(const FrobeniusNormalizeAttributes& args) {
    return std::bit_cast<uint32_t>(args.epsilon);
}

// Kernel handles in the built Program: creation order, minus the kernels with no cores. The non-origin reader exists
// only when more than one core has work, the non-origin group 1 compute kernel only when group 1 has more cores than
// the origin, and the group 2 compute kernel only on an uneven split. The origin (0,0) is the first core of the walk,
// so it always has work and is in group 1.
struct KernelIndices {
    uint32_t reader_origin = 0U;
    std::optional<uint32_t> reader;
    uint32_t writer = 0U;
    uint32_t compute_origin = 0U;
    std::optional<uint32_t> compute_group_1;
    std::optional<uint32_t> compute_group_2;
    uint32_t num_kernels = 0U;
};

KernelIndices compute_kernel_indices(const WorkSplit& split) {
    KernelIndices indices;
    uint32_t next = 0U;
    indices.reader_origin = next++;
    if (split.num_cores > 1U) {
        indices.reader = next++;
    }
    indices.writer = next++;
    indices.compute_origin = next++;
    if (split.num_group_1_cores > 1U) {
        indices.compute_group_1 = next++;
    }
    if (split.has_group_2()) {
        indices.compute_group_2 = next++;
    }
    indices.num_kernels = next;
    return indices;
}

}  // namespace

tt::tt_metal::ProgramDescriptor FrobeniusNormalizeProgramFactory::create_descriptor(
    const FrobeniusNormalizeAttributes& args,
    const FrobeniusNormalizeTensorArgs& tensor_args,
    FrobeniusNormalizeTensorReturn& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor desc;

    // -------------------------------------------------------------------------
    // 1) Data formats and tile sizes and compute split
    // -------------------------------------------------------------------------
    tt::DataFormat bf16_format = tt::DataFormat::Float16_b;
    tt::DataFormat fp32_format = tt::DataFormat::Float32;

    uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    uint32_t total_tiles = input.physical_volume() / tt::constants::TILE_HW;

    auto compute_grid = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles);
    const WorkSplit split = compute_frobenius_work_split(input);
    check_work_split(split, num_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2);

    uint32_t block_size = std::min(4U, tiles_per_core_g1);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t input_buf_tiles = 2 * block_size;
    const uint32_t output_buf_tiles = 2 * block_size;

    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbInput, bf16_format, bf16_tile_size, input_buf_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbSqAcc, fp32_format, fp32_tile_size, 1));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbRecv, fp32_format, fp32_tile_size, 1));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbNorm, fp32_format, fp32_tile_size, 1));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbOutput, bf16_format, bf16_tile_size, output_buf_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCbSqPartial, fp32_format, fp32_tile_size, 1));

    // -------------------------------------------------------------------------
    // 3) Create semaphores
    // -------------------------------------------------------------------------
    // Reduction: non-origin cores increment after writing partial to origin's L1
    // Origin waits for count == num_cores - 1 before reading all partials
    desc.semaphores.push_back(make_semaphore_descriptor(kReductionSemId, all_cores, 0U));
    // Broadcast signal: origin multicasts 1 after norm is ready
    // All cores wait on this before starting Phase 4 (normalization)
    desc.semaphores.push_back(make_semaphore_descriptor(kBcastSemId, all_cores, 0U));

    // Multicast bounding box: physical coords covering all active cores
    auto get_phys = [&](uint32_t lx, uint32_t ly) -> tt::tt_metal::CoreCoord {
        return device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{lx, ly});
    };
    auto bb = all_cores.bounding_box();
    auto mcast_start = get_phys(bb.start_coord.x, bb.start_coord.y);
    auto mcast_end = get_phys(bb.end_coord.x, bb.end_coord.y);
    uint32_t mcast_num_dests = bb.size();

    // -------------------------------------------------------------------------
    // 4) Create kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* output_buffer = output[0].buffer();

    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Got {}",
        enchantum::to_string(input_buffer->buffer_type()));
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Got {}",
        enchantum::to_string(output_buffer->buffer_type()));

    // Origin = core (0,0)
    tt::tt_metal::CoreCoord origin_core{0, 0};
    auto origin_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(origin_core));
    auto non_origin_cores = all_cores.subtract(origin_set);
    std::map<std::string, std::string> defines;
    auto origin_defines = defines;
    origin_defines["IS_ORIGIN"] = "1";

    auto origin_phys = get_phys(0, 0);
    std::vector<uint32_t> reader_ct_args{
        static_cast<uint32_t>(origin_phys.x),
        static_cast<uint32_t>(origin_phys.y),
        num_cores,
        static_cast<uint32_t>(mcast_start.x),
        static_cast<uint32_t>(mcast_start.y),
        static_cast<uint32_t>(mcast_end.x),
        static_cast<uint32_t>(mcast_end.y),
        mcast_num_dests,
        block_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);

    auto reader_origin = make_reader_kernel_descriptor(origin_set, reader_ct_args, origin_defines, kReaderKernelPath);
    auto reader_kernel = make_reader_kernel_descriptor(non_origin_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = make_writer_kernel_descriptor(all_cores, writer_ct_args, defines, kWriterKernelPath);

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest[kCbInput] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbSqAcc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbRecv] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    auto make_compute_kernel = [&](const tt::tt_metal::CoreRangeSet& cores,
                                   const std::vector<uint32_t>& ct_args,
                                   const std::map<std::string, std::string>& defs) {
        return make_compute_kernel_descriptor(
            cores,
            ct_args,
            defs,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true,
            unpack_to_dest,
            tt::tt_metal::MathFidelity::HiFi4,
            /*math_approx_mode=*/false);
    };

    // Origin compute kernel (IS_ORIGIN=1)
    uint32_t origin_tiles = core_group_1.contains(origin_core) ? tiles_per_core_g1 : tiles_per_core_g2;
    auto compute_origin = make_compute_kernel(origin_set, {origin_tiles, block_size}, origin_defines);

    // Non-origin compute kernels (core_group_1/2 minus origin)
    auto compute_g1 = make_compute_kernel(core_group_1.subtract(origin_set), {tiles_per_core_g1, block_size}, defines);
    auto compute_g2 = make_compute_kernel(core_group_2.subtract(origin_set), {tiles_per_core_g2, block_size}, defines);

    // Kernels in creation order, minus those with no cores; compute_kernel_indices() gives their handles.
    const KernelIndices kernel_idx = compute_kernel_indices(split);
    for (auto* kernel : {&reader_origin, &reader_kernel, &writer_kernel, &compute_origin, &compute_g1, &compute_g2}) {
        if (!kernel->core_ranges.ranges().empty()) {
            desc.kernels.push_back(std::move(*kernel));
        }
    }
    TT_FATAL(
        desc.kernels.size() == kernel_idx.num_kernels,
        "FrobeniusNormalize: {} kernels have cores, expected {}",
        desc.kernels.size(),
        kernel_idx.num_kernels);

    // -------------------------------------------------------------------------
    // 5) Set per-core runtime args
    // -------------------------------------------------------------------------
    // On a cache hit override_runtime_arguments() rewrites the input and output addresses and epsilon (the k*Idx slots)
    // on every core. Tile counts and offsets, semaphore ids and the walk index follow from the hashed input spec.
    const uint32_t epsilon = epsilon_bits(args);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [logical_core, core_index, tiles_this_core, tiles_written, in_group_1] = work;
        const bool is_origin = logical_core == origin_core;
        desc.kernels[is_origin ? kernel_idx.reader_origin : kernel_idx.reader.value()].emplace_runtime_args(
            logical_core, {input_buffer, tiles_this_core, tiles_written, kReductionSemId, core_index, kBcastSemId});
        desc.kernels[kernel_idx.writer].emplace_runtime_args(
            logical_core, {output_buffer, tiles_this_core, tiles_written});
        const uint32_t compute_idx = is_origin    ? kernel_idx.compute_origin
                                     : in_group_1 ? kernel_idx.compute_group_1.value()
                                                  : kernel_idx.compute_group_2.value();
        desc.kernels[compute_idx].emplace_runtime_args(logical_core, {epsilon});
    });

    // -------------------------------------------------------------------------
    // 6) Return the program descriptor
    // -------------------------------------------------------------------------
    return desc;
}

void FrobeniusNormalizeProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const FrobeniusNormalizeAttributes& operation_attributes,
    const FrobeniusNormalizeTensorArgs& tensor_args,
    FrobeniusNormalizeTensorReturn& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    // Tile counts and offsets, semaphore ids and walk indices follow from the hashed input spec and keep the values
    // create_descriptor() wrote.
    const uint32_t input_address = tensor_args.input.buffer()->address();
    const uint32_t output_address = output[0].buffer()->address();
    const uint32_t epsilon = epsilon_bits(operation_attributes);
    const WorkSplit split = compute_frobenius_work_split(tensor_args.input);
    const KernelIndices kernel_idx = compute_kernel_indices(split);

    auto& reader_origin_rt = tt::tt_metal::GetRuntimeArgs(program, kernel_idx.reader_origin);
    auto& writer_rt = tt::tt_metal::GetRuntimeArgs(program, kernel_idx.writer);
    auto& compute_origin_rt = tt::tt_metal::GetRuntimeArgs(program, kernel_idx.compute_origin);
    // A kernel that create_descriptor() left out has no cores, so its stand-in (the origin's args) is never indexed.
    auto& reader_rt = kernel_idx.reader ? tt::tt_metal::GetRuntimeArgs(program, *kernel_idx.reader) : reader_origin_rt;
    auto& compute_g1_rt = kernel_idx.compute_group_1
                              ? tt::tt_metal::GetRuntimeArgs(program, *kernel_idx.compute_group_1)
                              : compute_origin_rt;
    auto& compute_g2_rt = kernel_idx.compute_group_2
                              ? tt::tt_metal::GetRuntimeArgs(program, *kernel_idx.compute_group_2)
                              : compute_origin_rt;

    const tt::tt_metal::CoreCoord origin_core{0, 0};
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;
        const bool is_origin = core == origin_core;
        auto& rt = is_origin ? reader_origin_rt : reader_rt;
        rt[core.x][core.y][kReaderInputAddrIdx] = input_address;
        writer_rt[core.x][core.y][kWriterOutputAddrIdx] = output_address;

        auto& compute_rt = is_origin ? compute_origin_rt : (work.in_group_1 ? compute_g1_rt : compute_g2_rt);
        compute_rt[core.x][core.y][kComputeEpsilonIdx] = epsilon;
    });
}

}  // namespace ttml::metal::ops::frobenius_normalize::device
