// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/gumbel_sample/device/kernels/dataflow/reader_gumbel_sample.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/gumbel_sample/device/kernels/dataflow/writer_gumbel_sample.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/gumbel_sample/device/kernels/compute/gumbel_sample_kernel.cpp";

// reader runtime arg slots
constexpr uint32_t kReaderLogitsBufferIdx = 0U;
constexpr uint32_t kReaderMaskBufferIdx = 1U;
// writer runtime arg slots
constexpr uint32_t kWriterOutputBufferIdx = 0U;
// compute runtime arg slots
constexpr uint32_t kComputeSeedIdx = 0U;
constexpr uint32_t kComputeRandFromIdx = 1U;
constexpr uint32_t kComputeRandScaleIdx = 2U;
constexpr uint32_t kComputeInvTemperatureIdx = 3U;
constexpr uint32_t kComputeRandStreamIdx = 4U;

constexpr auto kLogitsCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskCbIndex = tt::CBIndex::c_1;
constexpr auto kScoresCbIndex = tt::CBIndex::c_2;
constexpr auto kOutputStagingCbIndex = tt::CBIndex::c_3;

const std::string kDoLogitsMaskDefineKey = "DO_LOGITS_MASK";
const std::string kDoGumbelNoiseDefineKey = "DO_GUMBEL_NOISE";

// Uniform draw bounds for the Gumbel transform g = -log(-log(U)).
//
// Lower bound 2^-32 matches ttnn_fixed::sample's gumbel_uniform_lower_bound and caps the noise at
// g <= -log(-log(2^-32)) ~ -3.1 on the low side.
//
// The UPPER bound deliberately differs from the composed implementation. `rand_tile` produces
// values on a CLOSED interval [from, from + scale], and the composed path passes 1.0F as the upper
// bound -- so U == 1.0 is attainable, and then log(1) = 0, -log(0) = +inf, g = +inf, which pins the
// argmax onto that token with certainty. It is a ~2^-32-per-element event, but it is a real one, so
// here the top of the range is the largest float32 strictly below 1.0 and g stays finite (max
// ~16.6).
constexpr float kGumbelUniformLowerBound = 0x1p-32F;
constexpr float kGumbelUniformUpperBound = 0x1.fffffep-1F;  // nextafterf(1.0F, 0.0F)

// `rand_tile` is documented as inclusive of `from + scale`. Shrink the scale by one ULP if rounding
// would push the top of the range past the intended upper bound -- same guard compute_uniform.cpp
// applies on the device side.
uint32_t compute_rand_scale_bits(float lower, float upper) {
    float scale = upper - lower;
    uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (lower + scale > upper && scale_bits != 0U) {
        --scale_bits;
    }
    return scale_bits;
}

// Linear index of this device among the SEEDED (data-parallel) mesh axes only. Devices that differ
// solely on a replicated axis get the same index -- and therefore the same RNG stream -- which is
// what keeps a tensor-parallel replica group in sync. Mirrors the shard_linear_idx computation
// inside ttnn::rand's program factory.
uint32_t seeded_linear_index(
    const ttnn::MeshCoordinate& coord,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<uint32_t>& seed_axes) {
    auto is_seeded = [&](size_t axis) {
        return mesh_shape[axis] > 1U &&
               std::find(seed_axes.begin(), seed_axes.end(), static_cast<uint32_t>(axis)) != seed_axes.end();
    };

    uint32_t linear_index = 0U;
    uint32_t stride = 1U;
    for (int axis = static_cast<int>(mesh_shape.dims()) - 1; axis >= 0; --axis) {
        if (is_seeded(static_cast<size_t>(axis))) {
            linear_index += static_cast<uint32_t>(coord[static_cast<size_t>(axis)]) * stride;
            stride *= static_cast<uint32_t>(mesh_shape[static_cast<size_t>(axis)]);
        }
    }
    return linear_index;
}

}  // namespace

namespace ttml::metal::ops::gumbel_sample::device {

namespace {

// Everything the per-core loop needs, derived once so create_mesh_workload and
// override_runtime_arguments can never disagree about the work split.
struct GumbelSampleLayout {
    uint32_t Wt{};              // vocab tiles per row
    uint32_t Ht{};              // token tiles per batch entry
    uint32_t total_rows{};      // NC * Ht -- one "row" is a 32-token tile row
    uint32_t block_size{};      // vocab tiles streamed per CB block; always divides Wt
    uint32_t logical_vocab{};   // V, for bounding the argmax scan past tile padding
    uint32_t logical_tokens{};  // tokens, for bounding the row scan past tile padding
    uint32_t num_cores{};
    uint32_t num_cores_y{};
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t rows_per_core_group_1{};
    uint32_t rows_per_core_group_2{};
};

GumbelSampleLayout compute_layout(const ttnn::Tensor& logits) {
    GumbelSampleLayout layout;

    const auto padded_shape = logits.padded_shape();
    const auto logical_shape = logits.logical_shape();
    TT_FATAL(padded_shape.rank() == 4U, "GumbelSample: logits must be 4D, got rank {}", padded_shape.rank());

    layout.Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    layout.Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    layout.total_rows = NC * layout.Ht;
    layout.block_size = get_block_size(layout.Wt, 4U);
    layout.logical_vocab = logical_shape[-1];
    layout.logical_tokens = logical_shape[-2];

    auto* device = logits.device();
    const auto grid = device->compute_with_storage_grid_size();
    layout.num_cores_y = grid.y;

    auto [num_cores, all_cores, group_1, group_2, rows_1, rows_2] =
        tt::tt_metal::split_work_to_cores(grid, layout.total_rows);
    layout.num_cores = num_cores;
    layout.all_cores = all_cores;
    layout.core_group_1 = group_1;
    layout.core_group_2 = group_2;
    layout.rows_per_core_group_1 = rows_1;
    layout.rows_per_core_group_2 = rows_2;

    return layout;
}

// Per-core work assignment, single-sourced so the cache-miss build and the cache-hit patch derive
// identical (core, rows, start_row) triples -- and therefore identical RNG streams.
struct CoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t num_rows{};
    uint32_t start_row{};
};

std::vector<CoreWork> core_layout(const GumbelSampleLayout& layout) {
    std::vector<CoreWork> work;
    work.reserve(layout.num_cores);
    uint32_t rows_written = 0U;
    for (uint32_t i = 0; i < layout.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core{i / layout.num_cores_y, i % layout.num_cores_y};
        uint32_t rows = 0U;
        if (layout.core_group_1.contains(core)) {
            rows = layout.rows_per_core_group_1;
        } else if (layout.core_group_2.contains(core)) {
            rows = layout.rows_per_core_group_2;
        } else {
            TT_FATAL(false, "GumbelSample: core ({}, {}) is not in either core group", core.x, core.y);
        }
        work.push_back({core, rows, rows_written});
        rows_written += rows;
    }
    return work;
}

// Domain-separate the RNG per (device, core). rand_tile_init folds stream_id into the seed, so
// distinct stream ids give disjoint deterministic streams; devices that share a stream id (replicas
// on a non-seeded axis) intentionally draw identical noise.
uint32_t rand_stream_id(const GumbelSampleLayout& layout, uint32_t device_index, uint32_t start_row) {
    return device_index * layout.total_rows + start_row;
}

tt::tt_metal::Program build_program(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const GumbelSampleLayout& layout,
    uint32_t device_index,
    GumbelSampleSharedVariables& shared_vars) {
    tt::tt_metal::Program program{};

    const auto& logits = tensor_args.logits;
    const bool has_mask = tensor_args.logits_padding_mask.has_value();

    const tt::DataFormat logits_format = datatype_to_dataformat_converter(logits.dtype());
    const uint32_t logits_tile_bytes = tt::tile_size(logits_format);
    const uint32_t score_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    // -------------------------------------------------------------------------
    // Circular buffers. Peak L1 is a handful of tiles regardless of V: this is the whole point of
    // the fusion -- the composed path materializes several full [B, 1, tokens, V] tensors in DRAM.
    // -------------------------------------------------------------------------
    const uint32_t streamed_tiles = 2U * layout.block_size;  // double-buffered

    create_circular_buffer(program, layout.all_cores, kLogitsCbIndex, logits_format, logits_tile_bytes, streamed_tiles);
    if (has_mask) {
        create_circular_buffer(
            program, layout.all_cores, kMaskCbIndex, logits_format, logits_tile_bytes, streamed_tiles);
    }
    // Scores stay FP32: the Gumbel noise is generated in FP32 and a bf16 round trip here would
    // quantize the very comparisons the argmax is about to make.
    create_circular_buffer(
        program, layout.all_cores, kScoresCbIndex, tt::DataFormat::Float32, score_tile_bytes, streamed_tiles);
    // Staging for one tile-row of results: 32 token ids, each in its own NOC-aligned slot (see
    // kOutputSlotBytes in the writer kernel) so all 32 page writes can be issued behind one barrier.
    constexpr uint32_t kOutputSlotBytes = 32U;
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kOutputStagingCbIndex,
        tt::DataFormat::UInt32,
        tt::constants::TILE_HEIGHT * kOutputSlotBytes);

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    auto* logits_buffer = logits.buffer();
    auto* mask_buffer = has_mask ? tensor_args.logits_padding_mask->buffer() : nullptr;
    auto* output_buffer = output.buffer();

    // temperature == 0 selects the greedy variant of the compute kernel: no RNG, no scaling, the
    // logits go straight through to the writer's running argmax.
    const bool do_gumbel_noise = args.temperature > 0.0F;

    std::map<std::string, std::string> defines;
    if (has_mask) {
        defines[kDoLogitsMaskDefineKey] = "1";
    }
    if (do_gumbel_noise) {
        defines[kDoGumbelNoiseDefineKey] = "1";
    }

    std::vector<uint32_t> reader_ct_args{layout.block_size, layout.Wt};
    tt::tt_metal::TensorAccessorArgs(logits_buffer).append_to(reader_ct_args);
    if (has_mask) {
        tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    shared_vars.reader_kernel_id =
        create_reader_kernel(program, layout.all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{layout.Wt, layout.logical_vocab, layout.logical_tokens, layout.Ht};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    shared_vars.writer_kernel_id =
        create_writer_kernel(program, layout.all_cores, writer_ct_args, defines, kWriterKernelPath);

    const std::vector<uint32_t> compute_ct_args_g1{layout.rows_per_core_group_1, layout.block_size, layout.Wt};
    shared_vars.compute_kernel_group_1_id = create_compute_kernel(
        program, layout.core_group_1, compute_ct_args_g1, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    if (!layout.core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_ct_args_g2{layout.rows_per_core_group_2, layout.block_size, layout.Wt};
        shared_vars.compute_kernel_group_2_id = create_compute_kernel(
            program, layout.core_group_2, compute_ct_args_g2, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // -------------------------------------------------------------------------
    // Runtime args
    // -------------------------------------------------------------------------
    const uint32_t rand_from_bits = std::bit_cast<uint32_t>(kGumbelUniformLowerBound);
    const uint32_t rand_scale_bits = compute_rand_scale_bits(kGumbelUniformLowerBound, kGumbelUniformUpperBound);
    // Guard the reciprocal: temperature == 0 would make this inf. The greedy kernel never reads it,
    // but an inf sitting in a runtime arg is a trap for anyone who later makes it read it.
    const uint32_t inv_temperature_bits = do_gumbel_noise ? std::bit_cast<uint32_t>(1.0F / args.temperature) : 0U;

    for (const auto& [core, num_rows, start_row] : core_layout(layout)) {
        SetRuntimeArgs(
            program,
            shared_vars.reader_kernel_id,
            core,
            {logits_buffer->address(), has_mask ? mask_buffer->address() : 0U, num_rows, start_row});

        SetRuntimeArgs(program, shared_vars.writer_kernel_id, core, {output_buffer->address(), num_rows, start_row});

        const auto compute_kernel = layout.core_group_1.contains(core) ? shared_vars.compute_kernel_group_1_id
                                                                       : shared_vars.compute_kernel_group_2_id;
        SetRuntimeArgs(
            program,
            compute_kernel,
            core,
            {args.seed,
             rand_from_bits,
             rand_scale_bits,
             inv_temperature_bits,
             rand_stream_id(layout, device_index, start_row)});
    }

    shared_vars.core_group_1 = layout.core_group_1;
    shared_vars.core_group_2 = layout.core_group_2;
    shared_vars.num_cores = layout.num_cores;
    shared_vars.num_cores_y = layout.num_cores_y;
    shared_vars.device_seed_offset = device_index;

    return program;
}

}  // namespace

GumbelSampleProgramFactory::cached_mesh_workload_t GumbelSampleProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& logits = tensor_args.logits;
    auto* mesh_device = logits.device();
    TT_FATAL(mesh_device != nullptr, "GumbelSample: logits must live on a mesh device");

    const auto mesh_shape = mesh_device->shape();
    const auto layout = compute_layout(logits);

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : coord_range) {
            const uint32_t device_index = seeded_linear_index(mesh_coord, mesh_shape, operation_attributes.seed_axes);

            shared_variables_t vars{};
            auto program =
                build_program(operation_attributes, tensor_args, tensor_return_value, layout, device_index, vars);

            ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
            mesh_workload.add_program(single_coord_range, std::move(program));
            shared_vars[single_coord_range] = std::move(vars);
        }
    }

    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void GumbelSampleProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& logits = tensor_args.logits;
    const bool has_mask = tensor_args.logits_padding_mask.has_value();

    const auto layout = compute_layout(logits);
    const auto work = core_layout(layout);

    const uint32_t logits_address = logits.buffer()->address();
    const uint32_t mask_address = has_mask ? tensor_args.logits_padding_mask->buffer()->address() : 0U;
    const uint32_t output_address = tensor_return_value.buffer()->address();

    // seed and temperature are runtime-only (deliberately excluded from the program hash so that
    // changing either reuses the cached program), so they must be re-applied on every cache hit
    // alongside the buffer addresses.
    const uint32_t rand_from_bits = std::bit_cast<uint32_t>(kGumbelUniformLowerBound);
    const uint32_t rand_scale_bits = compute_rand_scale_bits(kGumbelUniformLowerBound, kGumbelUniformUpperBound);
    const uint32_t inv_temperature_bits =
        operation_attributes.temperature > 0.0F ? std::bit_cast<uint32_t>(1.0F / operation_attributes.temperature) : 0U;

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& vars = cached_workload.shared_variables.at(coord_range);

        auto& reader_args = GetRuntimeArgs(program, vars.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, vars.writer_kernel_id);
        auto& compute_g1_args = GetRuntimeArgs(program, vars.compute_kernel_group_1_id);
        auto& compute_g2_args = vars.core_group_2.ranges().empty()
                                    ? compute_g1_args
                                    : GetRuntimeArgs(program, vars.compute_kernel_group_2_id);

        for (const auto& [core, num_rows, start_row] : work) {
            {
                auto& core_args = reader_args[core.x][core.y];
                core_args[kReaderLogitsBufferIdx] = logits_address;
                core_args[kReaderMaskBufferIdx] = mask_address;
            }
            {
                auto& core_args = writer_args[core.x][core.y];
                core_args[kWriterOutputBufferIdx] = output_address;
            }
            {
                auto& core_args = vars.core_group_1.contains(core) ? compute_g1_args[core.x][core.y]
                                                                   : compute_g2_args[core.x][core.y];
                core_args[kComputeSeedIdx] = operation_attributes.seed;
                core_args[kComputeRandFromIdx] = rand_from_bits;
                core_args[kComputeRandScaleIdx] = rand_scale_bits;
                core_args[kComputeInvTemperatureIdx] = inv_temperature_bits;
                core_args[kComputeRandStreamIdx] = rand_stream_id(layout, vars.device_seed_offset, start_row);
            }
        }
    }
}

}  // namespace ttml::metal::ops::gumbel_sample::device
