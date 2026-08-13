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
constexpr auto kRecordsCbIndex = tt::CBIndex::c_4;

// Boundary-row partials exchanged between cores: [valid, row, 32 maxima, 32 indices]
// padded to 288 bytes, two per core (a core's first and last row). Bounded by core
// count, never by shape -- see writer_gumbel_sample.cpp.
constexpr uint32_t kRecordBytes = 72U * sizeof(uint32_t);
constexpr uint32_t kRecordsPerCore = 2U;

const std::string kDoLogitsMaskDefineKey = "DO_LOGITS_MASK";
const std::string kDoGumbelNoiseDefineKey = "DO_GUMBEL_NOISE";
const std::string kDoPositionsDefineKey = "DO_POSITIONS";

// The reader carries Ht as a runtime arg at slot 4 (see reader_gumbel_sample.cpp), so its positions
// start one slot later than the writer's. These two MUST diverge -- bumping them together would make
// the writer read its core_index as positions[0], so every core would write a different wrong row
// with no fault raised.
constexpr uint32_t kReaderHtIdx = 4U;

// First runtime-arg slot of the per-entry token positions, which are appended to both dataflow
// kernels' fixed args. Every core receives the FULL local list (B_local entries), not just the
// entries it touches: the origin core has to re-derive the target row of any boundary entry it
// merges, and B_local is a couple of dozen at most.
constexpr uint32_t kReaderPositionsArgBase = 5U;
constexpr uint32_t kWriterPositionsArgBase = 4U;
static_assert(kReaderPositionsArgBase == kReaderHtIdx + 1U);

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
    uint32_t total_tiles{};  // the unit work is actually split over -- see below
    uint32_t tiles_per_core_group_1{};
    uint32_t tiles_per_core_group_2{};
    bool position_aware{};   // sample one row per batch entry instead of every row
    uint32_t num_entries{};  // NC -- one output row each, when position_aware
};

GumbelSampleLayout compute_layout(const ttnn::Tensor& logits, bool position_aware) {
    GumbelSampleLayout layout;
    layout.position_aware = position_aware;

    const auto padded_shape = logits.padded_shape();
    const auto logical_shape = logits.logical_shape();
    TT_FATAL(padded_shape.rank() == 4U, "GumbelSample: logits must be 4D, got rank {}", padded_shape.rank());

    layout.Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    layout.Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    layout.num_entries = NC;
    layout.total_rows = NC * layout.Ht;
    layout.block_size = get_block_size(layout.Wt, 4U);
    layout.logical_vocab = logical_shape[-1];
    layout.logical_tokens = logical_shape[-2];

    auto* device = logits.device();
    const auto grid = device->compute_with_storage_grid_size();
    layout.num_cores_y = grid.y;

    // Split over TILES, not tile rows. A row-based split yields only NC*Ht units, and in decode
    // (tokens == 1 => Ht == 1) that is just the local batch: a few dozen units on an ~80 core grid,
    // leaving most of it idle while each active core carried a whole vocabulary. Every ttnn op this
    // kernel replaces splits by tile, which is why six of them beat one of this.
    //
    // When positions were supplied the tile space shrinks further, to one tile ROW per batch entry
    // instead of Ht of them: a "virtual" tile vt maps to entry vt / Wt, column vt % Wt, and the
    // reader turns that into the real page using the entry's position. Prefill hands this op the
    // logits for every token position but only ever consumes one row per sequence, so this is an
    // Ht-fold cut in tiles read, reduced and DRAM-touched -- at tokens = 448 that is 14x, and it
    // brings prefill sampling down to decode cost regardless of context length.
    layout.total_tiles = position_aware ? (NC * layout.Wt) : (layout.total_rows * layout.Wt);

    auto [num_cores, all_cores, group_1, group_2, tiles_1, tiles_2] =
        tt::tt_metal::split_work_to_cores(grid, layout.total_tiles);
    layout.num_cores = num_cores;
    layout.all_cores = all_cores;
    layout.core_group_1 = group_1;
    layout.core_group_2 = group_2;
    layout.tiles_per_core_group_1 = tiles_1;
    layout.tiles_per_core_group_2 = tiles_2;

    return layout;
}

// Per-core work assignment, single-sourced so the cache-miss build and the cache-hit patch derive
// identical (core, rows, start_row) triples -- and therefore identical RNG streams.
struct CoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t num_tiles{};
    uint32_t start_tile{};
};

std::vector<CoreWork> core_layout(const GumbelSampleLayout& layout) {
    std::vector<CoreWork> work;
    work.reserve(layout.num_cores);
    uint32_t tiles_assigned = 0U;
    for (uint32_t i = 0; i < layout.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core{i / layout.num_cores_y, i % layout.num_cores_y};
        uint32_t tiles = 0U;
        if (layout.core_group_1.contains(core)) {
            tiles = layout.tiles_per_core_group_1;
        } else if (layout.core_group_2.contains(core)) {
            tiles = layout.tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "GumbelSample: core ({}, {}) is not in either core group", core.x, core.y);
        }
        work.push_back({core, tiles, tiles_assigned});
        tiles_assigned += tiles;
    }
    return work;
}

// Domain-separate the RNG per (device, core). rand_tile_init folds stream_id into the seed, so
// distinct stream ids give disjoint deterministic streams; devices that share a stream id (replicas
// on a non-seeded axis) intentionally draw identical noise.
uint32_t rand_stream_id(const GumbelSampleLayout& layout, uint32_t device_index, uint32_t start_tile) {
    return device_index * layout.total_tiles + start_tile;
}

// This device's slice of the caller's GLOBAL position list.
//
// A tensor's shape here is its LOCAL shard, so the op sees B_local rows and cannot tell on its own
// which global rows those are. The mapping is exactly the one the RNG already uses: batch shards are
// laid out along the SEEDED (data-parallel) axes, and devices that differ only on a replicated axis
// hold the SAME rows -- so they must get the same slice, which is what seeded_linear_index returns.
// A list already sized to B_local is taken as replicated and used verbatim.
std::vector<uint32_t> local_positions_of(
    const operation_attributes_t& args, const GumbelSampleLayout& layout, uint32_t device_index) {
    if (!layout.position_aware) {
        return {};
    }
    if (args.positions.size() == layout.num_entries) {
        return args.positions;
    }
    const uint32_t offset = device_index * layout.num_entries;
    TT_FATAL(
        offset + layout.num_entries <= args.positions.size(),
        "GumbelSample: positions list of {} is too short for batch shard {} of {} rows",
        args.positions.size(),
        device_index,
        layout.num_entries);
    return std::vector<uint32_t>(args.positions.begin() + offset, args.positions.begin() + offset + layout.num_entries);
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

    // Boundary-row partials. Every core reserves the FULL table so the origin can receive each
    // core's records at that core's own offset -- a fixed 2 records per core, independent of shape.
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kRecordsCbIndex,
        tt::DataFormat::UInt32,
        layout.num_cores * kRecordsPerCore * kRecordBytes);

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
    if (layout.position_aware) {
        defines[kDoPositionsDefineKey] = "1";
    }

    // Slot 2 held Ht, which is now a runtime arg so that one program serves every prompt length.
    // The slot is kept (pinned) rather than removed: reader_gumbel_sample.cpp hard-codes
    // TensorAccessorArgs<3> and chains the mask accessor off its offset, so shifting it would
    // misdecode the accessor words instead of failing to compile.
    std::vector<uint32_t> reader_ct_args{layout.block_size, layout.Wt, 1U};
    tt::tt_metal::TensorAccessorArgs(logits_buffer).append_to(reader_ct_args);
    if (has_mask) {
        tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    shared_vars.reader_kernel_id =
        create_reader_kernel(program, layout.all_cores, reader_ct_args, defines, kReaderKernelPath);

    // Origin core (index 0) merges the boundary rows; the others signal it once each.
    const uint32_t reduction_sem_id = tt::tt_metal::CreateSemaphore(program, layout.all_cores, 0);
    const auto origin_logical = tt::tt_metal::CoreCoord{0, 0};
    const auto origin_phys = logits.device()->worker_core_from_logical_core(origin_logical);

    std::vector<uint32_t> writer_ct_args{
        layout.Wt,
        layout.logical_vocab,
        // Both of these are dead under DO_POSITIONS -- their only uses sit past unconditional
        // returns -- but a compile-time arg is hashed into the kernel binary whether it is read or
        // not, so they are pinned to keep the build independent of the token dimension. ONE, never
        // zero: the dead fallback divides by Ht in code that is still compiled, and the JIT builds
        // with -Wall -Werror, so a zero here is a -Werror=div-by-zero build failure. At 1 the dead
        // path degenerates to exactly what the position path does, which keeps it harmless if the
        // guard is ever refactored away.
        layout.position_aware ? 1U : layout.logical_tokens,
        layout.position_aware ? 1U : layout.Ht,
        layout.num_cores,
        reduction_sem_id,
        static_cast<uint32_t>(origin_phys.x),
        static_cast<uint32_t>(origin_phys.y)};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    shared_vars.writer_kernel_id =
        create_writer_kernel(program, layout.all_cores, writer_ct_args, defines, kWriterKernelPath);

    const std::vector<uint32_t> compute_ct_args_g1{layout.tiles_per_core_group_1, layout.block_size};
    shared_vars.compute_kernel_group_1_id = create_compute_kernel(
        program, layout.core_group_1, compute_ct_args_g1, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    if (!layout.core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_ct_args_g2{layout.tiles_per_core_group_2, layout.block_size};
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

    const auto local_positions = local_positions_of(args, layout, device_index);

    uint32_t core_index = 0U;
    for (const auto& [core, num_tiles, start_tile] : core_layout(layout)) {
        std::vector<uint32_t> reader_args{
            logits_buffer->address(), has_mask ? mask_buffer->address() : 0U, num_tiles, start_tile, layout.Ht};
        reader_args.insert(reader_args.end(), local_positions.begin(), local_positions.end());
        SetRuntimeArgs(program, shared_vars.reader_kernel_id, core, reader_args);

        std::vector<uint32_t> writer_args{output_buffer->address(), num_tiles, start_tile, core_index};
        writer_args.insert(writer_args.end(), local_positions.begin(), local_positions.end());
        SetRuntimeArgs(program, shared_vars.writer_kernel_id, core, writer_args);

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
             rand_stream_id(layout, device_index, start_tile)});
        ++core_index;
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
    const auto layout = compute_layout(logits, !operation_attributes.positions.empty());

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

    const auto layout = compute_layout(logits, !operation_attributes.positions.empty());
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
        // Positions are runtime-only and change on every prefill (each sequence's prompt ends
        // somewhere different), so they must be re-applied here just like the seed.
        const auto local_positions = local_positions_of(operation_attributes, layout, vars.device_seed_offset);

        auto& reader_args = GetRuntimeArgs(program, vars.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, vars.writer_kernel_id);
        auto& compute_g1_args = GetRuntimeArgs(program, vars.compute_kernel_group_1_id);
        auto& compute_g2_args = vars.core_group_2.ranges().empty()
                                    ? compute_g1_args
                                    : GetRuntimeArgs(program, vars.compute_kernel_group_2_id);

        // core_index is not re-patched here: it is a property of the work split, which is identical
        // on every dispatch. Only the buffer addresses and the seed change.
        for (const auto& [core, num_tiles, start_tile] : work) {
            {
                auto& core_args = reader_args[core.x][core.y];
                core_args[kReaderLogitsBufferIdx] = logits_address;
                core_args[kReaderMaskBufferIdx] = mask_address;
                // Ht is runtime-only (deliberately out of the program hash in position mode) so it
                // must be re-applied on every cache hit. Unconditional, because the slot exists in
                // both modes -- patching it only in position mode would run one past the end of the
                // 4-word decode arg vector, and RuntimeArgsData's bounds check is a TT_ASSERT that
                // compiles away in Release, so the write would land in the packed dispatch command.
                core_args[kReaderHtIdx] = layout.Ht;
                for (size_t i = 0; i < local_positions.size(); ++i) {
                    core_args[kReaderPositionsArgBase + i] = local_positions[i];
                }
            }
            {
                auto& core_args = writer_args[core.x][core.y];
                core_args[kWriterOutputBufferIdx] = output_address;
                for (size_t i = 0; i < local_positions.size(); ++i) {
                    core_args[kWriterPositionsArgBase + i] = local_positions[i];
                }
            }
            {
                auto& core_args = vars.core_group_1.contains(core) ? compute_g1_args[core.x][core.y]
                                                                   : compute_g2_args[core.x][core.y];
                core_args[kComputeSeedIdx] = operation_attributes.seed;
                core_args[kComputeRandFromIdx] = rand_from_bits;
                core_args[kComputeRandScaleIdx] = rand_scale_bits;
                core_args[kComputeInvTemperatureIdx] = inv_temperature_bits;
                core_args[kComputeRandStreamIdx] = rand_stream_id(layout, vars.device_seed_offset, start_tile);
            }
        }
    }
}

}  // namespace ttml::metal::ops::gumbel_sample::device
