// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"
#include "ttnn/operations/uniform/uniform_range.hpp"

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
// compute runtime arg slots. Slots 1 and 2 hold the rand from/scale bits -- process constants set
// once at build and never re-patched on cache hits, so no index names them.
constexpr uint32_t kComputeSeedIdx = 0U;
constexpr uint32_t kComputeInvTemperatureIdx = 3U;
constexpr uint32_t kComputeRandStreamIdx = 4U;

constexpr auto kLogitsCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskCbIndex = tt::CBIndex::c_1;
constexpr auto kScoresCbIndex = tt::CBIndex::c_2;
constexpr auto kOutputStagingCbIndex = tt::CBIndex::c_3;
constexpr auto kRecordsCbIndex = tt::CBIndex::c_4;

// Boundary-row partials exchanged between cores: [valid, row, 32 maxima, 32 indices], padded to
// 288 bytes -- see writer_gumbel_sample.cpp.
constexpr uint32_t kRecordBytes = 72U * sizeof(uint32_t);

// The reader carries Ht at slot 4 and the writer does not, so the positions address lands at a
// different slot in each kernel; the static_asserts below pin the layouts. Everything that derives
// from the token dimension (Ht, logical tokens) or the mask shape (stride) is a RUNTIME arg so one
// cached program serves every prompt length and both mask shapes.
constexpr uint32_t kReaderHtIdx = 4U;
constexpr uint32_t kReaderPositionsBufferIdx = 5U;
constexpr uint32_t kWriterPositionsBufferIdx = 3U;
static_assert(kReaderPositionsBufferIdx == kReaderHtIdx + 1U);
// The writer's merge routing (owner x/y, send slot, expected shards) follows its positions address.
constexpr uint32_t kWriterMergeRoutingArgs = 4U;
constexpr uint32_t kReaderLogicalTokensIdx = 6U;
constexpr uint32_t kReaderMaskStrideIdx = 7U;
constexpr uint32_t kWriterLogicalTokensIdx = 8U;
static_assert(kReaderLogicalTokensIdx == kReaderPositionsBufferIdx + 1U);
static_assert(kReaderMaskStrideIdx == kReaderLogicalTokensIdx + 1U);
static_assert(kWriterLogicalTokensIdx == kWriterPositionsBufferIdx + kWriterMergeRoutingArgs + 1U);

// Per-entry token positions live in a small device tensor; each kernel stages the entry window its
// tile run touches into these CBs (see position_window.hpp).
constexpr auto kReaderPositionsCbIndex = tt::CBIndex::c_5;
constexpr auto kWriterPositionsCbIndex = tt::CBIndex::c_6;

// Uniform draw bounds for g = -log(-log(U)). The lower bound caps the noise at ~-3.1. The upper
// bound must stay strictly below 1.0: rand_tile's interval is CLOSED, and U == 1.0 would make
// g = +inf and pin the argmax onto that token. gumbel_sfpu.h's approximate log drops its zero
// guard on the strength of exactly these bounds -- change them only together with that header.
constexpr float kGumbelUniformLowerBound = 0x1p-32F;
const float kGumbelUniformUpperBound = ttnn::operations::uniform::largest_supported_float32_below(1.0F);

// Derived once per process and consumed at build only; deliberately not re-derived on cache hits
// (a second derivation site could silently drift, and only cache hits would see it).
const uint32_t kRandFromBits = std::bit_cast<uint32_t>(kGumbelUniformLowerBound);
const uint32_t kRandScaleBits = ttml::metal::ops::gumbel_sample::device::compute_rand_scale_bits(
    kGumbelUniformLowerBound, kGumbelUniformUpperBound);

// Linear index of this device among the SEEDED (data-parallel) mesh axes only: devices differing
// solely on a replicated axis share an index, and therefore an RNG stream.
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
    uint32_t total_tiles{};  // the unit work is split over -- see compute_layout
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

    // Split over TILES, not tile rows: in decode a row-based split leaves most of the grid idle.
    // With positions the tile space shrinks to one VIRTUAL tile row per batch entry (entry vt / Wt,
    // column vt % Wt; the reader maps to the real page via the entry's position) -- an Ht-fold cut
    // that brings prefill sampling down to decode cost regardless of context length.
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

// Op-local stand-ins for the shared core-walk helpers PR #56524 adds to program_utils.hpp
// (ttml::metal::CoreWork / for_each_core_with_work): same walk order, same fields, same call
// shape. The struct is named GumbelCoreWork so it cannot shadow the shared ttml::metal::CoreWork
// once that PR lands; migration is then: delete this block and s/GumbelCoreWork/CoreWork/.
struct GumbelCoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t index;      // position in the walk: core == {index / num_cores_y, index % num_cores_y}
    uint32_t num_units;  // tiles this core processes
    uint32_t start;      // tiles handed to the cores before it
    bool in_group_1;     // which split_work_to_cores group the core is in; picks the compute kernel
};

template <typename Fn>
void for_each_core_with_work(
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
        fn(GumbelCoreWork{core, i, num_units, num_units_written, in_group_1});
        num_units_written += num_units;
    }
}

// Materialized (unlike most ops' walk-and-set) because the merge routing searches backward through
// earlier cores for a split row's owner.
std::vector<GumbelCoreWork> core_layout(const GumbelSampleLayout& layout) {
    std::vector<GumbelCoreWork> work;
    work.reserve(layout.num_cores);
    for_each_core_with_work(
        layout.num_cores,
        layout.num_cores_y,
        layout.core_group_1,
        layout.core_group_2,
        layout.tiles_per_core_group_1,
        layout.tiles_per_core_group_2,
        [&work](const GumbelCoreWork& core_work) { work.push_back(core_work); });
    return work;
}

// Domain-separates the RNG per (device, core); replicas on non-seeded axes intentionally share a
// stream.
uint32_t rand_stream_id(const GumbelSampleLayout& layout, uint32_t device_index, uint32_t start_tile) {
    return device_index * layout.total_tiles + start_tile;
}

// Runtime-arg values the cache-miss build and the cache-hit patch must derive IDENTICALLY -- one
// derivation site, deliberately layout-free so the cache-hit path never recomputes the split.
struct DerivedRuntimeArgs {
    uint32_t inv_temperature_bits{};
    uint32_t positions_address{};
    uint32_t mask_entry_stride{};
};

DerivedRuntimeArgs derive_runtime_args(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    DerivedRuntimeArgs derived{};
    // uses_gumbel_noise guarantees the reciprocal is finite; greedy gets 0, not inf.
    derived.inv_temperature_bits =
        uses_gumbel_noise(args.temperature) ? std::bit_cast<uint32_t>(1.0F / args.temperature) : 0U;
    // Zero when absent: the slot exists in both modes so the cache-hit patch is unconditional.
    derived.positions_address = tensor_args.positions.has_value() ? tensor_args.positions->buffer()->address() : 0U;
    // 0 for a shared [1,1,1,V] mask, Wt for per-row [B,1,1,V]; one cached program serves both.
    derived.mask_entry_stride =
        (tensor_args.logits_mask.has_value() && tensor_args.logits_mask->logical_shape()[0] > 1U)
            ? (tensor_args.logits.padded_shape()[-1] / tt::constants::TILE_WIDTH)
            : 0U;
    return derived;
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
    const bool has_mask = tensor_args.logits_mask.has_value();

    const tt::DataFormat logits_format = datatype_to_dataformat_converter(logits.dtype());
    const uint32_t logits_tile_bytes = tt::tile_size(logits_format);
    const uint32_t score_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    // Split-row merge routing. A row's owner is the core holding its FIRST tile; the contiguous
    // split means a core sends at most one record and runs at most one merge. Senders to an owner
    // are enumerated in core order, giving each a collision-free slot in the owner's records CB.
    const auto work = core_layout(layout);
    std::vector<uint32_t> expected_shards(layout.num_cores, 0U);
    std::vector<std::array<uint32_t, 3U>> send_routing(layout.num_cores, {0U, 0U, 0U});  // x, y, slot
    for (uint32_t sender = 1U; sender < layout.num_cores; ++sender) {
        if (work[sender].start % layout.Wt == 0U) {
            continue;  // first row starts here: nothing to send
        }
        const uint32_t row_first_tile = (work[sender].start / layout.Wt) * layout.Wt;
        uint32_t owner = sender - 1U;
        while (work[owner].start > row_first_tile) {
            --owner;
        }
        const auto owner_phys = logits.device()->worker_core_from_logical_core(work[owner].core);
        send_routing[sender] = {
            static_cast<uint32_t>(owner_phys.x), static_cast<uint32_t>(owner_phys.y), expected_shards[owner]};
        ++expected_shards[owner];
    }
    const uint32_t max_foreign_shards = *std::max_element(expected_shards.begin(), expected_shards.end());
    // The fan-in is bounded by the split, not the core count; this guards the derivation against a
    // work-split change (the records CB is sized from the exact fan-in above).
    const uint32_t min_tiles_per_core =
        layout.core_group_2.ranges().empty()
            ? layout.tiles_per_core_group_1
            : std::min(layout.tiles_per_core_group_1, layout.tiles_per_core_group_2);
    TT_FATAL(
        max_foreign_shards <= (layout.Wt - 1U) / std::max(min_tiles_per_core, 1U) + 1U,
        "GumbelSample: merge fan-in {} exceeds the split-derived bound (Wt={}, min tiles/core={})",
        max_foreign_shards,
        layout.Wt,
        min_tiles_per_core);

    // Circular buffers. Peak L1 is a handful of tiles regardless of V -- the point of the fusion.
    const uint32_t streamed_tiles = 2U * layout.block_size;  // double-buffered

    create_circular_buffer(program, layout.all_cores, kLogitsCbIndex, logits_format, logits_tile_bytes, streamed_tiles);
    if (has_mask) {
        create_circular_buffer(
            program, layout.all_cores, kMaskCbIndex, logits_format, logits_tile_bytes, streamed_tiles);
    }
    // Scores stay FP32: a bf16 round trip would quantize the very comparisons the argmax makes.
    create_circular_buffer(
        program, layout.all_cores, kScoresCbIndex, tt::DataFormat::Float32, score_tile_bytes, streamed_tiles);
    // The writer's output ring: 32 token ids, each in its own NOC-aligned slot.
    constexpr uint32_t kOutputSlotBytes = 32U;
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kOutputStagingCbIndex,
        tt::DataFormat::UInt32,
        tt::constants::TILE_HEIGHT * kOutputSlotBytes);

    // Boundary-row partials: receive slots plus one staging slot for the outgoing record.
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kRecordsCbIndex,
        tt::DataFormat::UInt32,
        (max_foreign_shards + 1U) * kRecordBytes);

    // Positions staging: one ALIGNED page per entry (the NOC requires the L1 destination to match
    // the DRAM alignment), sized for the larger core group's entry window -- a run of n tiles
    // spans at most (n - 1) / Wt + 2 entries. Sizing by num_entries instead would scale the L1 and
    // read footprint with the global batch rather than a core's share of it.
    if (layout.position_aware) {
        const uint32_t slot_bytes = static_cast<uint32_t>(tensor_args.positions->buffer()->aligned_page_size());
        const uint32_t max_local_entries =
            std::min(layout.num_entries, (layout.tiles_per_core_group_1 - 1U) / layout.Wt + 2U);
        for (auto cb_index : {kReaderPositionsCbIndex, kWriterPositionsCbIndex}) {
            create_circular_buffer_bytes(
                program, layout.all_cores, cb_index, tt::DataFormat::UInt32, max_local_entries * slot_bytes);
        }
    }

    // Kernels.
    auto* logits_buffer = logits.buffer();
    auto* mask_buffer = has_mask ? tensor_args.logits_mask->buffer() : nullptr;
    auto* output_buffer = output.buffer();

    // Greedy vs noisy is decided by uses_gumbel_noise, NOT `temperature > 0`: a positive
    // temperature whose reciprocal overflows float32 must build the greedy kernel. The program
    // hash uses the same predicate.
    const bool do_gumbel_noise = uses_gumbel_noise(args.temperature);

    // Compile-time args. Keep the leading count in step with TensorAccessorArgs<N> in each kernel:
    // the accessor offsets are hard-coded there and further accessors chain off them, so a
    // mismatch misdecodes the accessor words rather than failing to compile. Mode flags are
    // appended past the accessor chain, matching each kernel's read order. Nothing derived from
    // the token dimension may be a compile-time arg -- the normalized program hash depends on it.
    std::vector<uint32_t> reader_ct_args{layout.block_size, layout.Wt};
    tt::tt_metal::TensorAccessorArgs(logits_buffer).append_to(reader_ct_args);
    if (has_mask) {
        tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    // The null appends keep the accessor chain's length mode-independent.
    if (layout.position_aware) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.positions->buffer()).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    reader_ct_args.push_back(has_mask ? 1U : 0U);               // do_logits_mask
    reader_ct_args.push_back(layout.position_aware ? 1U : 0U);  // do_positions
    shared_vars.reader_kernel_id =
        create_reader_kernel(program, layout.all_cores, reader_ct_args, {}, kReaderKernelPath);

    // Each split row's owner counts its senders on its own copy of this semaphore.
    const uint32_t reduction_sem_id = tt::tt_metal::CreateSemaphore(program, layout.all_cores, 0);

    std::vector<uint32_t> writer_ct_args{
        layout.Wt,
        layout.logical_vocab,
        // Ht is dead in position mode but still hashed into the binary, so it is pinned to keep
        // the build token-independent. ONE, never zero: the dead fallback divides by Ht in code
        // that is still compiled, and a zero is a -Werror=div-by-zero build failure.
        layout.position_aware ? 1U : layout.Ht,
        reduction_sem_id,
        max_foreign_shards};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    if (layout.position_aware) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.positions->buffer()).append_to(writer_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(writer_ct_args);
    }
    writer_ct_args.push_back(layout.position_aware ? 1U : 0U);  // do_positions
    shared_vars.writer_kernel_id =
        create_writer_kernel(program, layout.all_cores, writer_ct_args, {}, kWriterKernelPath);

    // FLOAT32 logits (and the mask, validated to the same dtype) unpack straight into DST instead
    // of through the 19-bit SrcA registers, which would round them to TF32 and decide greedy
    // argmax near-ties differently from ttnn::argmax. Host-side only: the mode flips the generated
    // unpack formats that copy_tile's unpack-to-dest path gates on. bf16 fits losslessly in TF32
    // and stays on the default path. Built with a bespoke ComputeConfig because
    // create_compute_kernel does not expose unpack_to_dest_mode.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (logits.dtype() == tt::tt_metal::DataType::FLOAT32) {
        unpack_to_dest_mode[static_cast<size_t>(kLogitsCbIndex)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[static_cast<size_t>(kMaskCbIndex)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    auto create_gumbel_compute_kernel = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t tiles_per_core) {
        return tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .math_approx_mode = false,
                .compile_args = {tiles_per_core, layout.block_size, has_mask ? 1U : 0U, do_gumbel_noise ? 1U : 0U}});
    };
    shared_vars.compute_kernel_group_1_id =
        create_gumbel_compute_kernel(layout.core_group_1, layout.tiles_per_core_group_1);

    if (!layout.core_group_2.ranges().empty()) {
        shared_vars.compute_kernel_group_2_id =
            create_gumbel_compute_kernel(layout.core_group_2, layout.tiles_per_core_group_2);
    }

    // Runtime args.
    const auto [inv_temperature_bits, positions_address, mask_entry_stride] = derive_runtime_args(args, tensor_args);

    shared_vars.core_info.reserve(layout.num_cores);
    for (const auto& [core, core_index, num_tiles, start_tile, in_group_1] : work) {
        SetRuntimeArgs(
            program,
            shared_vars.reader_kernel_id,
            core,
            {logits_buffer->address(),
             has_mask ? mask_buffer->address() : 0U,
             num_tiles,
             start_tile,
             layout.Ht,
             positions_address,
             layout.logical_tokens,
             mask_entry_stride});

        SetRuntimeArgs(
            program,
            shared_vars.writer_kernel_id,
            core,
            {output_buffer->address(),
             num_tiles,
             start_tile,
             positions_address,
             send_routing[core_index][0],
             send_routing[core_index][1],
             send_routing[core_index][2],
             expected_shards[core_index],
             layout.logical_tokens});

        const uint32_t stream_id = rand_stream_id(layout, device_index, start_tile);
        SetRuntimeArgs(
            program,
            in_group_1 ? shared_vars.compute_kernel_group_1_id : shared_vars.compute_kernel_group_2_id,
            core,
            {args.seed, kRandFromBits, kRandScaleBits, inv_temperature_bits, stream_id});

        shared_vars.core_info.push_back({core, stream_id, in_group_1});
    }

    shared_vars.has_compute_group_2 = !layout.core_group_2.ranges().empty();

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
    const auto layout = compute_layout(logits, tensor_args.positions.has_value());

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
    const bool has_mask = tensor_args.logits_mask.has_value();

    // Deliberately NO compute_layout / core_layout here: the work split is a function of hashed
    // quantities only and was cached in the shared variables at build; this op dispatches once per
    // generated token, so re-deriving it would be paid on every dispatch. The only layout values
    // that can differ under the same program hash are the token-dimension pair below (position
    // mode normalizes the token dim away), and both are plain shape reads.
    const uint32_t Ht = logits.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t logical_tokens = logits.logical_shape()[-2];

    const uint32_t logits_address = logits.buffer()->address();
    const uint32_t mask_address = has_mask ? tensor_args.logits_mask->buffer()->address() : 0U;
    const uint32_t output_address = tensor_return_value.buffer()->address();

    // seed and temperature are runtime-only (excluded from the program hash so changing either
    // reuses the cached program) and must be re-applied on every hit, from the SAME derivation the
    // build used. A temperature whose kernel selection changed hashes to a different program, so a
    // cached program is never patched with the wrong variant's args.
    const auto [inv_temperature_bits, positions_address, mask_entry_stride] =
        derive_runtime_args(operation_attributes, tensor_args);

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& vars = cached_workload.shared_variables.at(coord_range);

        auto& reader_args = GetRuntimeArgs(program, vars.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, vars.writer_kernel_id);
        auto& compute_g1_args = GetRuntimeArgs(program, vars.compute_kernel_group_1_id);
        auto& compute_g2_args =
            vars.has_compute_group_2 ? GetRuntimeArgs(program, vars.compute_kernel_group_2_id) : compute_g1_args;

        // The merge routing and per-core tile runs are split properties, identical on every
        // dispatch, and are not re-patched. Everything patched below is either a buffer address or
        // token-derived: replayed stale, a cached program would read real but WRONG rows/pages --
        // in bounds, no fault, silently wrong samples.
        for (const auto& info : vars.core_info) {
            const auto& core = info.core;
            {
                auto& core_args = reader_args[core.x][core.y];
                core_args[kReaderLogitsBufferIdx] = logits_address;
                core_args[kReaderMaskBufferIdx] = mask_address;
                core_args[kReaderHtIdx] = Ht;
                core_args[kReaderPositionsBufferIdx] = positions_address;
                core_args[kReaderLogicalTokensIdx] = logical_tokens;
                core_args[kReaderMaskStrideIdx] = mask_entry_stride;
            }
            {
                auto& core_args = writer_args[core.x][core.y];
                core_args[kWriterOutputBufferIdx] = output_address;
                core_args[kWriterPositionsBufferIdx] = positions_address;
                core_args[kWriterLogicalTokensIdx] = logical_tokens;
            }
            {
                auto& core_args =
                    info.in_compute_group_1 ? compute_g1_args[core.x][core.y] : compute_g2_args[core.x][core.y];
                core_args[kComputeSeedIdx] = operation_attributes.seed;
                core_args[kComputeInvTemperatureIdx] = inv_temperature_bits;
                // Baked at build from the same split this loop iterates; see CoreRuntimeInfo.
                core_args[kComputeRandStreamIdx] = info.rand_stream_id;
            }
        }
    }
}

}  // namespace ttml::metal::ops::gumbel_sample::device
