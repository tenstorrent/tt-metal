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
// compute runtime arg slots
constexpr uint32_t kComputeSeedIdx = 0U;
// Slots 1 and 2 hold the rand from/scale bits -- process constants set once at build and never
// re-patched on cache hits, so nothing reads these indices; they stay to document the layout.
[[maybe_unused]] constexpr uint32_t kComputeRandFromIdx = 1U;
[[maybe_unused]] constexpr uint32_t kComputeRandScaleIdx = 2U;
constexpr uint32_t kComputeInvTemperatureIdx = 3U;
constexpr uint32_t kComputeRandStreamIdx = 4U;

constexpr auto kLogitsCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskCbIndex = tt::CBIndex::c_1;
constexpr auto kScoresCbIndex = tt::CBIndex::c_2;
constexpr auto kOutputStagingCbIndex = tt::CBIndex::c_3;
constexpr auto kRecordsCbIndex = tt::CBIndex::c_4;

// Boundary-row partials exchanged between cores: [valid, row, 32 maxima, 32 indices]
// padded to 288 bytes; valid and row are watcher/debug breadcrumbs the merge never reads. Each
// split row is merged by the core holding its FIRST tile, so a core sends at most one record (its
// first row's shard) and receives at most the row's shard fan-in -- see writer_gumbel_sample.cpp.
constexpr uint32_t kRecordBytes = 72U * sizeof(uint32_t);

// The reader carries Ht as a runtime arg at slot 4 (see reader_gumbel_sample.cpp), the writer does
// not, so the positions BUFFER ADDRESS lands at a different slot in each kernel. These two MUST
// stay independent -- equalizing them would make one kernel read a neighbouring arg as the
// positions address.
constexpr uint32_t kReaderHtIdx = 4U;
constexpr uint32_t kReaderPositionsBufferIdx = 5U;
constexpr uint32_t kWriterPositionsBufferIdx = 3U;
static_assert(kReaderPositionsBufferIdx == kReaderHtIdx + 1U);
// The writer's merge routing (owner x/y, send slot, expected shards) occupies the four slots after
// the positions address.
constexpr uint32_t kWriterMergeRoutingArgs = 4U;
// Logical token count, appended LAST in both kernels so every earlier slot keeps its index. It
// bounds the position clamp in both kernels (they consume disjoint bit fields of the SAME clamped
// value, so both need it). A RUNTIME arg for the same reason Ht is: it derives from the token
// dimension, which the program hash normalizes away in position mode -- as a compile-time arg it
// would put every prompt length back on the JIT-miss path.
constexpr uint32_t kReaderLogicalTokensIdx = 6U;
// Per-entry mask-page stride: 0 for [1, 1, 1, V], Wt for a [B, 1, 1, V] per-row mask. Runtime so
// both shapes share one program; re-derived from the mask's shape on every dispatch.
constexpr uint32_t kReaderMaskStrideIdx = 7U;
constexpr uint32_t kWriterLogicalTokensIdx = 8U;
static_assert(kReaderLogicalTokensIdx == kReaderPositionsBufferIdx + 1U);
static_assert(kReaderMaskStrideIdx == kReaderLogicalTokensIdx + 1U);
static_assert(kWriterLogicalTokensIdx == kWriterPositionsBufferIdx + kWriterMergeRoutingArgs + 1U);

// Per-entry token positions live in a small device TENSOR, not in runtime args. Each core stages the
// whole local list into L1 once at kernel start (slots are indexed by absolute entry id, so the full
// list keeps the addressing uniform even though only local entries are consumed).
constexpr auto kReaderPositionsCbIndex = tt::CBIndex::c_5;
constexpr auto kWriterPositionsCbIndex = tt::CBIndex::c_6;

// Uniform draw bounds for the Gumbel transform g = -log(-log(U)).
//
// Lower bound 2^-32 caps the noise at g <= -log(-log(2^-32)) ~ -3.1 on the low side.
//
// The UPPER bound : `rand_tile` produces values on a CLOSED interval [from, from + scale],
// so U == 1.0 is attainable, and then log(1) = 0, -log(0) = +inf, g = +inf, which pins the
// argmax onto that token with certainty. It is a ~2^-32-per-element event, but it is a real one,
// so here the top of the range is the largest float32 strictly below 1.0 and g stays FINITE --
// that finiteness is the point of the bound. The ceiling itself is ~16.6 with an exact log; the
// approximate log in gumbel_sfpu.h caps it lower, near 13.81.
//
// gumbel_sfpu.h's approximate log drops its zero guard on the strength of exactly these bounds --
// change them only together with that header.
constexpr float kGumbelUniformLowerBound = 0x1p-32F;
const float kGumbelUniformUpperBound = ttnn::operations::uniform::largest_supported_float32_below(1.0F);

// `rand_tile` is documented as inclusive of `from + scale`. Shrink the scale by one ULP if rounding
// would push the top of the range past the intended upper bound. This mirrors the guard in the
// uniform op's DEVICE kernel (compute_uniform.cpp) -- there is no host-side header that ships it,
// and this op cannot reuse the kernel's copy because its compute kernel takes `from` and `scale`
// as runtime args (the scale must be computed here, on the host) rather than the two endpoints.
uint32_t compute_rand_scale_bits(float lower, float upper) {
    float scale = upper - lower;
    uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (lower + scale > upper && scale_bits != 0U) {
        --scale_bits;
    }
    return scale_bits;
}

// Derived once per process, consumed by the cache-miss build. Deliberately NOT re-derived (or even
// re-patched) on cache hits: the bounds are process constants, and a second derivation site is a
// divergence trap -- a drift between the two would manifest only on cache hits, which single-shape
// unit tests never exercise.
const uint32_t kRandFromBits = std::bit_cast<uint32_t>(kGumbelUniformLowerBound);
const uint32_t kRandScaleBits = compute_rand_scale_bits(kGumbelUniformLowerBound, kGumbelUniformUpperBound);

// Linear index of this device among the SEEDED (data-parallel) mesh axes only. Devices that differ
// solely on a replicated axis get the same index -- and therefore the same RNG stream -- which is
// what keeps a tensor-parallel replica group in sync.
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
    // leaving most of it idle while each active core carried a whole vocabulary.
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

    // -------------------------------------------------------------------------
    // Split-row merge routing. The owner of a row is the core holding its FIRST tile; the split
    // hands each core one contiguous tile range, so a core can hold a foreign shard only of its
    // first row (=> at most one record to send) and can own a split row only as its last (=> one
    // wait). Senders to a given owner are enumerated in core order, which hands each one a
    // collision-free slot in the owner's records CB. Derived from the same core_layout as the
    // work-split runtime args, so the two can never disagree.
    // -------------------------------------------------------------------------
    const auto work = core_layout(layout);
    std::vector<uint32_t> expected_shards(layout.num_cores, 0U);
    std::vector<std::array<uint32_t, 3U>> send_routing(layout.num_cores, {0U, 0U, 0U});  // x, y, slot
    for (uint32_t sender = 1U; sender < layout.num_cores; ++sender) {
        if (work[sender].start_tile % layout.Wt == 0U) {
            continue;  // first row starts here: nothing to send
        }
        const uint32_t row_first_tile = (work[sender].start_tile / layout.Wt) * layout.Wt;
        uint32_t owner = sender - 1U;
        while (work[owner].start_tile > row_first_tile) {
            --owner;
        }
        const auto owner_phys = logits.device()->worker_core_from_logical_core(work[owner].core);
        send_routing[sender] = {
            static_cast<uint32_t>(owner_phys.x), static_cast<uint32_t>(owner_phys.y), expected_shards[owner]};
        ++expected_shards[owner];
    }
    const uint32_t max_foreign_shards = *std::max_element(expected_shards.begin(), expected_shards.end());
    // Every sender's run BEGINS strictly inside the owned row's Wt tiles, and consecutive starts
    // are at least the smaller group's tile count apart, which bounds the fan-in by the split
    // rather than the core count. The CB is sized from the exact fan-in above; this guards the
    // derivation against a work-split change.
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

    // -------------------------------------------------------------------------
    // Circular buffers. Peak L1 is a handful of tiles regardless of V: this is the whole point of
    // the fusion -- avoiding materializing several full [B, 1, tokens, V] tensors in DRAM.
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
    // Staging for the writer's output ring: 32 token ids, each in its own NOC-aligned slot (see
    // kOutputSlotBytes in the writer kernel). The writer rotates through the slots and barriers
    // only when recycling one (and once at kernel end), so up to 32 page writes ride behind each
    // flush regardless of how the rows that produced them were grouped.
    constexpr uint32_t kOutputSlotBytes = 32U;
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kOutputStagingCbIndex,
        tt::DataFormat::UInt32,
        tt::constants::TILE_HEIGHT * kOutputSlotBytes);

    // Boundary-row partials: `max_foreign_shards` receive slots for the one row this core may own,
    // plus one staging slot for the record it may send. Sized by the actual split fan-in (a few
    // records), never by the core count.
    create_circular_buffer_bytes(
        program,
        layout.all_cores,
        kRecordsCbIndex,
        tt::DataFormat::UInt32,
        (max_foreign_shards + 1U) * kRecordBytes);

    // Positions staging. One ALIGNED page per entry, not four packed bytes: a DRAM read moves a
    // whole aligned page, and the NOC requires the L1 destination to match the DRAM alignment
    // (64 B on Blackhole) rather than L1's own -- see the alignment_mask logic in
    // hw/inc/internal/debug/sanitize.h.
    //
    // Each core stages only the entry WINDOW its contiguous tile run touches (see PositionWindow
    // in dataflow_utils.hpp): a run of n tiles spans at most (n - 1) / Wt + 2 entries. Sized by
    // the LARGER core group so one CB config serves both; cores with smaller windows leave the
    // tail unused. Sizing by num_entries instead would make the positions footprint -- L1 bytes
    // AND per-core staging page reads, in BOTH kernels -- scale with the global batch rather than
    // a core's share of it.
    if (layout.position_aware) {
        const uint32_t slot_bytes = static_cast<uint32_t>(tensor_args.positions->buffer()->aligned_page_size());
        const uint32_t max_local_entries =
            std::min(layout.num_entries, (layout.tiles_per_core_group_1 - 1U) / layout.Wt + 2U);
        for (auto cb_index : {kReaderPositionsCbIndex, kWriterPositionsCbIndex}) {
            create_circular_buffer_bytes(
                program, layout.all_cores, cb_index, tt::DataFormat::UInt32, max_local_entries * slot_bytes);
        }
    }

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    auto* logits_buffer = logits.buffer();
    auto* mask_buffer = has_mask ? tensor_args.logits_mask->buffer() : nullptr;
    auto* output_buffer = output.buffer();

    // Greedy vs noisy is decided by uses_gumbel_noise, NOT a bare `temperature > 0`: a positive
    // temperature whose reciprocal overflows float32 (below ~2.9e-39) must build the greedy kernel,
    // or the +inf scale factor collapses every positive logit to one bit pattern and the argmax
    // degenerates to "first positive column". The hash uses the same predicate.
    const bool do_gumbel_noise = uses_gumbel_noise(args.temperature);

    // Mode flags (mask / noise / positions) travel as compile-time args rather than -D defines.
    // The two are equivalent to the JIT -- both are -D macros on the compile line, both hashed
    // into the kernel-binary identity -- so this is purely about keeping every compile-time input
    // in one channel. Each kernel reads its flags PAST its accessor chain, at
    // next_compile_time_args_offset(), so the hand-numbered accessor offsets never move when a
    // flag is added; the appends below must match each kernel's read order.

    // Ht is NOT here: it is a runtime arg, so that one program serves every prompt length. Keep this
    // count in step with TensorAccessorArgs<N> in reader_gumbel_sample.cpp -- the accessor offset is
    // hard-coded there and the mask accessor chains off it, so a mismatch misdecodes the accessor
    // words (page size read as the config flags) instead of failing to compile.
    // num_entries is NC = padded_shape[0] * padded_shape[1] -- token-INDEPENDENT, so it adds no new
    // cache-miss source. Anything derived from the token dimension must never land in a compile-time
    // arg here; that is what the normalized program hash depends on.
    std::vector<uint32_t> reader_ct_args{layout.block_size, layout.Wt, layout.num_entries};
    tt::tt_metal::TensorAccessorArgs(logits_buffer).append_to(reader_ct_args);
    if (has_mask) {
        tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    // The null append is mandatory in the non-position case: without it the accessor chain's length
    // becomes mode-dependent and the next accessor misdecodes its page size as the config flags.
    if (layout.position_aware) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.positions->buffer()).append_to(reader_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(reader_ct_args);
    }
    reader_ct_args.push_back(has_mask ? 1U : 0U);               // do_logits_mask
    reader_ct_args.push_back(layout.position_aware ? 1U : 0U);  // do_positions
    shared_vars.reader_kernel_id =
        create_reader_kernel(program, layout.all_cores, reader_ct_args, {}, kReaderKernelPath);

    // Each split row's owner counts its senders on its own copy of this semaphore; cores that own
    // nothing never wait on it.
    const uint32_t reduction_sem_id = tt::tt_metal::CreateSemaphore(program, layout.all_cores, 0);

    // Keep this count in step with TensorAccessorArgs<6> in writer_gumbel_sample.cpp -- the
    // accessor offset is hard-coded there and the positions accessor chains off it, so a mismatch
    // misdecodes the accessor words (page size read as the config flags) instead of failing to
    // compile.
    std::vector<uint32_t> writer_ct_args{
        layout.Wt,
        layout.logical_vocab,
        // logical_tokens is NOT here: it rides as a runtime arg in both modes (see
        // kWriterLogicalTokensIdx) -- in position mode it must stay out of the program-cache key,
        // and in non-position mode its uses are multiplies and compares that gain nothing from
        // being constexpr. Ht cannot follow it: the writer divides by Ht, which folds to
        // shift/multiply only for a compile-time constant.
        //
        // Ht is dead in position mode -- its only uses sit past unconditional returns -- but a
        // compile-time arg is hashed into the kernel binary whether it is read or not, so it is
        // pinned to keep the build independent of the token dimension. ONE, never zero: the dead
        // fallback divides by Ht in code that is still compiled, and the JIT builds with
        // -Wall -Werror, so a zero here is a -Werror=div-by-zero build failure. At 1 the dead path
        // degenerates to exactly what the position path does, which keeps it harmless if the guard
        // is ever refactored away.
        layout.position_aware ? 1U : layout.Ht,
        reduction_sem_id,
        max_foreign_shards,
        layout.num_entries};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    if (layout.position_aware) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.positions->buffer()).append_to(writer_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(writer_ct_args);
    }
    writer_ct_args.push_back(layout.position_aware ? 1U : 0U);  // do_positions
    shared_vars.writer_kernel_id =
        create_writer_kernel(program, layout.all_cores, writer_ct_args, {}, kWriterKernelPath);

    const std::vector<uint32_t> compute_ct_args_g1{
        layout.tiles_per_core_group_1, layout.block_size, has_mask ? 1U : 0U, do_gumbel_noise ? 1U : 0U};
    shared_vars.compute_kernel_group_1_id = create_compute_kernel(
        program, layout.core_group_1, compute_ct_args_g1, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    if (!layout.core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_ct_args_g2{
            layout.tiles_per_core_group_2, layout.block_size, has_mask ? 1U : 0U, do_gumbel_noise ? 1U : 0U};
        shared_vars.compute_kernel_group_2_id = create_compute_kernel(
            program, layout.core_group_2, compute_ct_args_g2, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // -------------------------------------------------------------------------
    // Runtime args
    // -------------------------------------------------------------------------
    // Guard the reciprocal: only computed when the noisy kernel will read it. uses_gumbel_noise
    // guarantees the reciprocal is FINITE here (that is the predicate's whole point); greedy gets a
    // zero because an inf sitting in a runtime arg is a trap for anyone who later makes it read it.
    const uint32_t inv_temperature_bits = do_gumbel_noise ? std::bit_cast<uint32_t>(1.0F / args.temperature) : 0U;

    // Zero when absent: the slot exists in BOTH modes so override_runtime_arguments can patch it
    // unconditionally, exactly as it does for Ht.
    const uint32_t positions_address = layout.position_aware ? tensor_args.positions->buffer()->address() : 0U;
    const uint32_t mask_entry_stride = (has_mask && tensor_args.logits_mask->logical_shape()[0] > 1U) ? layout.Wt : 0U;

    shared_vars.core_info.reserve(layout.num_cores);
    for (uint32_t core_index = 0U; core_index < layout.num_cores; ++core_index) {
        const auto& [core, num_tiles, start_tile] = work[core_index];
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

        const bool in_group_1 = layout.core_group_1.contains(core);
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

    // Deliberately NO compute_layout / core_layout here: the work split and everything derived
    // from it (per-core tile runs, merge routing, RNG stream ids, core-group membership) is a
    // function of hashed quantities only, so it was derived once at build time and cached in the
    // shared variables. This op dispatches once per generated token; re-deriving the split would
    // pay a device grid query, split_work_to_cores and per-core CoreRangeSet scans on every one of
    // those dispatches. The only layout values that CAN differ under the same program hash are the
    // token-dimension pair below (position mode normalizes the token dim away), and they are plain
    // shape reads.
    const uint32_t Ht = logits.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t logical_tokens = logits.logical_shape()[-2];

    const uint32_t logits_address = logits.buffer()->address();
    const uint32_t mask_address = has_mask ? tensor_args.logits_mask->buffer()->address() : 0U;
    const uint32_t output_address = tensor_return_value.buffer()->address();
    const uint32_t positions_address =
        tensor_args.positions.has_value() ? tensor_args.positions->buffer()->address() : 0U;
    // Wt derived from the logits shape, same as Ht above -- this function deliberately avoids
    // recomputing the full layout on cache hits.
    const uint32_t mask_entry_stride =
        (tensor_args.logits_mask.has_value() && tensor_args.logits_mask->logical_shape()[0] > 1U)
            ? (logits.padded_shape()[-1] / tt::constants::TILE_WIDTH)
            : 0U;

    // seed and temperature are runtime-only (deliberately excluded from the program hash so that
    // changing either reuses the cached program), so they must be re-applied on every cache hit
    // alongside the buffer addresses. The guard is uses_gumbel_noise, matching build_program and
    // the hash: it keeps the reciprocal finite, and a temperature whose kernel selection CHANGED
    // (crossing zero or the reciprocal-overflow floor) hashes to a different program anyway, so a
    // cached program is never patched with the wrong variant's args. The rand from/scale bits are
    // process constants baked at build time (kRandFromBits/kRandScaleBits) and are not re-patched.
    const uint32_t inv_temperature_bits = uses_gumbel_noise(operation_attributes.temperature)
                                              ? std::bit_cast<uint32_t>(1.0F / operation_attributes.temperature)
                                              : 0U;

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& vars = cached_workload.shared_variables.at(coord_range);

        auto& reader_args = GetRuntimeArgs(program, vars.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, vars.writer_kernel_id);
        auto& compute_g1_args = GetRuntimeArgs(program, vars.compute_kernel_group_1_id);
        auto& compute_g2_args =
            vars.has_compute_group_2 ? GetRuntimeArgs(program, vars.compute_kernel_group_2_id) : compute_g1_args;

        // The merge routing (owner coords, slot, expected shard count) and the per-core tile runs
        // are not re-patched here: they are properties of the work split, which is identical on
        // every dispatch. Only the buffer addresses, the token-dimension pair, the seed and the
        // temperature change.
        for (const auto& info : vars.core_info) {
            const auto& core = info.core;
            {
                auto& core_args = reader_args[core.x][core.y];
                core_args[kReaderLogitsBufferIdx] = logits_address;
                core_args[kReaderMaskBufferIdx] = mask_address;
                // Ht is runtime-only (deliberately out of the program hash in position mode) so it
                // must be re-applied on every cache hit. This is unconditional only because the
                // reader emits the slot in BOTH modes. Were it appended for position mode alone,
                // decode's arg vector would stop at four words and this line would write one past
                // the end -- and RuntimeArgsData's bounds check is a TT_ASSERT that compiles away in
                // Release, so it would land silently in the packed dispatch command.
                core_args[kReaderHtIdx] = Ht;
                // The positions BUFFER moves between dispatches (each prefill builds a new one), and
                // a cached program replayed against a stale address reads whatever DRAM now occupies
                // that region -- in bounds, no fault, a plausible-looking token. This patch prevents that.
                core_args[kReaderPositionsBufferIdx] = positions_address;
                // Like Ht: runtime-only and token-derived, so a cached program replayed without
                // this patch would clamp positions against a STALE token count -- either rejecting
                // valid rows or readmitting the padding band the clamp exists to keep out.
                core_args[kReaderLogicalTokensIdx] = logical_tokens;
                // Re-derived per dispatch: the SAME cached program serves both mask shapes, so a
                // dispatch that switches between a shared and a per-row mask must re-patch the stride
                // or the reader would walk the wrong mask pages -- in bounds, silently wrong rows.
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
