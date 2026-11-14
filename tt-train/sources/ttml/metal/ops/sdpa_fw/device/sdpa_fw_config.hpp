// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "metal/common/const_utils.hpp"
#include "metal/common/sdpa_item_math.hpp"
#include "metal/common/sdpa_work_split.hpp"
#include "metal/ops/sdpa_fw/sdpa_fw_program_config.hpp"
#include "metal/ttnn_all_includes.hpp"

// Everything the SDPA forward program depends on besides the kernels themselves: how the user knobs
// turn into chunk sizes, how much L1 the circular buffers take and how the work items are shared out.
// Shared by validate (early errors) and the program factory (which also serves the ring SDPA path).
namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardShape {
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t St;   // sequence tiles
    uint32_t qWt;  // Q/K inner-dim tiles
    uint32_t vWt;  // V inner-dim tiles
    AttentionMaskType mask_type;
    bool has_gate;
    bool return_intermediates;
};

// The shape as the kernels see it (padded dims). The L1 the circular buffers may use on a core is
// available_l1_bytes(device) from metal/common/program_utils.hpp.
SDPAForwardShape shape_of(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    AttentionMaskType mask_type,
    bool has_gate,
    bool return_intermediates);

// Reader runtime args: the buffer addresses first, then the work list.
constexpr uint32_t kReaderArgQuery = 0U;
constexpr uint32_t kReaderArgKey = 1U;
constexpr uint32_t kReaderArgValue = 2U;
constexpr uint32_t kReaderArgMask = 3U;
constexpr uint32_t kReaderArgGate = 4U;
constexpr uint32_t kReaderArgScratch = 5U;  // partials scratch (split units)
constexpr uint32_t kReaderAddressArgs = kReaderArgScratch + 1U;

// The program config in tiles. rows = Sq_chunk_t * heads_per_item tile-rows make one work item.
struct SDPAForwardChunks {
    uint32_t Sq_chunk_t;
    uint32_t Sk_chunk_t;
    uint32_t heads_per_item;
    // False when the split units' circular buffers do not fit L1 beside the smallest chunking the knobs allow
    // (wide heads: D >= 704 at d_v = d_qk with nothing pinned, 1.3 MB budget): the schedule then never cuts.
    bool allow_split = true;

    uint32_t rows() const {
        return Sq_chunk_t * heads_per_item;
    }
};

// The program's circular buffers; the kernels use the same indices.
constexpr auto kQueryCbIndex = tt::CBIndex::c_0;
constexpr auto kKeyCbIndex = tt::CBIndex::c_1;
constexpr auto kValueCbIndex = tt::CBIndex::c_2;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_3;      // causal: two on-chip tiles; arbitrary: one step's mask tiles
constexpr auto kIntermediateCbIndex = tt::CBIndex::c_4;  // lse, one fp32 tile per row
constexpr auto kReductionScalerCbIndex = tt::CBIndex::c_5;   // the ones tile of the row reductions
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;      // the ones-column tile of the matmul row reduce, lse mask
constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_7;  // one step's scores, rewritten in place by softmax
// Online-softmax state, one tile per item row; prev/cur swap roles every K step.
constexpr auto kPrevMaxValueCbIndex = tt::CBIndex::c_8;
constexpr auto kCurMaxValueCbIndex = tt::CBIndex::c_9;
constexpr auto kExpMaxDiffCbIndex = tt::CBIndex::c_10;  // exp(scale * (prev max - cur max)), the rescale factor
constexpr auto kPrevSumExpCbIndex = tt::CBIndex::c_11;
constexpr auto kCurSumExpCbIndex = tt::CBIndex::c_12;
constexpr auto kPrevMmOutCbIndex = tt::CBIndex::c_13;  // running P @ V accumulator
constexpr auto kCurMmOutCbIndex = tt::CBIndex::c_14;
constexpr auto kOutputCbIndex = tt::CBIndex::c_15;
constexpr auto kGateCbIndex = tt::CBIndex::c_16;
// Split units: a sender's running state goes out through c_18 to its scratch slot, a merger's partials
// come in through c_17; c_19 holds the partial's rescale factor while it is folded in.
constexpr auto kPartialInCbIndex = tt::CBIndex::c_17;
constexpr auto kPartialOutCbIndex = tt::CBIndex::c_18;
constexpr auto kPartialAlphaCbIndex = tt::CBIndex::c_19;
constexpr uint32_t kNumSDPAForwardCbs = 20U;

// Circular-buffer depths in tiles; 0 = the CB is not created.
struct SDPAForwardCbTiles {
    uint32_t query;
    uint32_t key;
    uint32_t value;
    uint32_t attn_mask;
    uint32_t gate;
    uint32_t intermediate;
    uint32_t attention_weights;
    uint32_t stat;    // each of max prev/cur, exp max diff, sum prev/cur
    uint32_t mm_out;  // each of prev/cur
    uint32_t output;
    uint32_t reduction_scaler;  // the ones tile of the row reductions
    uint32_t matmul_reduce;     // the ones-column tile of the matmul row reduce and the lse mask
    uint32_t partial;           // each of partial in / partial out (split units): four rows, two per NoC batch
    uint32_t alpha;             // the partial's rescale factor (split units)
};

// with_partials: the CBs of split units (sender / merger) are present.
SDPAForwardCbTiles cb_tiles(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, bool with_partials);

// One circular buffer as the program creates it. cb_plan is the one place that says which CBs are bf16
// and which fp32: the factory creates the CBs from it and l1_bytes sums it, so the fit check cannot drift
// from the program.
struct SDPAForwardCb {
    tt::CBIndex index;
    tt::DataFormat format;
    uint32_t tiles;
};
std::array<SDPAForwardCb, kNumSDPAForwardCbs> cb_plan(const SDPAForwardCbTiles& tiles);
uint64_t l1_bytes(const SDPAForwardCbTiles& tiles);

// Work item = one Q chunk of one head stack, see kernels/sdpa_fw_work_list.hpp. Each core gets a list
// of units: a run of whole items, or one K/V step range of one item that is sent to / merged on a core.
using UnitRole = sdpa_items::UnitRole;

struct WorkUnit {
    uint32_t item = 0;
    UnitRole role = UnitRole::Whole;
    uint32_t num_items = 1;   // Whole: run length
    uint32_t step_begin = 0;  // Sender / Merger
    uint32_t step_end = 0;
    uint32_t num_partials = 0;  // Merger
    uint32_t sem_idx = 0;       // Merger: semaphore on its core the senders bump
    uint32_t slot = 0;          // Sender: scratch slot; Merger: first of its partials' slots
    uint32_t merger_core = 0;   // Sender: logical core index of its merger
};
using CoreUnits = std::vector<WorkUnit>;
constexpr uint32_t kWordsPerUnit = 3U;

// Runtime-arg words of one unit (layout in metal/common/sdpa_item_math.hpp); merger_noc_x/y = the merger core's NoC
// coordinates (senders only). Throws when a field does not fit its width.
void append_unit_words(
    std::vector<uint32_t>& words, const WorkUnit& unit, uint32_t merger_noc_x, uint32_t merger_noc_y);

uint32_t num_work_items(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);
// K/V steps item attends to: causal up to its diagonal, else all.
uint32_t num_kv_steps(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t item);

constexpr uint32_t kMaxPieces = sdpa_items::unit_field_max(sdpa_items::kUnitPartialsBits);  // 63
constexpr uint32_t kMaxMergersPerCore = sdpa_items::kMaxMergersPerCore;
constexpr uint32_t kDefaultMaxPieces = 4U;

uint32_t partial_tiles_per_row(const SDPAForwardShape& shape);  // a sender's row: max, sum, output tiles

struct SDPAForwardSchedule {
    std::vector<CoreUnits> per_core;
    uint32_t num_slots = 0;     // scratch slots (one per sender)
    double predicted_us = 0.0;  // the cost model's makespan: the busiest core's units, merges after their partials
};

// Work for num_cores cores. Causal: items heavier than a core's fair share are cut into up to
// max_pieces step ranges (senders + one merger), then all pieces go longest first to the least loaded
// core; every core runs its senders first, then its whole items, then its mergers, so a merger never
// waits for a piece behind another merger. The cut schedule is kept only when the cost model predicts it
// faster than the same items uncut (with items at one to two per core the mergers land on cores the whole
// items already fill, and cutting loses). Everything else: contiguous blocks. When the lists would not fit
// the runtime args, tries one piece per item, then heavy/light chunk pairs, then contiguous blocks.
// causal_pairs skips longest first: causal items go straight to the heavy/light chunk pairs (main's
// placement), or to contiguous blocks when the pairs do not apply; max_pieces is then 1.
SDPAForwardSchedule make_schedule(
    const SDPAForwardShape& shape,
    const SDPAForwardChunks& chunks,
    uint32_t num_cores,
    uint32_t max_pieces,
    bool causal_pairs = false);

// 1 with causal_pairs, else the knob (0 = kDefaultMaxPieces).
uint32_t effective_max_pieces(const SDPAForwardProgramConfig& config);
bool may_split(const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape);
uint32_t effective_num_cores(const SDPAForwardProgramConfig& config, uint32_t grid_cores);

// make_schedule for config on a grid of grid_cores, on all of the config's cores (the idle ones are dropped);
// one piece when chunks.allow_split is false. What the program factory builds; the tests and the benchmark
// reproduce it.
SDPAForwardSchedule schedule_for(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    const SDPAForwardChunks& chunks,
    uint32_t grid_cores);

// Cheap knob checks for validate: each set knob must divide its axis, and the config must fit l1_budget_bytes
// with every unset chunk size at its one-tile floor (the smallest chunking the pick can reach), without the
// split units' CBs: resolve_chunks drops the cuts when those do not fit beside the floor. Throws otherwise.
void check_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t grid_cores);

// check_chunks, then picks the unset knobs by a shape rule (four rows per item; up to four heads of a K/V
// group stacked when there is a mask tensor, one otherwise; four or eight K/V tiles per step), shrinking them
// until they fit l1_budget_bytes (with the split units' CBs when the schedule cuts) and keep the config's
// cores busy. Throws when even the smallest config does not fit.
SDPAForwardChunks resolve_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t grid_cores);

}  // namespace ttml::metal::ops::sdpa_fw::device
