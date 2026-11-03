// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "metal/common/const_utils.hpp"
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
constexpr uint32_t kReaderAddressArgs = kReaderArgGate + 1U;

// The program config in tiles. rows = Sq_chunk_t * heads_per_item tile-rows make one work item.
struct SDPAForwardChunks {
    uint32_t Sq_chunk_t;
    uint32_t Sk_chunk_t;
    uint32_t heads_per_item;

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
constexpr uint32_t kNumSDPAForwardCbs = 17U;

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
    uint32_t reduction_scaler;
    uint32_t matmul_reduce;
};

SDPAForwardCbTiles cb_tiles(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);

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
// of runs of consecutive items (metal/common/sdpa_work_split.hpp, shared with sdpa_bw).
using ttml::metal::CoreItemRuns;
using ttml::metal::ItemRun;

uint32_t num_work_items(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);

// Light/heavy chunk pairs of a causal shape with an even chunk count (so every pair is the same work); 0
// when the shape does not pair.
uint32_t num_causal_pairs(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);

// What the split hands out whole, as on main: causal pairs when there are at least num_available_cores of
// them (so no core idles for the balance), items otherwise. The program runs on min(available, units) cores.
uint32_t num_work_units(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t num_available_cores);

// Per-core item runs for num_cores <= num_work_units cores: causal pairs when num_work_units pairs, otherwise
// contiguous blocks of items. Throws if the paired lists would not fit the runtime args (they cannot today).
std::vector<CoreItemRuns> split_work(
    const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t num_cores, uint32_t num_available_cores);

// Cheap knob checks for validate: each set knob must divide its axis, and the config must fit l1_budget_bytes
// with every unset chunk size at its one-tile floor (the smallest chunking the pick can reach). Throws otherwise.
void check_chunks(const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape, uint64_t l1_budget_bytes);

// check_chunks, then picks the unset knobs by a shape rule, shrinking them until they fit l1_budget_bytes
// and give num_available_cores a causal pair (or an item) each. Throws when even the smallest chunks do not fit.
SDPAForwardChunks resolve_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t num_available_cores);

}  // namespace ttml::metal::ops::sdpa_fw::device
