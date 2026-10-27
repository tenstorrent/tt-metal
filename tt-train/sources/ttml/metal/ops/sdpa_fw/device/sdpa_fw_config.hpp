// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "metal/common/const_utils.hpp"
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

// The shape as the kernels see it (padded dims), and the L1 the circular buffers may use on a core.
SDPAForwardShape shape_of(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    AttentionMaskType mask_type,
    bool has_gate,
    bool return_intermediates);
uint64_t l1_budget_bytes(const tt::tt_metal::IDevice* device);

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
};

SDPAForwardCbTiles cb_tiles(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);
uint64_t l1_bytes(const SDPAForwardCbTiles& tiles);

// Work item = one Q chunk of one head stack, see kernels/sdpa_fw_work_list.hpp. Each core gets a list
// of runs of consecutive items.
struct ItemRun {
    uint32_t first_item;
    uint32_t num_items;
};
using CoreItemRuns = std::vector<ItemRun>;

uint32_t num_work_items(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);

// Per-core item runs for num_cores cores: causal chunks as light/heavy pairs when there are enough pairs,
// otherwise contiguous blocks; falls back to contiguous blocks when the lists would not fit the runtime args.
std::vector<CoreItemRuns> split_work(
    const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t num_cores);

// Cheap knob checks for validate: each set knob must divide its axis, and a fully specified config must
// fit l1_budget_bytes. Throws otherwise.
void check_chunks(const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape, uint64_t l1_budget_bytes);

// check_chunks, then picks the unset knobs by a shape rule, shrinking them until they fit l1_budget_bytes
// and keep num_available_cores busy. Throws when even the smallest chunks do not fit.
SDPAForwardChunks resolve_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t num_available_cores);

}  // namespace ttml::metal::ops::sdpa_fw::device
