// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Work item arithmetic shared by the host (work split, cost model) and the three kernels. A work item
// is Sq_chunk_t query tile-rows of heads_per_item consecutive Q heads (one K/V group):
//   item = (batch * stacks_per_batch + stack) * q_num_chunks + chunk,  stacks_per_batch = q_heads / heads_per_item
// Plain C++17, no kernel or host includes.
namespace sdpa_fw_items {

constexpr uint32_t first_row_tile_of(uint32_t item, uint32_t q_num_chunks, uint32_t Sq_chunk_t) {
    return (item % q_num_chunks) * Sq_chunk_t;
}

// K/V steps (Sk_chunk_t tiles each) a chunk attends to: causal up to its last diagonal tile, else all.
constexpr uint32_t num_kv_steps_of_chunk(
    bool causal, uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t first_row_tile) {
    return causal ? (first_row_tile + Sq_chunk_t + Sk_chunk_t - 1U) / Sk_chunk_t : Ht / Sk_chunk_t;
}

struct ItemCoords {
    uint32_t batch;
    uint32_t head0;  // first Q head of the stack
    uint32_t first_row_tile;
};

constexpr ItemCoords decode_item(
    uint32_t item, uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item) {
    const uint32_t stacks_per_batch = q_heads / heads_per_item;
    const uint32_t stack = item / q_num_chunks;
    return {
        stack / stacks_per_batch,
        (stack % stacks_per_batch) * heads_per_item,
        first_row_tile_of(item, q_num_chunks, Sq_chunk_t)};
}

// A core's work list: units of three runtime-arg words, encoded by the host (sdpa_fw_config.cpp) and
// decoded by the kernels (kernels/sdpa_fw_work_list.hpp):
//   w0: item [0,20) | role [20,22) | num_partials [22,28) | sem_idx [28,32)
//   w1: whole: run length; sender / merger: step_begin [0,16) | step_end [16,32)
//   w2: slot [0,16) | merger NoC x [16,24) | merger NoC y [24,32)
constexpr uint32_t kUnitWhole = 0U;   // a run of consecutive items over their full K/V range
constexpr uint32_t kUnitSender = 1U;  // one item's step range, sent to a merger as a partial
constexpr uint32_t kUnitMerger = 2U;  // one item's step range, then the senders' partials folded in

constexpr uint32_t kUnitItemBits = 20U;
constexpr uint32_t kUnitRoleBits = 2U;
constexpr uint32_t kUnitPartialsBits = 6U;
constexpr uint32_t kUnitSemBits = 4U;
constexpr uint32_t kUnitStepBits = 16U;
constexpr uint32_t kUnitSlotBits = 16U;
constexpr uint32_t kUnitNocBits = 8U;
constexpr uint32_t kUnitRoleShift = kUnitItemBits;
constexpr uint32_t kUnitPartialsShift = kUnitRoleShift + kUnitRoleBits;
constexpr uint32_t kUnitSemShift = kUnitPartialsShift + kUnitPartialsBits;
constexpr uint32_t kUnitNocXShift = kUnitSlotBits;
constexpr uint32_t kUnitNocYShift = kUnitSlotBits + kUnitNocBits;
constexpr uint32_t unit_field_max(uint32_t bits) {
    return (1U << bits) - 1U;
}

constexpr uint32_t kMaxMergersPerCore = 4U;  // semaphores per core, one per merger

struct WorkUnitFields {
    uint32_t item;
    uint32_t role;
    uint32_t num_items;     // whole
    uint32_t step_begin;    // sender / merger
    uint32_t step_end;      // sender / merger
    uint32_t num_partials;  // merger
    uint32_t sem_idx;       // merger: which of its core's semaphores the senders bump
    uint32_t slot;          // sender: scratch slot it writes; merger: first slot it reads
    uint32_t merger_x;      // sender: merger core, NoC coordinates
    uint32_t merger_y;
};

struct UnitWords {
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
};

// Fields must be within their widths; the host checks that before encoding.
constexpr UnitWords encode_unit(const WorkUnitFields& u) {
    return {
        u.item | (u.role << kUnitRoleShift) | (u.num_partials << kUnitPartialsShift) | (u.sem_idx << kUnitSemShift),
        u.role == kUnitWhole ? u.num_items : (u.step_begin | (u.step_end << kUnitStepBits)),
        u.slot | (u.merger_x << kUnitNocXShift) | (u.merger_y << kUnitNocYShift)};
}

constexpr WorkUnitFields decode_unit(uint32_t w0, uint32_t w1, uint32_t w2) {
    WorkUnitFields u{};
    u.item = w0 & unit_field_max(kUnitItemBits);
    u.role = (w0 >> kUnitRoleShift) & unit_field_max(kUnitRoleBits);
    u.num_partials = (w0 >> kUnitPartialsShift) & unit_field_max(kUnitPartialsBits);
    u.sem_idx = (w0 >> kUnitSemShift) & unit_field_max(kUnitSemBits);
    if (u.role == kUnitWhole) {
        u.num_items = w1;
    } else {
        u.num_items = 1U;
        u.step_begin = w1 & unit_field_max(kUnitStepBits);
        u.step_end = w1 >> kUnitStepBits;
    }
    u.slot = w2 & unit_field_max(kUnitSlotBits);
    u.merger_x = (w2 >> kUnitNocXShift) & unit_field_max(kUnitNocBits);
    u.merger_y = (w2 >> kUnitNocYShift) & unit_field_max(kUnitNocBits);
    return u;
}

}  // namespace sdpa_fw_items
