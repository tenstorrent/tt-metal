// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The chain program of a chained MTP verify (tt/spec_chain.py ChainInputs) on one data-movement core: the inputs of
// verify n + 1 from the unread verify n, chain_words in spec_step_words.hpp.
//
// Inputs (ROW_MAJOR, one page per row):
//   ids        uint32 [1, 1, 1, 32]          verify n's argmax lanes, word src * T + j
//   draws      uint32 [1, 1, 1, R]           verify n's sampler draws (SAMPLER only)
//   pack       uint32 [2K + 5, >= R]         the drafter pack of verify n (row draft_anchor(K) is read)
//   overlay    int32  [9, >= B]              the host rows of verify n + 1 (spec_chain.chain_overlay)
//   draft k    uint32 [1, 1, B, 1]           K tensors: the drafts made from verify n's commit, one id per page
//   cand       int32  [T * B, 2 * NV * G]    the ctrl candidates of every count (GdnBlockState.ctrl_candidates)
//   seed_cand  int32  [T * B, T]             the seed candidates of every count (SAMPLER only)
// Outputs (every word written, as the op graph's ttnn.copy writes them):
//   step       uint32 [9, STEP_WORDS]        the verify step buffer of verify n + 1
//   ctrl g     int32  [1, CTRL_WORDS]        G GDN verify ctrl pages
//   seeds      uint32 [R]                    the sampler's seeds (SAMPLER only)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "spec_step_words.hpp"

namespace {

using namespace qwen36_spec;

constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t NV = get_compile_time_arg_val(2);
constexpr uint32_t G = get_compile_time_arg_val(3);
constexpr uint32_t STEP_WORDS = get_compile_time_arg_val(4);
constexpr uint32_t CTRL_WORDS = get_compile_time_arg_val(5);
constexpr int32_t TOP = static_cast<int32_t>(get_compile_time_arg_val(6));
constexpr bool SAMPLER = get_compile_time_arg_val(7) != 0;
constexpr uint32_t DRAFT_PAGE = get_compile_time_arg_val(8);  // bytes read per draft id page
constexpr uint32_t CB_L1 = get_compile_time_arg_val(9);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(10);
constexpr uint32_t CTA_ACCESSORS = 11;

constexpr uint32_t T = K + 1;
constexpr uint32_t R = B * T;
constexpr uint32_t CAND_WORDS = 2 * NV * G;
constexpr ChainLayout L(B, K, NV, G, STEP_WORDS, CTRL_WORDS, SAMPLER);

static_assert(B >= 1 && K >= 1 && R <= 32 && G >= 1, "one 32-lane ids row");
static_assert(R <= STEP_WORDS && 1 + B + 2 * B * NV <= CTRL_WORDS, "step row and ctrl page widths");
static_assert(DRAFT_PAGE >= 4 && DRAFT_PAGE <= SLOT, "one draft id per slot");
static_assert(L.bytes + SLOT <= CB_BYTES, "the scratch CB holds the L1 layout");

}  // namespace

void kernel_main() {
    constexpr auto ids_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto draws_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto pack_args = TensorAccessorArgs<draws_args.next_compile_time_args_offset()>();
    constexpr auto ovr_args = TensorAccessorArgs<pack_args.next_compile_time_args_offset()>();
    constexpr auto cand_args = TensorAccessorArgs<ovr_args.next_compile_time_args_offset()>();
    constexpr auto seedc_args = TensorAccessorArgs<cand_args.next_compile_time_args_offset()>();
    constexpr auto draft_args = TensorAccessorArgs<seedc_args.next_compile_time_args_offset()>();
    constexpr auto step_args = TensorAccessorArgs<draft_args.next_compile_time_args_offset()>();
    constexpr auto ctrl_args = TensorAccessorArgs<step_args.next_compile_time_args_offset()>();
    constexpr auto seeds_args = TensorAccessorArgs<ctrl_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(0));
    const auto draws = TensorAccessor(draws_args, get_common_arg_val<uint32_t>(1));
    const auto pack = TensorAccessor(pack_args, get_common_arg_val<uint32_t>(2));
    const auto ovr = TensorAccessor(ovr_args, get_common_arg_val<uint32_t>(3));
    const auto cand = TensorAccessor(cand_args, get_common_arg_val<uint32_t>(4));
    const auto seedc = TensorAccessor(seedc_args, get_common_arg_val<uint32_t>(5));
    const auto step = TensorAccessor(step_args, get_common_arg_val<uint32_t>(6));
    const auto seeds = TensorAccessor(seeds_args, get_common_arg_val<uint32_t>(7));
    constexpr uint32_t ARG_DRAFTS = 8;              // K draft id tensors
    constexpr uint32_t ARG_CTRLS = ARG_DRAFTS + K;  // G ctrl pages

    const uint32_t base = align(get_write_ptr(CB_L1), SLOT);

    noc_async_read(ids.get_noc_addr(0), base + L.ids, R * 4);
    if constexpr (SAMPLER) {
        noc_async_read(draws.get_noc_addr(0), base + L.draws, R * 4);
    }
    noc_async_read(pack.get_noc_addr(draft_anchor(K)), base + L.anchor, B * 4);
    for (uint32_t i = 0; i < CHAIN_OVERLAY_ROWS; ++i) {
        noc_async_read(ovr.get_noc_addr(i), base + L.overlay + i * slot_bytes(B), B * 4);
    }
    for (uint32_t k = 0; k < K; ++k) {
        const auto draft = TensorAccessor(draft_args, get_common_arg_val<uint32_t>(ARG_DRAFTS + k));
        for (uint32_t u = 0; u < B; ++u) {
            noc_async_read(draft.get_noc_addr(u), base + L.drafts + (k * B + u) * SLOT, DRAFT_PAGE);
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        noc_async_read(cand.get_noc_addr(r), base + L.cand + r * slot_bytes(CAND_WORDS), CAND_WORDS * 4);
    }
    if constexpr (SAMPLER) {
        for (uint32_t r = 0; r < R; ++r) {
            noc_async_read(seedc.get_noc_addr(r), base + L.seed_cand + r * slot_bytes(T), T * 4);
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    chain_words(
        ChainShape{B, K, NV, G, STEP_WORDS, CTRL_WORDS, TOP},
        ChainIn{
            reinterpret_cast<const uint32_t*>(base + L.ids),
            SAMPLER ? reinterpret_cast<const uint32_t*>(base + L.draws) : nullptr,
            reinterpret_cast<const uint32_t*>(base + L.anchor),
            reinterpret_cast<const int32_t*>(base + L.overlay),
            slot_bytes(B) / 4,
            reinterpret_cast<const uint32_t*>(base + L.drafts),
            SLOT / 4,
            reinterpret_cast<const int32_t*>(base + L.cand),
            slot_bytes(CAND_WORDS) / 4,
            SAMPLER ? reinterpret_cast<const int32_t*>(base + L.seed_cand) : nullptr,
            slot_bytes(T) / 4},
        ChainOut{
            reinterpret_cast<uint32_t*>(base + L.step),
            slot_bytes(STEP_WORDS) / 4,
            reinterpret_cast<int32_t*>(base + L.ctrl),
            slot_bytes(CTRL_WORDS) / 4,
            SAMPLER ? reinterpret_cast<uint32_t*>(base + L.seeds) : nullptr});
    asm volatile("" ::: "memory");

    for (uint32_t i = 0; i < VERIFY_STEP_ROWS; ++i) {
        noc_async_write(base + L.step + i * slot_bytes(STEP_WORDS), step.get_noc_addr(i), STEP_WORDS * 4);
    }
    for (uint32_t g = 0; g < G; ++g) {
        const auto ctrl = TensorAccessor(ctrl_args, get_common_arg_val<uint32_t>(ARG_CTRLS + g));
        noc_async_write(base + L.ctrl + g * slot_bytes(CTRL_WORDS), ctrl.get_noc_addr(0), CTRL_WORDS * 4);
    }
    if constexpr (SAMPLER) {
        noc_async_write(base + L.seeds, seeds.get_noc_addr(0), R * 4);
    }
    noc_async_write_barrier();
}
