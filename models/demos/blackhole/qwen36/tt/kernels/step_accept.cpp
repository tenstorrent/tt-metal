// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The step program of a one-enqueue MTP step (tt/spec_accept.py StepAccept) on one data-movement core: the greedy
// walk over the verify ids and the drafter's inputs, accept_words in spec_step_words.hpp.
//
// Inputs (ROW_MAJOR, one page per row):
//   ids    uint32 [1, 1, 1, 32]   the verify's argmax lanes, word u * T + j
//   draws  uint32 [1, 1, 1, R]    the sampler's draws (DRAWS only)
//   step   uint32 [9, >= R]       the verify step buffer: row VERIFY_TOKENS and rows STEP_NUM_VALID .. STEP_KEEP
//   pt     int32  [R, NB]         row u * T + j holds user u's drafter page table row
// Outputs (every word of every row written, as the op graph's ttnn.copy writes them):
//   pack   uint32 [2K + 5, PACK_WORDS]  the drafter pack
//   rs     int32  [R, NB]               the reseed table: a real reseed row keeps its pt row, the others SCRATCH
// The reseed table moves in column chunks of CHUNK bytes, so L1 holds R + 1 chunks whatever NB is.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "spec_step_words.hpp"

namespace {

using namespace qwen36_spec;

constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t PACK_WORDS = get_compile_time_arg_val(2);
constexpr uint32_t NB = get_compile_time_arg_val(3);
constexpr int32_t TOP = static_cast<int32_t>(get_compile_time_arg_val(4));
constexpr int32_t SCRATCH = static_cast<int32_t>(get_compile_time_arg_val(5));
constexpr bool DRAWS = get_compile_time_arg_val(6) != 0;
constexpr uint32_t CHUNK = get_compile_time_arg_val(7);
constexpr uint32_t CB_L1 = get_compile_time_arg_val(8);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(9);
constexpr uint32_t CTA_ACCESSORS = 10;

constexpr uint32_t T = K + 1;
constexpr uint32_t R = B * T;
constexpr uint32_t ROW_BYTES = NB * 4;
constexpr AcceptLayout L(B, K, PACK_WORDS, CHUNK);

static_assert(B >= 1 && K >= 1 && R <= 32, "one 32-lane ids row");
static_assert(R <= PACK_WORDS && NB >= 1, "pack row and page table widths");
static_assert(CHUNK % SLOT == 0 && CHUNK >= SLOT, "column chunks start on a slot");
static_assert(L.bytes + SLOT <= CB_BYTES, "the scratch CB holds the L1 layout");

}  // namespace

void kernel_main() {
    constexpr auto ids_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto draws_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto step_args = TensorAccessorArgs<draws_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<step_args.next_compile_time_args_offset()>();
    constexpr auto pack_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto rs_args = TensorAccessorArgs<pack_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(0));
    const auto draws = TensorAccessor(draws_args, get_common_arg_val<uint32_t>(1));
    const auto step = TensorAccessor(step_args, get_common_arg_val<uint32_t>(2));
    const auto pt = TensorAccessor(pt_args, get_common_arg_val<uint32_t>(3));
    const auto pack = TensorAccessor(pack_args, get_common_arg_val<uint32_t>(4));
    const auto rs = TensorAccessor(rs_args, get_common_arg_val<uint32_t>(5));

    const uint32_t base = align(get_write_ptr(CB_L1), SLOT);

    noc_async_read(ids.get_noc_addr(0), base + L.ids, R * 4);
    if constexpr (DRAWS) {
        noc_async_read(draws.get_noc_addr(0), base + L.draws, R * 4);
    }
    noc_async_read(step.get_noc_addr(VERIFY_TOKENS), base + L.tokens, R * 4);
    for (uint32_t i = 0; i < STEP_USER_ROWS; ++i) {
        noc_async_read(step.get_noc_addr(STEP_NUM_VALID + i), base + L.user_rows + i * slot_bytes(B), B * 4);
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    uint32_t* real = reinterpret_cast<uint32_t*>(base + L.real);
    accept_words(
        AcceptShape{B, K, PACK_WORDS, TOP},
        AcceptIn{
            reinterpret_cast<const uint32_t*>(base + L.ids),
            DRAWS ? reinterpret_cast<const uint32_t*>(base + L.draws) : nullptr,
            reinterpret_cast<const uint32_t*>(base + L.tokens),
            reinterpret_cast<const uint32_t*>(base + L.user_rows),
            slot_bytes(B) / 4},
        AcceptOut{reinterpret_cast<uint32_t*>(base + L.pack), slot_bytes(PACK_WORDS) / 4, real});
    uint32_t* scratch = reinterpret_cast<uint32_t*>(base + L.scratch);
    for (uint32_t i = 0; i < CHUNK / 4; ++i) {
        scratch[i] = static_cast<uint32_t>(SCRATCH);
    }
    asm volatile("" ::: "memory");

    for (uint32_t i = 0; i < pack_rows(K); ++i) {
        noc_async_write(base + L.pack + i * slot_bytes(PACK_WORDS), pack.get_noc_addr(i), PACK_WORDS * 4);
    }
    for (uint32_t lo = 0; lo < ROW_BYTES; lo += CHUNK) {
        const uint32_t n = ROW_BYTES - lo < CHUNK ? ROW_BYTES - lo : CHUNK;
        if (lo > 0) {
            noc_async_writes_flushed();  // the previous chunk's pt pieces have left L1
        }
        for (uint32_t r = 0; r < R; ++r) {
            if (real[r] != 0) {
                noc_async_read(pt.get_noc_addr(r, lo), base + L.pt + r * CHUNK, n);
            }
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src = real[r] != 0 ? base + L.pt + r * CHUNK : base + L.scratch;
            noc_async_write(src, rs.get_noc_addr(r, lo), n);
        }
    }
    noc_async_write_barrier();
}
