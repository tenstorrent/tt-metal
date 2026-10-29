// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_SPEC_ARGMAX_FUSED, stage 2 of 2 (tt/spec_argmax_fused.py): the global pick from the all-gathered stage-1
// tiles.
//
// Input : [1,1,32,32*D] uint32 TILE; tile column d is device d's stage-1 tile (row r: lane 0 = order value of the row
//         max on vocab shard d, lane 1 = its local column).
// Output: uint32 ROW_MAJOR in the contract of the tail it replaces:
//         MODE 0 (LANES) [1,1,1,32]: lane r = row r's id, lanes >= R = 0 (the verify's ids row);
//         MODE 1 (ROWS)  [1,1,R,1] : one id per row (the drafter's draft_argmax ids).
// Per row: the device with the strictly greatest order value, scanning d upward, so the lowest device (whose ids are
// all lower) wins a tie; id = d * SHARD + its local column. Together with stage 1 this is torch.argmax over the whole
// vocab row: the lowest global id among the maxima.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t R = get_compile_time_arg_val(0);
constexpr uint32_t D = get_compile_time_arg_val(1);  // devices (vocab shards)
constexpr uint32_t SHARD = get_compile_time_arg_val(2);  // vocab ids per device
constexpr uint32_t MODE = get_compile_time_arg_val(3);  // 0 = LANES, 1 = ROWS
constexpr uint32_t CB_IN = get_compile_time_arg_val(4);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(5);
constexpr uint32_t OUT_PAGE = get_compile_time_arg_val(6);  // ROWS: aligned page size of the output (<= 64)

static_assert(R >= 1 && R <= 32, "one tile row of rows");
static_assert(MODE == 0 || MODE == 1, "LANES or ROWS");
static_assert(MODE == 0 || (OUT_PAGE >= 4 && OUT_PAGE <= 64), "ROWS writes one id per <= 64 B page slot");

constexpr uint32_t IN_ROW_BYTES = 16 * 4;  // one uint32 face row
constexpr uint32_t IN_FACE_BYTES = 16 * IN_ROW_BYTES;
constexpr uint32_t STAGE_BYTES = 2 * IN_FACE_BYTES;  // per device: face 0 rows, then face 2 rows
constexpr uint32_t RA = R < 16 ? R : 16;  // rows in face 0
constexpr uint32_t RB = R > 16 ? R - 16 : 0;  // rows in face 2

FORCE_INLINE uint32_t align64(uint32_t a) { return (a + 63u) & ~63u; }

FORCE_INLINE uint32_t row_off(uint32_t r) {
    return r < 16 ? r * IN_ROW_BYTES : IN_FACE_BYTES + (r - 16) * IN_ROW_BYTES;
}

}  // namespace

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    constexpr auto in_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t l1 = align64(get_write_ptr(CB_IN));
    for (uint32_t d = 0; d < D; ++d) {
        noc_async_read(in.get_noc_addr(d, 0), l1 + d * STAGE_BYTES, RA * IN_ROW_BYTES);
        if constexpr (RB > 0) {
            noc_async_read(
                in.get_noc_addr(d, 2 * IN_FACE_BYTES), l1 + d * STAGE_BYTES + IN_FACE_BYTES, RB * IN_ROW_BYTES);
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    const uint32_t ob = align64(get_write_ptr(CB_OUT));
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob);
    // Zero what is written: LANES lanes >= R must read 0, ROWS page padding stays clean.
    constexpr uint32_t OUT_WORDS = MODE == 0 ? 32 : R * 16;
    for (uint32_t j = 0; j < OUT_WORDS; ++j) {
        o[j] = 0;
    }
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t off = row_off(r);
        int32_t best = *reinterpret_cast<const volatile tt_l1_ptr int32_t*>(l1 + off);
        uint32_t bd = 0;
        for (uint32_t d = 1; d < D; ++d) {
            const int32_t v = *reinterpret_cast<const volatile tt_l1_ptr int32_t*>(l1 + d * STAGE_BYTES + off);
            if (v > best) {
                best = v;
                bd = d;
            }
        }
        const uint32_t col = *reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(l1 + bd * STAGE_BYTES + off + 4);
        const uint32_t id = bd * SHARD + col;
        if constexpr (MODE == 0) {
            o[r] = id;
        } else {
            o[r * 16] = id;  // one 64 B slot per output page
        }
    }
    if constexpr (MODE == 0) {
        noc_async_write(ob, out.get_noc_addr(0), 32 * 4);
    } else {
        for (uint32_t r = 0; r < R; ++r) {
            noc_async_write(ob + r * 64, out.get_noc_addr(r), OUT_PAGE);
        }
    }
    noc_async_write_barrier();
}
