// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_SPEC_ARGMAX_FUSED, stage 1 of 2 (tt/spec_argmax_fused.py): per-device argmax of a vocab-sharded TILE logit
// shard, read in place (no untilize, no second max pass).
//
// Input : this device's logits shard [1,1,R,W], TILE, bf16 or fp32, interleaved. R <= 32, so the rows are one tile row.
// Output: one uint32 [32,32] TILE. Row r < R: lane 0 = the row max as an ORDER VALUE (int32 bits), lane 1 = its LOCAL
//         column (the lowest one on a tie), lanes 2..15 = 0. Rows >= R and lanes 16..31 are not written.
//
// Order value o(v), in the value's own bit width (16 for bf16, 32 for fp32): o = |v| for v >= +0, -|v| for v <= -0.
// It is monotone in v and maps +0 and -0 to the same 0, so a strict '>' over increasing columns returns the first
// column of the maximum: torch.argmax, and the rule of ttnn.argmax (argmax_common.hpp) the shipped tail uses.
// A positive NaN outranks every number and the first one is kept (o clamps to INT32_MAX); a negative NaN ranks below
// -inf. NaN logits are the only inputs where this can differ from torch.argmax.
//
// Work split: core c = y * GX + x (the first NCORES cores, row-wise) owns tiles [c*TPC, min((c+1)*TPC, WT)); this RISC
// owns rows [ROW_LO, ROW_HI) (BRISC the first half of the rows, NCRISC the second). It reads only the face rows of
// those rows and scans them. The (o, col) per row then merge in two levels, each in increasing column order with a
// strict '>' (a lower slot holds lower columns, so it wins a tie): every core posts into slot x of its grid row's
// leader (core (0, y)); the leader merges its row and posts into slot y of the root (core (0, 0)); the root merges the
// grid rows and writes this RISC's rows of the output. Each post is acked before its semaphore increment.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t R = get_compile_time_arg_val(0);  // live rows of the shard (1..32)
constexpr uint32_t W = get_compile_time_arg_val(1);  // live columns (the vocab shard width)
constexpr uint32_t WT = get_compile_time_arg_val(2);  // tiles along W (padded width / 32)
constexpr bool FP32 = get_compile_time_arg_val(3) != 0;
constexpr uint32_t GX = get_compile_time_arg_val(4);  // grid width: cores per grid row, the level-1 merge group
constexpr uint32_t NCORES = get_compile_time_arg_val(5);
constexpr uint32_t TPC = get_compile_time_arg_val(6);  // tiles per core (the last core may own fewer)
constexpr uint32_t ROW_LO = get_compile_time_arg_val(7);  // this RISC's rows [ROW_LO, ROW_HI)
constexpr uint32_t ROW_HI = get_compile_time_arg_val(8);
constexpr uint32_t HALF = get_compile_time_arg_val(9);  // 0 = BRISC rows, 1 = NCRISC rows
constexpr uint32_t CB_IN = get_compile_time_arg_val(10);  // this RISC's face-row staging
constexpr uint32_t CB_PART1 = get_compile_time_arg_val(11);  // GX level-1 slots (read on a grid-row leader)
constexpr uint32_t CB_PART2 = get_compile_time_arg_val(12);  // NG level-2 slots (read on the root)
constexpr uint32_t CB_OUT = get_compile_time_arg_val(13);  // the output tile (root only)
// Common runtime args: [in_addr, out_addr, then (x, y) NOC coordinates of the leader of grid row g, g = 0..NG-1].

static_assert(R >= 1 && R <= 32, "one tile row of logits");
static_assert(ROW_LO < ROW_HI && ROW_HI <= R, "a RISC with no rows is not launched");
static_assert(ROW_HI - ROW_LO <= 16, "each RISC owns at most half of the 32 rows");
static_assert(HALF <= 1, "two RISC halves");

constexpr uint32_t NG = (NCORES + GX - 1) / GX;  // grid rows in use = level-2 slots
constexpr uint32_t SEM_L1 = HALF;  // per-leader post counter of this half
constexpr uint32_t SEM_L2 = 2 + HALF;  // root post counter of this half

constexpr uint32_t ESZ = FP32 ? 4 : 2;
constexpr uint32_t FACE_ROW_BYTES = 16 * ESZ;  // 16 values of one face row
constexpr uint32_t FACE_BYTES = 16 * FACE_ROW_BYTES;  // a 16x16 face
constexpr uint32_t ROW_ALIGN = 64 / FACE_ROW_BYTES;  // face rows per 64 B: reads stay 64 B aligned (DRAM)

constexpr uint32_t clamp16(uint32_t v) { return v < 16 ? v : 16; }
constexpr uint32_t align_dn(uint32_t r) { return r / ROW_ALIGN * ROW_ALIGN; }
constexpr uint32_t align_up(uint32_t r) { return (r + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN; }

// Row group A = tile rows [0,16) (faces 0 | 1), group B = tile rows [16,32) (faces 2 | 3), each clipped to this RISC's
// rows; *_AL / *_AH are the face-local row range actually read (widened to the 64 B read alignment).
constexpr uint32_t A_LO = clamp16(ROW_LO), A_HI = clamp16(ROW_HI);
constexpr uint32_t B_LO = (ROW_LO > 16 ? ROW_LO : 16) - 16, B_HI = (ROW_HI > 16 ? ROW_HI : 16) - 16;
constexpr bool HAS_A = A_HI > A_LO;
constexpr bool HAS_B = B_HI > B_LO;
constexpr uint32_t A_AL = HAS_A ? align_dn(A_LO) : 0, A_AH = HAS_A ? align_up(A_HI) : 0;
constexpr uint32_t B_AL = HAS_B ? align_dn(B_LO) : 0, B_AH = HAS_B ? align_up(B_HI) : 0;
constexpr uint32_t A_CHUNK = (A_AH - A_AL) * FACE_ROW_BYTES;  // bytes of one face's row block
constexpr uint32_t B_CHUNK = (B_AH - B_AL) * FACE_ROW_BYTES;
// Staging layout of one tile: [A: face 0 | face 1][B: face 2 | face 3].
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 2 * A_CHUNK;
constexpr uint32_t TILE_STRIDE = 2 * (A_CHUNK + B_CHUNK);

constexpr uint32_t SLOT_ROW_BYTES = 16;  // (o, col, 0, 0): 16 B keeps every NOC write 16 B aligned
constexpr uint32_t SLOT_BYTES = 32 * SLOT_ROW_BYTES;
constexpr uint32_t OUT_FACE_BYTES = 16 * 16 * 4;  // uint32 output face
constexpr uint32_t OUT_ROW_BYTES = 16 * 4;

constexpr int32_t ORDER_INF = FP32 ? 0x7F800000 : 0x7F80;  // o(+inf); anything above is a positive NaN
constexpr int32_t MAG_MASK = FP32 ? 0x7FFFFFFF : 0x7FFF;
constexpr uint32_t NR = ROW_HI - ROW_LO;

using elem_t = std::conditional_t<FP32, int32_t, int16_t>;

FORCE_INLINE uint32_t align64(uint32_t a) { return (a + 63u) & ~63u; }

FORCE_INLINE void take(int32_t v, uint32_t col, int32_t& best, uint32_t& bidx) {
    best = v > ORDER_INF ? INT32_MAX : v;  // a positive NaN: nothing later can beat it
    bidx = col;
}

// Any running max, one value at a time: the order value of every element, then a strict '>'.
FORCE_INLINE void scan_slow(const elem_t* p, uint32_t col0, uint32_t n, int32_t& best, uint32_t& bidx) {
    for (uint32_t j = 0; j < n; ++j) {
        const int32_t s = p[j];
        const int32_t v = s >= 0 ? s : -(s & MAG_MASK);
        if (v > best) {
            take(v, col0 + j, best, bidx);
        }
    }
}

// 16 consecutive values (one face row) at columns col0..col0+15.
FORCE_INLINE void scan16(const elem_t* p, uint32_t col0, int32_t& best, uint32_t& bidx) {
    if (best < 0) {
        scan_slow(p, col0, 16, best, bidx);
        return;
    }
    if (best == INT32_MAX) {
        return;  // a NaN already holds the row
    }
    // Non-negative running max: a value with the sign bit set cannot beat it, and every other value's raw bits read as
    // a signed integer ARE its order value, so a plain signed compare ranks exactly.
    if constexpr (FP32) {
#pragma GCC unroll 16
        for (uint32_t j = 0; j < 16; ++j) {
            const int32_t v = p[j];
            if (v > best) {
                take(v, col0 + j, best, bidx);
            }
        }
    } else {
        // bf16: two columns per 32-bit word (the lower half is the earlier column). A half h beats best iff
        // (h << 16) >= (best + 1) << 16 as signed int32, so one threshold tests both: the high half in place (the low
        // half below it cannot lift it past a multiple of 2^16), the low half shifted up.
        const int32_t* w = reinterpret_cast<const int32_t*>(p);
        int32_t t = (best + 1) << 16;  // best <= 0x7F80 here: no overflow
#pragma GCC unroll 8
        for (uint32_t j = 0; j < 8; ++j) {
            const int32_t x = w[j];
            const int32_t lo = static_cast<int32_t>(static_cast<uint32_t>(x) << 16);
            if (lo >= t || x >= t) {
                const int32_t lov = lo >> 16;
                const int32_t hiv = x >> 16;
                if (lov > best) {
                    take(lov, col0 + 2 * j, best, bidx);
                }
                if (hiv > best) {
                    take(hiv, col0 + 2 * j + 1, best, bidx);
                }
                if (best == INT32_MAX) {
                    return;
                }
                t = (best + 1) << 16;
            }
        }
    }
}

// Merge rows [ROW_LO, ROW_HI) of the n slots at `base` in slot order (strict '>': the lowest slot wins a tie) and write
// the winners' (o, col) into rows [ROW_LO, ROW_HI) of the slot at `dst`. Track the winning SLOT per row, so the inner
// loop is load + compare, and fetch its column once.
FORCE_INLINE void merge_slots(uint32_t base, uint32_t n, uint32_t dst) {
    int32_t bv[NR];
    uint32_t bs[NR];
    {
        const volatile tt_l1_ptr int32_t* s0 =
            reinterpret_cast<const volatile tt_l1_ptr int32_t*>(base + ROW_LO * SLOT_ROW_BYTES);
#pragma GCC unroll 16
        for (uint32_t k = 0; k < NR; ++k) {
            bv[k] = s0[k * (SLOT_ROW_BYTES / 4)];
            bs[k] = 0;
        }
    }
    for (uint32_t c = 1; c < n; ++c) {
        const volatile tt_l1_ptr int32_t* sc =
            reinterpret_cast<const volatile tt_l1_ptr int32_t*>(base + c * SLOT_BYTES + ROW_LO * SLOT_ROW_BYTES);
#pragma GCC unroll 16
        for (uint32_t k = 0; k < NR; ++k) {
            const int32_t v = sc[k * (SLOT_ROW_BYTES / 4)];
            if (v > bv[k]) {
                bv[k] = v;
                bs[k] = c;
            }
        }
    }
    for (uint32_t k = 0; k < NR; ++k) {
        const uint32_t row = (ROW_LO + k) * SLOT_ROW_BYTES;
        const uint32_t col =
            *reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(base + bs[k] * SLOT_BYTES + row + 4);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + row);
        d[0] = static_cast<uint32_t>(bv[k]);
        d[1] = col;
    }
}

// Post rows [ROW_LO, ROW_HI) of the local slot at `slot` to the same address on core (x, y) (skipped when that core is
// this one), then count it there.
FORCE_INLINE void post(uint32_t slot, bool remote, uint32_t x, uint32_t y, uint32_t sem_id) {
    const uint32_t src = slot + ROW_LO * SLOT_ROW_BYTES;
    if (remote) {
        noc_async_write(src, get_noc_addr(x, y, src), NR * SLOT_ROW_BYTES);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
}

FORCE_INLINE void wait_posts(uint32_t sem_id, uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, n);
    asm volatile("" ::: "memory");
}

}  // namespace

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    constexpr auto in_args = TensorAccessorArgs<14>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t gx = get_absolute_logical_x();
    const uint32_t gy = get_absolute_logical_y();
    const uint32_t core = gy * GX + gx;
    const uint32_t t0 = core * TPC;
    const uint32_t nt = t0 >= WT ? 0 : (WT - t0 < TPC ? WT - t0 : TPC);

    // 1. Stage this RISC's face rows of every owned tile: all reads in flight, one barrier.
    const uint32_t l1_in = align64(get_write_ptr(CB_IN));
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t tile = t0 + i;
        const uint32_t dst = l1_in + i * TILE_STRIDE;
        if constexpr (HAS_A) {
            if constexpr (A_AL == 0 && A_AH == 16) {  // whole faces 0 and 1: one contiguous read
                noc_async_read(in.get_noc_addr(tile, 0), dst + A_OFF, 2 * FACE_BYTES);
            } else {
                noc_async_read(in.get_noc_addr(tile, A_AL * FACE_ROW_BYTES), dst + A_OFF, A_CHUNK);
                noc_async_read(
                    in.get_noc_addr(tile, FACE_BYTES + A_AL * FACE_ROW_BYTES), dst + A_OFF + A_CHUNK, A_CHUNK);
            }
        }
        if constexpr (HAS_B) {
            if constexpr (B_AL == 0 && B_AH == 16) {  // whole faces 2 and 3
                noc_async_read(in.get_noc_addr(tile, 2 * FACE_BYTES), dst + B_OFF, 2 * FACE_BYTES);
            } else {
                noc_async_read(in.get_noc_addr(tile, 2 * FACE_BYTES + B_AL * FACE_ROW_BYTES), dst + B_OFF, B_CHUNK);
                noc_async_read(
                    in.get_noc_addr(tile, 3 * FACE_BYTES + B_AL * FACE_ROW_BYTES), dst + B_OFF + B_CHUNK, B_CHUNK);
            }
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    // 2. Scan: per row, every owned tile in column order (left face row, then right).
    const uint32_t part1 = get_write_ptr(CB_PART1);
    const uint32_t my_slot = part1 + gx * SLOT_BYTES;
    for (uint32_t r = ROW_LO; r < ROW_HI; ++r) {
        int32_t best = INT32_MIN;
        uint32_t bidx = 0;
        const bool in_a = r < 16;
        const uint32_t lo_off =
            in_a ? (A_OFF + (r - A_AL) * FACE_ROW_BYTES) : (B_OFF + (r - 16 - B_AL) * FACE_ROW_BYTES);
        const uint32_t hi_off = lo_off + (in_a ? A_CHUNK : B_CHUNK);
        for (uint32_t i = 0; i < nt; ++i) {
            const uint32_t base = l1_in + i * TILE_STRIDE;
            const uint32_t col = (t0 + i) * 32;
            const elem_t* lo = reinterpret_cast<const elem_t*>(base + lo_off);
            const elem_t* hi = reinterpret_cast<const elem_t*>(base + hi_off);
            if (col + 32 <= W) {
                scan16(lo, col, best, bidx);
                scan16(hi, col + 16, best, bidx);
            } else if (col < W) {  // the last tile of a shard whose width is not a multiple of 32
                const uint32_t valid = W - col;
                scan_slow(lo, col, valid < 16 ? valid : 16, best, bidx);
                if (valid > 16) {
                    scan_slow(hi, col + 16, valid - 16, best, bidx);
                }
            }
        }
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_slot + r * SLOT_ROW_BYTES);
        s[0] = static_cast<uint32_t>(best);
        s[1] = bidx;
        s[2] = 0;
        s[3] = 0;
    }

    // 3. Level 1: into slot gx of this grid row's leader.
    const uint32_t lx = get_common_arg_val<uint32_t>(2 + 2 * gy);
    const uint32_t ly = get_common_arg_val<uint32_t>(3 + 2 * gy);
    post(my_slot, gx != 0, lx, ly, SEM_L1);
    if (gx != 0) {
        return;
    }

    // 4. Leader: merge this grid row, post it into slot gy of the root.
    const uint32_t row_cores = NCORES - gy * GX < GX ? NCORES - gy * GX : GX;
    wait_posts(SEM_L1, row_cores);
    const uint32_t part2 = get_write_ptr(CB_PART2);
    const uint32_t row_slot = part2 + gy * SLOT_BYTES;
    merge_slots(part1, row_cores, row_slot);
    post(row_slot, gy != 0, get_common_arg_val<uint32_t>(2), get_common_arg_val<uint32_t>(3), SEM_L2);
    if (gy != 0) {
        return;
    }

    // 5. Root: merge the grid rows; this half's rows of the output tile, a 64 B face row each ([o, col, 0 x 14]).
    wait_posts(SEM_L2, NG);
    const uint32_t ob = align64(get_write_ptr(CB_OUT));
    const uint32_t fin = ob + 4 * OUT_FACE_BYTES;  // merged slot scratch, past the tile
    merge_slots(part2, NG, fin);
    for (uint32_t k = 0; k < NR; ++k) {
        const uint32_t r = ROW_LO + k;
        const uint32_t face = r < 16 ? 0 : 2;
        volatile tt_l1_ptr uint32_t* o =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob + face * OUT_FACE_BYTES + (r % 16) * OUT_ROW_BYTES);
        const volatile tt_l1_ptr uint32_t* m =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(fin + r * SLOT_ROW_BYTES);
        o[0] = m[0];
        o[1] = m[1];
        for (uint32_t j = 2; j < 16; ++j) {
            o[j] = 0;
        }
    }
    if constexpr (HAS_A) {
        noc_async_write(
            ob + A_LO * OUT_ROW_BYTES, out.get_noc_addr(0, A_LO * OUT_ROW_BYTES), (A_HI - A_LO) * OUT_ROW_BYTES);
    }
    if constexpr (HAS_B) {
        noc_async_write(
            ob + 2 * OUT_FACE_BYTES + B_LO * OUT_ROW_BYTES,
            out.get_noc_addr(0, 2 * OUT_FACE_BYTES + B_LO * OUT_ROW_BYTES),
            (B_HI - B_LO) * OUT_ROW_BYTES);
    }
    noc_async_write_barrier();
}
