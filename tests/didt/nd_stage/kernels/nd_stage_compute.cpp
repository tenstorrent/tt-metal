// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One stage probe per build, selected by ND_PROBE. Each probe drives one part of the Tensix pipeline at
// high duty for `reps` inner passes and leaves a small 32-bit result in cb_out, so a fault in that part
// persists into bytes the dataflow checker compares run-to-run. Probe ids match PROBES in nd_stage_lib.py.
//
// Compile-time args: cb_in0, cb_in1, cb_out, cb_loop, rt, ct, kt, in0_tiles, in1_tiles
// Runtime args:      n_iters, reps, check_every, dest_ref_addr, dest_result_addr, core_id,
//                    inject_iter, inject_word, inject_xor
// Only MM_DESTRD reads the args after reps: it checks DEST itself on the math thread, before the packer.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#define PROBE_MM_DEST 0
#define PROBE_MM_L1ACC 1
#define PROBE_FPU_ELTWISE 2
#define PROBE_SFPU_INT 3
#define PROBE_UNPACK_PACK 4
#define PROBE_PACK_L1ACC 5
#define PROBE_MM_DESTRD 6
#define PROBE_ELTWISE_SRCB 7
#define PROBE_SFPU_LDST 8
#define PROBE_SFPU_FP 9

#ifndef ND_PROBE
#error "ND_PROBE must be defined"
#endif

#if ND_PROBE == PROBE_MM_DESTRD
#include "api/compute/compute_kernel_api_debug.h"
#include "nd_stage_result.h"
#endif

namespace {

// Half-sync DEST holds four 32-bit tiles; every probe fills exactly that.
constexpr uint32_t DST_TILES = 4;

ALWI void pack_dst(uint32_t cb) {
    cb_reserve_back(cb, DST_TILES);
    tile_regs_wait();
    for (uint32_t i = 0; i < DST_TILES; ++i) {
        pack_tile(i, cb);
    }
    tile_regs_release();
    cb_push_back(cb, DST_TILES);
}

#if ND_PROBE == PROBE_MM_DESTRD && defined(TRISC_MATH)
// Compares DEST itself, before the packer reads it, against iteration 0's copy in L1, via the RISC's
// memory-mapped view of DEST (swizzled, so words are in the FPU's tile order). Full-sync DEST keeps the
// four tiles at offset 0 every iteration. Returns whether any word differed.
bool check_dest(
    nd_stage::Result& res,
    uint32_t it,
    uint32_t ref_addr,
    uint32_t inject_iter,
    uint32_t inject_word,
    uint32_t inject_xor) {
    constexpr uint32_t words_per_tile = 1024;
    volatile uint32_t* dest = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR);
    volatile tt_l1_ptr uint32_t* ref = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ref_addr);
    // The math thread's MVMULs are still in flight when the RISC gets here; wait until they land.
    ckernel::tensix_sync();
    if (it == inject_iter) {
        // Positive control: a flipped DEST bit, as a faulty accumulate would leave it, seen by both this
        // check and the packed output.
        dest[inject_word] ^= inject_xor;
        ckernel::tensix_sync();
    }
    bool bad = false;
    for (uint32_t w = 0; w < DST_TILES * words_per_tile; ++w) {
        const uint32_t got = dest[w];
        if (it == 0) {
            ref[w] = got;
            continue;
        }
        const uint32_t want = ref[w];
        if (got != want) {
            bad = true;
            res.record_mismatch<0>(it, w, got, want, words_per_tile);
        }
    }
    return bad;
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_loop = get_compile_time_arg_val(3);
    constexpr uint32_t rt = get_compile_time_arg_val(4);
    constexpr uint32_t ct = get_compile_time_arg_val(5);
    constexpr uint32_t kt = get_compile_time_arg_val(6);
    constexpr uint32_t in0_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t in1_tiles = get_compile_time_arg_val(8);
    static_assert(rt * ct == DST_TILES, "matmul block must fill DEST exactly");

    const uint32_t n_iters = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);

    // The inputs are pushed once by the checker and never popped: every iteration re-reads the same tiles.
    cb_wait_front(cb_in0, in0_tiles);
    if constexpr (in1_tiles > 0) {
        cb_wait_front(cb_in1, in1_tiles);
    }

#if ND_PROBE == PROBE_MM_DEST || ND_PROBE == PROBE_MM_L1ACC
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_block_init(cb_in0, cb_in1, 0, ct, rt, kt);
    for (uint32_t it = 0; it < n_iters; ++it) {
#if ND_PROBE == PROBE_MM_DEST
        // Accumulates reps * kt k-steps in DEST before one pack: the MVMUL/accumulate path dominates.
        tile_regs_acquire();
        for (uint32_t r = 0; r < reps; ++r) {
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_in0, cb_in1, k, k * ct, 0, 0, ct, rt, kt);
            }
        }
        tile_regs_commit();
        pack_dst(cb_out);
#else
        // The production spill path: every k-sweep packs a partial and the packer adds it onto the slot it
        // packed two blocks earlier (cb_out holds two slots, so blocks alternate between them).
        for (uint32_t b = 0; b < reps; ++b) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_in0, cb_in1, k, k * ct, 0, 0, ct, rt, kt);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, DST_TILES);
            tile_regs_wait();
            pack_reconfig_l1_acc(b >= 2 ? 1 : 0);
            for (uint32_t i = 0; i < DST_TILES; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, DST_TILES);
        }
#endif
    }
#if ND_PROBE == PROBE_MM_L1ACC
    pack_reconfig_l1_acc(0);
#endif

#elif ND_PROBE == PROBE_MM_DESTRD
    // MM_DEST with DEST read back by the math RISC on checked iterations, before the pack. Comparing this
    // record with the packed one splits "wrong in DEST" (unpack, FPU, DEST) from "wrong after DEST"
    // (packer, L1 write, checker read).
    const uint32_t check_every = get_arg_val<uint32_t>(2);
    const uint32_t dest_ref_addr = get_arg_val<uint32_t>(3);
    const uint32_t dest_result_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_id = get_arg_val<uint32_t>(5);
    const uint32_t inject_iter = get_arg_val<uint32_t>(6);
    const uint32_t inject_word = get_arg_val<uint32_t>(7);
    const uint32_t inject_xor = get_arg_val<uint32_t>(8);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_block_init(cb_in0, cb_in1, 0, ct, rt, kt);
#ifdef TRISC_MATH
    static_assert(DST_SYNC_MODE == DstSync::SyncFull, "check_dest reads DEST at offset 0");
    nd_stage::Result res(dest_result_addr, nd_stage::MAGIC_DEST, core_id);
    ckernel::configure_dest_access<ckernel::MathThreadId>(DataFormat::Float32, true);
#endif
    for (uint32_t it = 0; it < n_iters; ++it) {
        tile_regs_acquire();
        for (uint32_t r = 0; r < reps; ++r) {
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_in0, cb_in1, k, k * ct, 0, 0, ct, rt, kt);
            }
        }
#ifdef TRISC_MATH
        const bool check_it = it == 0 || it + 1 == n_iters || it % check_every == 0;
        const bool bad = check_it && check_dest(res, it, dest_ref_addr, inject_iter, inject_word, inject_xor);
        res.end_iteration(check_it && it > 0, bad, it);
#endif
        tile_regs_commit();
        pack_dst(cb_out);
    }
#ifdef TRISC_MATH
    res.finish();
#endif

#elif ND_PROBE == PROBE_FPU_ELTWISE || ND_PROBE == PROBE_ELTWISE_SRCB
    // DEST is moved back into a source register every step and +v / -v alternate from the other one. The
    // move drops fp32 mantissa bits, so DEST returns to the input only for small-integer data (ND_DATA=int,
    // golden-checked); random data is still deterministic. Exercises the FPU add datapath and the DEST move,
    // not MVMUL. DEST_TO_SRCA feeds v via unpacker 1 into SrcB and moves DEST to SrcA (MOVD2A); DEST_TO_SRCB
    // feeds v via unpacker 0 into SrcA and moves DEST to SrcB (MOVD2B), so the pair splits the two unpackers.
#if ND_PROBE == PROBE_FPU_ELTWISE
    constexpr auto reuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
#else
    constexpr auto reuse = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
#endif
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    for (uint32_t it = 0; it < n_iters; ++it) {
        tile_regs_acquire();
        copy_init(cb_in0);
        for (uint32_t i = 0; i < DST_TILES; ++i) {
            copy_tile(cb_in0, i, i);
        }
        add_reuse_dest_init<reuse>(cb_in1);
        for (uint32_t s = 0; s < reps; ++s) {
            for (uint32_t i = 0; i < DST_TILES; ++i) {
                add_reuse_dest_tiles<reuse>(cb_in1, s & 1, i);
            }
        }
        tile_regs_commit();
        pack_dst(cb_out);
    }

#elif ND_PROBE == PROBE_SFPU_INT || ND_PROBE == PROBE_SFPU_LDST || ND_PROBE == PROBE_SFPU_FP
    // Four 32-bit tiles unpack straight into DEST, then the SFPU works on them in place for reps rounds.
    //   SFPU_INT:  tiles 0..2 *= tile 3 (odd) as int32. x * c^reps mod 2^32 is exact, and an odd multiplier
    //              is invertible, so a single flipped bit never washes out. Integer multiply path.
    //   SFPU_LDST: tiles 2,3 hold copies of 0,1 and are copied 0->2, 1->3, 2->0, 3->1 each round: SFPLOAD,
    //              LREGs and SFPSTORE only, no arithmetic.
    //   SFPU_FP:   every tile x2 then x0.5 in fp32 (SFPMAD), exact for normal values.
    compute_kernel_hw_startup(cb_in0, cb_out);
    for (uint32_t it = 0; it < n_iters; ++it) {
        tile_regs_acquire();
        copy_init(cb_in0);
        for (uint32_t i = 0; i < DST_TILES; ++i) {
            copy_tile(cb_in0, i, i);
        }
#if ND_PROBE == PROBE_SFPU_INT
        mul_int_tile_init<DataFormat::Int32>();
        for (uint32_t s = 0; s < reps; ++s) {
            for (uint32_t i = 0; i < DST_TILES - 1; ++i) {
                mul_int_tile<DataFormat::Int32>(i, DST_TILES - 1, i);
            }
        }
#elif ND_PROBE == PROBE_SFPU_LDST
        copy_dest_values_init();
        for (uint32_t s = 0; s < reps; ++s) {
            copy_dest_values<DataFormat::Int32>(0, 2);
            copy_dest_values<DataFormat::Int32>(1, 3);
            copy_dest_values<DataFormat::Int32>(2, 0);
            copy_dest_values<DataFormat::Int32>(3, 1);
        }
#else
        binop_with_scalar_tile_init();
        for (uint32_t s = 0; s < reps; ++s) {
            for (uint32_t i = 0; i < DST_TILES; ++i) {
                mul_unary_tile(i, 0x40000000u);  // 2.0f
                mul_unary_tile(i, 0x3F000000u);  // 0.5f
            }
        }
#endif
        tile_regs_commit();
        pack_dst(cb_out);
    }

#elif ND_PROBE == PROBE_UNPACK_PACK
    // Eight tiles circulate cb_in0 -> cb_loop -> ... -> cb_loop -> cb_out, reps passes, with no arithmetic.
    // A 32-bit format unpacks straight into DEST; a 16-bit format goes via SrcA and the FPU datacopy.
    constexpr uint32_t n_tiles = 2 * DST_TILES;
    compute_kernel_hw_startup(cb_in0, cb_out);
    copy_init(cb_in0);
    for (uint32_t it = 0; it < n_iters; ++it) {
        for (uint32_t p = 0; p < reps; ++p) {
            const bool first = p == 0;
            const uint32_t src = first ? cb_in0 : cb_loop;
            const uint32_t dst = p + 1 == reps ? cb_out : cb_loop;
            for (uint32_t b = 0; b < n_tiles / DST_TILES; ++b) {
                if (!first) {
                    cb_wait_front(cb_loop, DST_TILES);
                }
                tile_regs_acquire();
                for (uint32_t i = 0; i < DST_TILES; ++i) {
                    copy_tile(src, (first ? b * DST_TILES : 0) + i, i);
                }
                tile_regs_commit();
                if (!first) {
                    cb_pop_front(cb_loop, DST_TILES);
                }
                pack_dst(dst);
            }
        }
    }

#elif ND_PROBE == PROBE_PACK_L1ACC
    // DEST is loaded once per iteration and packed reps times with L1 accumulation, alternating two slots:
    // the packer's L1 read-add-write path at full rate, with the FPU idle.
    compute_kernel_hw_startup(cb_in0, cb_out);
    copy_init(cb_in0);
    for (uint32_t it = 0; it < n_iters; ++it) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < DST_TILES; ++i) {
            copy_tile(cb_in0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t p = 0; p < reps; ++p) {
            cb_reserve_back(cb_out, DST_TILES);
            pack_reconfig_l1_acc(p >= 2 ? 1 : 0);
            for (uint32_t i = 0; i < DST_TILES; ++i) {
                pack_tile(i, cb_out);
            }
            cb_push_back(cb_out, DST_TILES);
        }
        pack_reconfig_l1_acc(0);
        tile_regs_release();
    }

#else
#error "unknown ND_PROBE"
#endif
}
