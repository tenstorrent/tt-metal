// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC "pre" — sync kernel (NCRISC, NoC1): core-to-core partials, messages and the root's re-layouts.
//
// Round 1  K core   CB_PART (M, S) -> leader CB_GATHER slot y (8 KB) + leader SEM_G1
//          leader   SEM_G1 == 8 -> push CB_GATHER (16); CB_GSUM -> root slot x + root SEM_G1
//          root     SEM_G1 == 8 -> push; CB_RED (z) -> CB_PLANE (plane layout); CB_COEF -> mailbox (pre, 4 x 32
//                   fp32) to the 8 leaders + their SEM_M1; post / comb output tiles
//          leader   SEM_M1 -> mailbox relayed to its 7 members + their SEM_M1
//          K core   SEM_M1 -> CB_PRE (pre_s in column 0 of tile s, rows = tokens)
// Round 2  same path with CB_S2 / one-tile slots / SEM_G2, the root's CB_INVR -> mailbox (1/rms(h), 32 fp32) ->
//          SEM_M2 -> CB_INV.
// Every data write is completed (write barrier) before the semaphore increment that publishes it. A core waits on
// its own semaphores only; the mailbox is read by this RISC after its semaphore, so the L1 cache is invalidated
// first. CB_PRE / CB_INV are reserved (and zeroed) at the start, long before their message can arrive.
//
// Compile-time args: PC_MODE (0: post [T,1,4,1] / comb [T,1,4,4] tiles per token; 1: post [1,4,T,1] / comb
// [1,1,T,16]), then TensorAccessorArgs of post, comb, dbg (root) / dbg (K).
// Runtime args, K: T, leader (x, y), slot, is_tap_core, dbg address, [leader: root (x, y), group, 7 x member (x, y)]
//               root: T, post, comb, dbg addresses, 8 x leader (x, y)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#if defined(MHC_DMA_ZERO)
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#endif
#include "mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#if defined(MHC_SYNC_MARKS)  // Stage 14 (probe runs): round-1 marks of the K / leader roles
#define MHC_SMARK() MK_MARK()
#else
#define MHC_SMARK() \
    do {            \
    } while (0)
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"

using namespace mhc;

namespace {

FORCE_INLINE volatile tt_l1_ptr uint32_t* words(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}

FORCE_INLINE void zero_l1(uint32_t l1, uint32_t bytes) {
#if defined(MHC_DMA_ZERO)
    // Stage 14c: a NoC loopback zero write (16-B aligned tile buffers) instead of bytes / 4 CPU stores (~2.3 us per
    // 4 KB tile on this RISC); completed before return, and this RISC only writes (never reads) the zeroed bytes
    // afterwards, so no cache line of it can be stale
    Noc noc;
    noc.async_write_zeros(CoreLocalMem<uint32_t>(l1), bytes);
    noc.write_zeros_l1_barrier();
#else
    volatile tt_l1_ptr uint32_t* p = words(l1);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
#endif
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

// bytes at local l1 -> (x, y):dst_l1, completed, then +1 on (x, y)'s semaphore `sem`
FORCE_INLINE void send(uint32_t l1, uint32_t bytes, uint32_t x, uint32_t y, uint32_t dst_l1, uint32_t sem) {
    noc_async_write(l1, get_noc_addr(x, y, dst_l1), bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    noc_async_atomic_barrier();
}

#if defined(DEBUG_TAPS)
template <typename Acc>
FORCE_INLINE void tap(const Acc& dbg, uint32_t index, uint32_t l1) {
    noc_async_write_page(index, dbg, l1);
    noc_async_write_barrier();
}
#endif

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("MH-SYNC");  // device-profiler zone (no-op without the profiler)
    constexpr uint32_t PC_MODE = get_compile_time_arg_val(PH_CT0 + 0);
    const uint32_t T = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t gather_l1 = get_write_ptr(CB_GATHER);
    const uint32_t msg_l1 = get_write_ptr(CB_MSG);

#if defined(MHC_K)
    constexpr auto dbg_args = TensorAccessorArgs<PH_CT0 + 1>();
    const uint32_t leader_x = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t leader_y = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint32_t slot = get_arg_val<uint32_t>(ph_rt0 + 3);
    [[maybe_unused]] const uint32_t tap_core = get_arg_val<uint32_t>(ph_rt0 + 4);
    [[maybe_unused]] const uint32_t dbg_addr = get_arg_val<uint32_t>(ph_rt0 + 5);
#if defined(DEBUG_TAPS)
    const auto dbg = TensorAccessor(dbg_args, dbg_addr);
#endif
#if defined(MHC_LEADER)
    const uint32_t root_x = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t root_y = get_arg_val<uint32_t>(ph_rt0 + 7);
    const uint32_t group = get_arg_val<uint32_t>(ph_rt0 + 8);
    constexpr uint32_t MEMBER_ARGS = 9;  // 7 x (x, y)
#endif

    // the expanded broadcast operands: owned (reserved) from the start, zero outside column 0 rows < T
    cb_reserve_back(CB_PRE, N_STREAMS);
    cb_reserve_back(CB_INV, 1);
    const uint32_t pre_l1 = get_write_ptr(CB_PRE);
    const uint32_t inv_l1 = get_write_ptr(CB_INV);
    zero_l1(pre_l1, N_STREAMS * FP32_TILE_BYTES);
    zero_l1(inv_l1, FP32_TILE_BYTES);
#if defined(MHC_LEADER)
    cb_reserve_back(CB_GATHER, 2 * GROUP);
#endif

    // ---- round 1: (M, S) partial -> leader slot
    cb_wait_front(CB_PART, 2);
    MHC_SMARK();  // s0: this core's (M, S) partial ready
#if defined(DEBUG_TAPS)
    if (tap_core) {
        tap(dbg, TAP_K0_M, get_read_ptr(CB_PART));
        tap(dbg, TAP_K0_S, get_read_ptr(CB_PART) + FP32_TILE_BYTES);
    }
#endif
    send(
        get_read_ptr(CB_PART), 2 * FP32_TILE_BYTES, leader_x, leader_y, gather_l1 + slot * 2 * FP32_TILE_BYTES, SEM_G1);
    cb_pop_front(CB_PART, 2);

#if defined(MHC_LEADER)
    noc_semaphore_wait(sem_ptr(SEM_G1), GROUP);
    MHC_SMARK();                            // s1 (leader): the group's 8 partials gathered
    noc_semaphore_set(sem_ptr(SEM_G1), 0);  // reset for a later instance in the same program
    cb_push_back(CB_GATHER, 2 * GROUP);
    cb_wait_front(CB_GSUM, 2);
    MHC_SMARK();  // s2 (leader): the group sum ready (sent to the root next)
    send(get_read_ptr(CB_GSUM), 2 * FP32_TILE_BYTES, root_x, root_y, gather_l1 + group * 2 * FP32_TILE_BYTES, SEM_G1);
    cb_pop_front(CB_GSUM, 2);
#endif

    // ---- round-1 message: pre_s(t) at msg[32 s + t]
    noc_semaphore_wait(sem_ptr(SEM_M1), 1);
    MHC_SMARK();                            // s3: the pre message in
    noc_semaphore_set(sem_ptr(SEM_M1), 0);  // reset for a later instance in the same program
#if defined(MHC_LEADER)
    for (uint32_t m = 0; m < GROUP - 1; ++m) {
        noc_async_write(
            msg_l1,
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m),
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m + 1),
                msg_l1),
            MSG_FLOATS * 4);
    }
    noc_async_write_barrier();
    for (uint32_t m = 0; m < GROUP - 1; ++m) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m),
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m + 1),
                get_semaphore(SEM_M1)),
            1);
    }
    noc_async_atomic_barrier();
#endif
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            volatile tt_l1_ptr uint32_t* tile = words(pre_l1 + s * FP32_TILE_BYTES);
            for (uint32_t t = 0; t < T; ++t) {
#if defined(MHC_FPU_COLLAPSE)
                tile[tile_off(t, t)] = msg[s * TILE + t];  // diag(pre_s): the collapse is D_s @ x_s on the FPU
#else
                tile[tile_off(t, 0)] = msg[s * TILE + t];
#endif
            }
        }
    }
#if defined(DEBUG_TAPS)
    if (tap_core) {
        tap(dbg, TAP_K0_PRE0, pre_l1);
    }
#endif
    cb_push_back(CB_PRE, N_STREAMS);

    // ---- round 2: S2 partial -> leader slot (one tile per slot)
#if defined(MHC_LEADER)
    cb_reserve_back(CB_GATHER, GROUP);
#endif
    cb_wait_front(CB_S2, 1);
    send(get_read_ptr(CB_S2), FP32_TILE_BYTES, leader_x, leader_y, gather_l1 + slot * FP32_TILE_BYTES, SEM_G2);
    cb_pop_front(CB_S2, 1);
#if defined(MHC_LEADER)
    noc_semaphore_wait(sem_ptr(SEM_G2), GROUP);
    noc_semaphore_set(sem_ptr(SEM_G2), 0);  // reset for a later instance in the same program
    cb_push_back(CB_GATHER, GROUP);
    cb_wait_front(CB_GSUM, 1);
    send(get_read_ptr(CB_GSUM), FP32_TILE_BYTES, root_x, root_y, gather_l1 + group * FP32_TILE_BYTES, SEM_G2);
    cb_pop_front(CB_GSUM, 1);
#endif

    // ---- round-2 message: 1/rms(h)(t) at msg[t]
    noc_semaphore_wait(sem_ptr(SEM_M2), 1);
    noc_semaphore_set(sem_ptr(SEM_M2), 0);  // reset for a later instance in the same program
#if defined(MHC_LEADER)
    for (uint32_t m = 0; m < GROUP - 1; ++m) {
        noc_async_write(
            msg_l1,
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m),
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m + 1),
                msg_l1),
            TILE * 4);
    }
    noc_async_write_barrier();
    for (uint32_t m = 0; m < GROUP - 1; ++m) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m),
                get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m + 1),
                get_semaphore(SEM_M2)),
            1);
    }
    noc_async_atomic_barrier();
#endif
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        volatile tt_l1_ptr uint32_t* tile = words(inv_l1);
        for (uint32_t t = 0; t < T; ++t) {
            tile[tile_off(t, 0)] = msg[t];
        }
    }
#if defined(DEBUG_TAPS)
    if (tap_core) {
        tap(dbg, TAP_K0_INV, inv_l1);
    }
#endif
    cb_push_back(CB_INV, 1);

#elif defined(MHC_ROOT)
    MK_MARK();  // 0: start
    constexpr auto post_args = TensorAccessorArgs<PH_CT0 + 1>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto dbg_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto coef_args = TensorAccessorArgs<dbg_args.next_compile_time_args_offset()>();
    const uint32_t post_addr = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t comb_addr = get_arg_val<uint32_t>(ph_rt0 + 2);
    [[maybe_unused]] const uint32_t dbg_addr = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t coef_addr = get_arg_val<uint32_t>(ph_rt0 + 4);
    constexpr uint32_t LEADER_ARGS = 5;  // 8 x (x, y)
    const auto post = TensorAccessor(post_args, post_addr);
    const auto comb = TensorAccessor(comb_args, comb_addr);
    const auto coef_out = TensorAccessor(coef_args, coef_addr);
#if defined(MHC_DEFER_INV)
    // Stage 14: runtime args after the leaders: invh address, n readers, n x (x, y); compile-time: invh's accessor
    constexpr auto invh_args = TensorAccessorArgs<coef_args.next_compile_time_args_offset()>();
    constexpr uint32_t INVH_ARGS = LEADER_ARGS + 2 * N_GROUPS;
    const auto invh_out = TensorAccessor(invh_args, get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS));
    const uint32_t n_invh = get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS + 1);
#endif
#if defined(DEBUG_TAPS)
    const auto dbg = TensorAccessor(dbg_args, dbg_addr);
#endif

    // ---- round 1: group partials
    cb_reserve_back(CB_GATHER, 2 * N_GROUPS);
    noc_semaphore_wait(sem_ptr(SEM_G1), N_GROUPS);
    MHC_SMARK();                            // root: the 8 group sums gathered
    noc_semaphore_set(sem_ptr(SEM_G1), 0);  // reset for a later instance in the same program
    cb_push_back(CB_GATHER, 2 * N_GROUPS);

    // z (rows = tokens, cols = the 24 mixes) -> plane layout (plane v, lane t); unused lanes / planes zero
    cb_wait_front(CB_RED, 1);
    MK_MARK();              // 1: z (round-1 sums) ready on the root
    invalidate_l1_cache();  // packer-written data read by this RISC
#if defined(DEBUG_TAPS)
    cb_wait_front(CB_DBG, 2);
    tap(dbg, TAP_M_TOTAL, get_read_ptr(CB_DBG));
    tap(dbg, TAP_S_TOTAL, get_read_ptr(CB_DBG) + FP32_TILE_BYTES);
    cb_pop_front(CB_DBG, 2);
    tap(dbg, TAP_Z, get_read_ptr(CB_RED));
#endif
    // z planes [v0, v1) -> the next CB_PLANE page (the rest of it zero)
    auto relayout = [&](uint32_t v0, uint32_t v1) {
        cb_reserve_back(CB_PLANE, 1);
        const uint32_t plane_l1 = get_write_ptr(CB_PLANE);
        zero_l1(plane_l1, FP32_TILE_BYTES);
#if defined(MHC_FAST_RELAYOUT)
        // 7a: plain loads / stores (the compiler overlaps them), token-major (z row t is contiguous within a face);
        // z was fenced in after the CB_RED wait (packer-written), and this RISC only writes the zeroed plane page: the
        // fence below lands every store before the push (Blackhole invalidate_l1_cache = fence)
        const uint32_t* z = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_RED));
        uint32_t* plane = reinterpret_cast<uint32_t*>(plane_l1);
        for (uint32_t t = 0; t < T; ++t) {
#pragma GCC unroll 8
            for (uint32_t v = v0; v < v1; ++v) {
                plane[plane_off(v, t)] = z[tile_off(t, v)];
            }
        }
        invalidate_l1_cache();
#else
        volatile tt_l1_ptr uint32_t* z = words(get_read_ptr(CB_RED));
        volatile tt_l1_ptr uint32_t* plane = words(plane_l1);
        for (uint32_t v = v0; v < v1; ++v) {
            for (uint32_t t = 0; t < T; ++t) {
                plane[plane_off(v, t)] = z[tile_off(t, v)];
            }
        }
#endif
#if defined(DEBUG_TAPS)
        tap(dbg, TAP_PLANE, plane_l1);
#endif
        cb_push_back(CB_PLANE, 1);
    };
#if defined(MHC_PRE_SPLIT)
    // 7a (B=32 probe: z -> pre planes 14.9 us, the 24 x T element relayout on this RISC): page 0 = the 4 pre planes
    // only (4 T elements), the pre message goes out from its coefficient page, THEN page 1 = every plane (the Sinkhorn
    // input; post / comb are needed only by mhc_post, much later)
    relayout(0, N_STREAMS);
#else
    relayout(0, N_MIX);
    cb_pop_front(CB_RED, 1);
#endif

    // coefficients -> mailbox (pre) -> leaders; post / comb output tiles
    cb_wait_front(CB_COEF, 1);
#if defined(MHC_PRE_EARLY)
    // Stage 14: page 0 holds the pre planes (ahead of the Sinkhorn): the message goes out now, post / comb from page 1
    MHC_SMARK();  // root: pre planes ready
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* coef = words(get_read_ptr(CB_COEF));
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        for (uint32_t i = 0; i < MSG_FLOATS; ++i) {
            msg[i] = 0;
        }
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t t = 0; t < T; ++t) {
                msg[s * TILE + t] = coef[plane_off(s, t)];
            }
        }
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_async_write(
                msg_l1,
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    msg_l1),
                MSG_FLOATS * 4);
        }
        noc_async_write_barrier();
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    get_semaphore(SEM_M1)),
                1);
        }
        noc_async_atomic_barrier();
    }
    cb_pop_front(CB_COEF, 1);
#if defined(MHC_PRE_SPLIT)
    relayout(0, N_MIX);
    cb_pop_front(CB_RED, 1);
#endif
    cb_wait_front(CB_COEF, 1);
#endif
    MK_MARK();  // 2: coefficients (Sinkhorn) done
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* coef = words(get_read_ptr(CB_COEF));
#if defined(DEBUG_TAPS)
        tap(dbg, TAP_COEF, get_read_ptr(CB_COEF));
#endif
#if !defined(MHC_PRE_EARLY)
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        for (uint32_t i = 0; i < MSG_FLOATS; ++i) {
            msg[i] = 0;
        }
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t t = 0; t < T; ++t) {
                msg[s * TILE + t] = coef[plane_off(s, t)];
            }
        }
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_async_write(
                msg_l1,
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    msg_l1),
                MSG_FLOATS * 4);
        }
        noc_async_write_barrier();
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    get_semaphore(SEM_M1)),
                1);
        }
        noc_async_atomic_barrier();
#endif

#if defined(MHC_COEF_PLANE)
        // Stage 14c "cp": the coefficient page as computed (plane N_STREAMS + v = compact coefficient v, token t in
        // lane t) -> the coefficient tensor; post / comb are not written (the megakernel's mhc_post reads only this)
        static_assert(PC_MODE == 1, "MHC_COEF_PLANE: stream-major post / comb (t > 4) only");
        noc_async_write_page(0, coef_out, get_read_ptr(CB_COEF));
        noc_async_write_barrier();
#else
        cb_reserve_back(CB_SCR, 1);
        const uint32_t scr = get_write_ptr(CB_SCR);
        volatile tt_l1_ptr uint32_t* o = words(scr);
        if constexpr (PC_MODE == 0) {
            // post [T, 1, 4, 1]: tile t, column 0 rows i; comb [T, 1, 4, 4]: tile t rows j, cols k
            for (uint32_t t = 0; t < T; ++t) {
                zero_l1(scr, FP32_TILE_BYTES);
                for (uint32_t i = 0; i < N_STREAMS; ++i) {
                    o[tile_off(i, 0)] = coef[plane_off(4 + i, t)];
                }
                noc_async_write_page(t, post, scr);
                noc_async_write_barrier();
                zero_l1(scr, FP32_TILE_BYTES);
                for (uint32_t j = 0; j < N_STREAMS; ++j) {
                    for (uint32_t k = 0; k < N_STREAMS; ++k) {
                        o[tile_off(j, k)] = coef[plane_off(8 + 4 * j + k, t)];
                    }
                }
                noc_async_write_page(t, comb, scr);
                noc_async_write_barrier();
            }
        } else {
            // post [1, 4, T, 1]: tile s, column 0 rows t; comb [1, 1, T, 16]: rows t, cols 4j + k
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                zero_l1(scr, FP32_TILE_BYTES);
                for (uint32_t t = 0; t < T; ++t) {
                    o[tile_off(t, 0)] = coef[plane_off(4 + s, t)];
                }
                noc_async_write_page(s, post, scr);
                noc_async_write_barrier();
            }
            zero_l1(scr, FP32_TILE_BYTES);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t e = 0; e < 16; ++e) {
                    o[tile_off(t, e)] = coef[plane_off(8 + e, t)];
                }
            }
            noc_async_write_page(0, comb, scr);
            noc_async_write_barrier();
        }
        // compact coefficients for the fused post-mix (mhc_post): row t = post_0..3 | comb_0..15 (C[i][j] at 4i + j)
        zero_l1(scr, FP32_TILE_BYTES);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t v = 0; v < N_MIX - N_STREAMS; ++v) {
                o[tile_off(t, v)] = coef[plane_off(N_STREAMS + v, t)];
            }
        }
        noc_async_write_page(0, coef_out, scr);
        noc_async_write_barrier();
#endif  // MHC_COEF_PLANE
    }
    cb_pop_front(CB_COEF, 1);

    // ---- round 2: group S2 partials -> 1/rms(h) -> mailbox -> leaders
    cb_reserve_back(CB_GATHER, N_GROUPS);
    noc_semaphore_wait(sem_ptr(SEM_G2), N_GROUPS);
    noc_semaphore_set(sem_ptr(SEM_G2), 0);  // reset for a later instance in the same program
    cb_push_back(CB_GATHER, N_GROUPS);
    cb_wait_front(CB_INVR, 1);
    MK_MARK();  // 3: round-2 (1/rms(h)) ready
    invalidate_l1_cache();
#if defined(MHC_DEFER_INV)
    // the readers scale their hn-matmul outputs by it: the tile (row t, column 0 = 1/rms(h)(t)) -> invh, then +1 on
    // every reader's MK_INVH_SEM
    noc_async_write_page(0, invh_out, get_read_ptr(CB_INVR));
    noc_async_write_barrier();
    for (uint32_t r = 0; r < n_invh; ++r) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS + 2 + 2 * r),
                get_arg_val<uint32_t>(ph_rt0 + INVH_ARGS + 3 + 2 * r),
                get_semaphore(MK_INVH_SEM)),
            1);
    }
    noc_async_atomic_barrier();
#endif
    {
        volatile tt_l1_ptr uint32_t* inv = words(get_read_ptr(CB_INVR));
#if defined(DEBUG_TAPS)
        tap(dbg, TAP_INVR, get_read_ptr(CB_INVR));
#endif
        volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
        for (uint32_t t = 0; t < TILE; ++t) {
            msg[t] = t < T ? inv[tile_off(t, 0)] : 0u;
        }
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_async_write(
                msg_l1,
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    msg_l1),
                TILE * 4);
        }
        noc_async_write_barrier();
        for (uint32_t g = 0; g < N_GROUPS; ++g) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g),
                    get_arg_val<uint32_t>(ph_rt0 + LEADER_ARGS + 2 * g + 1),
                    get_semaphore(SEM_M2)),
                1);
        }
        noc_async_atomic_barrier();
    }
    cb_pop_front(CB_INVR, 1);
    MK_MARK();  // 4: end
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
#endif
}
