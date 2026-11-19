// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel routed experts — compute of a DS bank reader (mkx_common.hpp), fused_experts' math (matmul_gate_up.cpp):
// per expert of the union: for each owned shard, gate | up = x @ gate_up shard (K accumulated in DST over the ring
// blocks) -> fp32 CB_MM -> SwiGLU silu(clamp(gate, -inf, L)) * clamp(up, -L, L) -> bfloat8_b m tile (CB_MO, to the
// gather); then on the gathered m (CB_M): for each owned shard, (m @ down shard) -> fp32 CB_MM -> x the routing weight
// tile -> bf16, added to the shard's running sum (bf16) — the last expert's sums go to CB_Y.
// MKX_XTP (expert TP, mkx_common.hpp): one gate_up per expert over this reader's K range [K0, K0 + KLEN); a sender
// packs its fp32 gate | up partial to CB_PS (the NCRISC sends it to the bank's leader); the leader adds the two
// senders' partials from CB_PART in K order ((own + sender 1) + sender 2, fp32 SFPU) before SwiGLU; down over the
// slice's I_TILES (8) gathered m tiles. MKX_XSL (batched expert TP): per group of 8 experts, the owned shards' m tiles
// (full K, x the expert's routing weight from CB_RS; a zero tile past the union), then per owned shard the down over
// K = KT_DN rows x the group's experts, fp32 running sums. Compile-time args: KT_GU, KT_DN, KB, I_TILES, LIMIT_BITS.
// Runtime args: NSH (+ MKX_XTP: role, K0, KLEN; MKX_XSL: the 3 owned shard indices, padded). Compute config: math
// fidelity of the call (LoFi at decode), fp32 DST, default unpack modes (= fused_experts; MKX_XTP / MKX_XSL: CB_ACC,
// MKX_XSL: CB_MM unpacked to DST in fp32).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

#ifndef MKX_MMBLOCK
#define MKX_MMBLOCK 1
#endif

using namespace mkx;

void kernel_main() {
    constexpr uint32_t KT_GU = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t KT_DN = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t I_TILES = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t LIMIT_BITS = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t NEG_INF_BITS = 0xFF800000u;
    constexpr uint32_t NEG_LIMIT_BITS = LIMIT_BITS ^ 0x80000000u;
    constexpr uint32_t BLOCK_TILES = KB * BLOCK_TILES_COLS;
    const uint32_t nsh = get_arg_val<uint32_t>(ph_rt0 + 0);
#if defined(MK_LAYERS) && MK_LAYERS > 1
    // Stage 14b stack program (kernels/mk/stack.hpp): this phase runs once per layer; its CBs start at their bases
    // every time (all empty at a layer boundary; the ring's slot addressing restarts at 0 below)
#if defined(MKX_Y2_INIT)
    // + CB_Y2 (MKX_CB0 + 11, fold_y2 / TP_SHARED): a reader owning fewer shards than the CB's depth pushes a part of it
    // per layer, so without the rewind its pointers drift past fifo_limit from layer 1 on (2026-10-20 review)
    mk::cb_rewind_range(MKX_CB0, 12);
#else
    mk::cb_rewind_range(MKX_CB0, 11);
#endif
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_RING, CB_MM);
    cb_wait_front(CB_LIST, 1);
    const uint32_t n = read_tile_value(CB_LIST, 0, 0);
    cb_pop_front(CB_LIST, 1);
#if !defined(MKX_XPROG)
    cb_wait_front(CB_X, KT_GU);  // MKX_XPROG: waited per K block in shard_matmul (xa.cpp pushes x in chunks)
#endif
#if defined(MKX_Y2_INIT)
    // y2 (another phase's output tiles of this reader's shards) joins the routed sum at its LAST step, in the fp32 DST
    // (one bf16 rounding of routed + y2, like the chain's fp32 add); with no local expert y2 is the output
    constexpr bool Y2 = true;
    cb_wait_front(CB_Y2, 2 * nsh);
    if (n == 0) {
        const uint32_t dst_cb = CB_Y;
        copy_init(CB_Y2);
        reconfig_data_format_srca(CB_Y2);
        pack_reconfig_data_format(dst_cb);
        for (uint32_t s = 0; s < nsh; ++s) {
            cb_reserve_back(dst_cb, 2);
            tile_regs_acquire();
            copy_tile(CB_Y2, 2 * s, 0);
            copy_tile(CB_Y2, 2 * s + 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst_cb);
            pack_tile(1, dst_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 2);
        }
    }
#else
    constexpr bool Y2 = false;
#endif

    // in0 tiles [k0, k0 + kt) of the activation CB @ the next kt / KB ring blocks of one shard -> 2 fp32 tiles of
    // cb_out
    // the K loop of one shard into DST 0 / 1 (accumulating onto whatever DST holds: MKX_XSL_SEED seeds it)
    auto mm_init_k = [&](uint32_t cb_in0) {
#if MKX_MMBLOCK
        // both shard columns per k in one call (one in0 unpack for the 2 weight tiles; same k order per output tile)
        matmul_block_init(cb_in0, CB_RING, 0, BLOCK_TILES_COLS, 1, 1);
#else
        matmul_init(cb_in0, CB_RING);
#endif
        reconfig_data_format(CB_RING, cb_in0);
    };
    auto mm_k = [&](uint32_t cb_in0, uint32_t k0, uint32_t kt) {
        for (uint32_t kb = 0; kb < kt; kb += KB) {
#if defined(MKX_XPROG)
            if (cb_in0 == CB_X) {
                cb_wait_front(CB_X, k0 + kb + KB);  // cumulative: x is never popped before the end
            }
#endif
            cb_wait_front(CB_RING, BLOCK_TILES);
            for (uint32_t k = 0; k < KB; ++k) {
#if MKX_MMBLOCK
                matmul_block(cb_in0, CB_RING, k0 + kb + k, k * BLOCK_TILES_COLS, 0, 0, BLOCK_TILES_COLS, 1, 1);
#else
                matmul_tiles(cb_in0, CB_RING, k0 + kb + k, k * BLOCK_TILES_COLS + 0, 0);
                matmul_tiles(cb_in0, CB_RING, k0 + kb + k, k * BLOCK_TILES_COLS + 1, 1);
#endif
            }
            cb_pop_front(CB_RING, BLOCK_TILES);
        }
    };
    auto shard_matmul = [&](uint32_t cb_in0, uint32_t k0, uint32_t kt, uint32_t cb_out) {
        mm_init_k(cb_in0);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, 2);
        tile_regs_acquire();
        mm_k(cb_in0, k0, kt);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 2);
    };

    // expert pos's gate_up runs before expert pos - 1's down: the m all-gather of pos - 1 (activation mover, all
    // readers) overlaps it
#if defined(MKX_XTP)
    const uint32_t xrole = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t xk0 = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint32_t xklen = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t ngu = xrole == 0 ? 1 : 0;  // the leader makes the bank's one m tile
#else
    const uint32_t ngu = nsh;
#endif
#if defined(MKX_XSL)
    // batched expert TP: groups of GS = 64 / W experts (pos = the group, W = KT_DN the slice width); owned shard j
    // carries the group's expert j / W
    constexpr uint32_t GS = 64 / KT_DN;
    uint32_t xq[3];
    for (uint32_t s = 0; s < 3; ++s) {
        xq[s] = get_arg_val<uint32_t>(ph_rt0 + 1 + s) / KT_DN;
    }
    const uint32_t npos = (n + GS - 1) / GS;
    uint32_t gpos = 0;  // gate_up's group (the lambda takes no argument)
#else
    const uint32_t npos = n;
#endif
    auto gate_up = [&]() {
#if defined(MKX_XTP)
        if (xrole != 0) {
            shard_matmul(CB_X, xk0, xklen, CB_PS);  // this reader's K part, sent to the leader
            return;
        }
#endif
        // ---- gate_up + SwiGLU, one m tile per owned shard (expert TP: the leader's one m tile)
        for (uint32_t s = 0; s < ngu; ++s) {
#if defined(MKX_XSL)
            if (GS * gpos + xq[s] >= n) {  // past the union: a zero m tile (the down's K stops before it)
                pack_reconfig_data_format(CB_MO);
                cb_reserve_back(CB_MO, 1);
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, 0.0f);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_MO);
                tile_regs_release();
                cb_push_back(CB_MO, 1);
                continue;
            }
#endif
#if defined(MKX_XTP)
            shard_matmul(CB_X, xk0, xklen, CB_MM);
            cb_wait_front(CB_PART, XTP_PART_SLOT_TILES);
#else
            shard_matmul(CB_X, 0, KT_GU, CB_MM);
#endif
            cb_wait_front(CB_MM, 2);
            copy_init(CB_MM);
            reconfig_data_format_srca(CB_MM);
            pack_reconfig_data_format(CB_MO);
            cb_reserve_back(CB_MO, 1);
            tile_regs_acquire();
            copy_tile(CB_MM, 0, 0);
            copy_tile(CB_MM, 1, 1);
#if defined(MKX_XINV)
            // the deferred FFN 1/rms(h): gate, up x the per-row inv (column 0 of CB_XINV) before SwiGLU
            cb_wait_front(CB_XINV, 1);  // (held to the end; repeated waits return at once)
            copy_init(CB_XINV);
            reconfig_data_format_srca(CB_XINV);
            copy_tile(CB_XINV, 0, 2);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 2);
            sfpu_mul_bcast_col(1, 2);
#endif
#if defined(MKX_XTP)
            // + sender 1's (gate, up), then + sender 2's: a fixed K order on every bank
            for (uint32_t q = 0; q < 2; ++q) {
                copy_init(CB_PART);
                reconfig_data_format_srca(CB_PART);
                copy_tile(CB_PART, 2 * q, 2);
                copy_tile(CB_PART, 2 * q + 1, 3);
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }
#endif
            clamp_tile_init();
            clamp_tile(0, NEG_INF_BITS, LIMIT_BITS);
            silu_tile_init();
            silu_tile(0);
            clamp_tile_init();
            clamp_tile(1, NEG_LIMIT_BITS, LIMIT_BITS);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#if defined(MKX_XSL)
            // x the expert's routing weight (row b = w[e, b]) in fp32 before the bfloat8_b m: the down then sums the
            // group's experts in one K loop (the 8 slices' outputs sum in the all-reduce)
            cb_wait_front(CB_RS, 1);  // this shard's expert's weight tile (xa.cpp, pushed a group ahead)
            copy_init(CB_RS);
            reconfig_data_format_srca(CB_RS);
            copy_tile(CB_RS, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_MO);
            tile_regs_release();
            cb_pop_front(CB_MM, 2);
#if defined(MKX_XTP)
            cb_pop_front(CB_PART, XTP_PART_SLOT_TILES);
#endif
#if defined(MKX_XSL)
            cb_pop_front(CB_RS, 1);
#endif
            cb_push_back(CB_MO, 1);
        }
#if defined(MKX_XSL)
        ++gpos;
#endif
    };
    // DST[0..1] += y2 tiles (2 s, 2 s + 1) of CB_Y2 (fp32 SFPU add; the caller packs once)
    auto add_y2 = [&](uint32_t s) {
#if defined(MKX_Y2_INIT)
        copy_init(CB_Y2);
        reconfig_data_format_srca(CB_Y2);
        copy_tile(CB_Y2, 2 * s, 2);
        copy_tile(CB_Y2, 2 * s + 1, 3);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        add_binary_tile(1, 3, 1);
#else
        (void)s;
#endif
    };
    auto down = [&](uint32_t pos) {
        // ---- down on the gathered m, weighted, accumulated
        cb_wait_front(CB_M, I_TILES);
#if defined(MKX_XSL)
        {
            // K = KT_DN slice rows per expert of the group (m already weighted), fp32 running sums (CB_ACC, unpacked
            // straight to DST like CB_MM: exact copies), one bf16 rounding at the last group
            const bool first = pos == 0, last = pos + 1 == npos;
            const uint32_t nq = n - GS * pos < GS ? n - GS * pos : GS;
#if defined(MKX_XSL_SEED)
            // the running sum SEEDS DST and this group's K accumulates on top of it: a row's non-zero products form ONE
            // sequential fp32 chain in union order whatever the grouping. S + G (below) re-associates a row's sum at
            // the group boundaries, which move with the OTHER rows' routing (the union) -> a live row's bits depended
            // on its co-rows (batch invariance, 14b 2026-10-22: 9's abortmix). Unselected experts add exact zeros.
            for (uint32_t s = 0; s < nsh; ++s) {
                const uint32_t dst = last ? CB_Y : CB_ACC;
                if (!first) {
                    cb_wait_front(CB_ACC, 2);  // this shard's running sum (FIFO: shards in order every group)
                }
                pack_reconfig_data_format(dst);
                cb_reserve_back(dst, 2);  // CB_ACC has 2 spare tiles beyond the 2 per shard it holds
                tile_regs_acquire();
                if (!first) {
                    copy_init(CB_ACC);
                    reconfig_data_format_srca(CB_ACC);
                    copy_tile(CB_ACC, 0, 0);
                    copy_tile(CB_ACC, 1, 1);
                }
                mm_init_k(CB_M);
                mm_k(CB_M, 0, KT_DN * nq);
                if (Y2 && last) {
                    add_y2(s);
                }
                tile_regs_commit();
                if (!first) {
                    cb_pop_front(CB_ACC, 2);
                }
                tile_regs_wait();
                pack_tile(0, dst);
                pack_tile(1, dst);
                tile_regs_release();
                cb_push_back(dst, 2);
            }
            cb_pop_front(CB_M, I_TILES);
            return;
#endif
            for (uint32_t s = 0; s < nsh; ++s) {
                shard_matmul(CB_M, 0, KT_DN * nq, CB_MM);
                const uint32_t dst = last ? CB_Y : CB_ACC;
                cb_wait_front(CB_MM, 2);
                copy_init(CB_MM);
                reconfig_data_format_srca(CB_MM);
                pack_reconfig_data_format(dst);
                cb_reserve_back(dst, 2);
                tile_regs_acquire();
                copy_tile(CB_MM, 0, 0);
                copy_tile(CB_MM, 1, 1);
                if (!first) {
                    cb_wait_front(CB_ACC, 2);  // this shard's running sum (FIFO: shards in order every group)
                    copy_init(CB_ACC);
                    reconfig_data_format_srca(CB_ACC);
                    copy_tile(CB_ACC, 0, 2);
                    copy_tile(CB_ACC, 1, 3);
                    add_binary_tile_init();
                    add_binary_tile(0, 2, 0);
                    add_binary_tile(1, 3, 1);
                }
                if (Y2 && last) {
                    add_y2(s);
                }
                tile_regs_commit();
                cb_pop_front(CB_MM, 2);
                if (!first) {
                    cb_pop_front(CB_ACC, 2);
                }
                tile_regs_wait();
                pack_tile(0, dst);
                pack_tile(1, dst);
                tile_regs_release();
                cb_push_back(dst, 2);
            }
            cb_pop_front(CB_M, I_TILES);
            return;
        }
#endif
        cb_wait_front(CB_RS, 1);
        const bool first = pos == 0, last = pos + 1 == n;
#if defined(MKX_XTP)
        // expert TP: a slice's partial products partly cancel across the 8 slices, so its running sum stays fp32
        // (CB_ACC fp32, unpacked straight to DST; SFPU adds) and is rounded to bf16 once, at the last expert (bf16
        // steps per expert summed over 8 slices: y PCC 0.9998 vs the full program, A4 2026-10-20)
        for (uint32_t s = 0; s < nsh; ++s) {
            shard_matmul(CB_M, 0, KT_DN, CB_MM);
            const uint32_t dst = last ? CB_Y : CB_ACC;
            cb_wait_front(CB_MM, 2);
            mul_init(CB_MM, CB_RS);
            reconfig_data_format(CB_MM, CB_RS);
            pack_reconfig_data_format(dst);
            cb_reserve_back(dst, 2);
            tile_regs_acquire();
            mul_tiles(CB_MM, CB_RS, 0, 0, 0);
            mul_tiles(CB_MM, CB_RS, 1, 0, 1);
            if (!first) {
                cb_wait_front(CB_ACC, 2);  // this shard's running sum (FIFO: shards in order every expert)
                copy_init(CB_ACC);
                reconfig_data_format_srca(CB_ACC);
                copy_tile(CB_ACC, 0, 2);
                copy_tile(CB_ACC, 1, 3);
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }
            if (Y2 && last) {
                add_y2(s);
            }
            tile_regs_commit();
            cb_pop_front(CB_MM, 2);
            if (!first) {
                cb_pop_front(CB_ACC, 2);
            }
            tile_regs_wait();
            pack_tile(0, dst);
            pack_tile(1, dst);
            tile_regs_release();
            cb_push_back(dst, 2);
        }
        cb_pop_front(CB_RS, 1);
        cb_pop_front(CB_M, I_TILES);
        return;
#endif
        // MKX_Y2_INIT (fold_y2): y2 is added to the FINISHED bf16 routed sum in a pass of its own, y = r(S + y2) for
        // every row. Folding it into the last expert's add (one rounding of S' + p_last + y2) made a row's bits depend
        // on whether ANOTHER row selected the union's last expert (row b's own last step is then r(S_b + 0) and its sum
        // was already rounded): batch invariance broke whenever the other rows' routing changed the union's tail
        // (Stage 14 2026-10-20, 14b's bisect #82 / #85). The finished sum goes through CB_WTMP (CB_ACC is the shards'
        // FIFO: an immediate pop there would take the next shard's sum).
#if defined(MKX_Y2_FOLD_LAST)  // the previous fold (one rounding of S' + p_last + y2; NOT batch invariant), A/B only
        constexpr bool Y2_SEP = false;
#else
        constexpr bool Y2_SEP = true;
#endif
        for (uint32_t s = 0; s < nsh; ++s) {
            shard_matmul(CB_M, 0, KT_DN, CB_MM);
            const bool y2_pass = Y2 && Y2_SEP && last;
            const uint32_t mul_dst = first ? (last ? (y2_pass ? CB_WTMP : CB_Y) : CB_ACC) : CB_WTMP;
            cb_wait_front(CB_MM, 2);
            mul_init(CB_MM, CB_RS);
            reconfig_data_format(CB_MM, CB_RS);
            pack_reconfig_data_format(mul_dst);
            cb_reserve_back(mul_dst, 2);
            tile_regs_acquire();
            mul_tiles(CB_MM, CB_RS, 0, 0, 0);
            mul_tiles(CB_MM, CB_RS, 1, 0, 1);
            if (Y2 && !Y2_SEP && first && last) {
                add_y2(s);
            }
            tile_regs_commit();
            cb_pop_front(CB_MM, 2);
            tile_regs_wait();
            pack_tile(0, mul_dst);
            pack_tile(1, mul_dst);
            tile_regs_release();
            cb_push_back(mul_dst, 2);
            if (!first) {
                const uint32_t add_dst = last ? (y2_pass ? CB_WTMP : CB_Y) : CB_ACC;
                cb_wait_front(CB_ACC, 2);  // this shard's running sum (FIFO: shards in order every expert)
                cb_wait_front(CB_WTMP, 2);
                add_init(CB_ACC, CB_WTMP);
                reconfig_data_format(CB_ACC, CB_WTMP);
                pack_reconfig_data_format(add_dst);
                if (!y2_pass) {
                    cb_reserve_back(add_dst, 2);  // CB_ACC has 2 spare tiles beyond the 2 per shard it holds
                }
                tile_regs_acquire();
                add_tiles(CB_ACC, CB_WTMP, 0, 0, 0);
                add_tiles(CB_ACC, CB_WTMP, 1, 1, 1);
                if (Y2 && !Y2_SEP && last) {
                    add_y2(s);
                }
                tile_regs_commit();
                cb_pop_front(CB_ACC, 2);
                cb_pop_front(CB_WTMP, 2);
                if (y2_pass) {
                    cb_reserve_back(CB_WTMP, 2);  // the 2 operand tiles just popped: the finished sum goes back there
                }
                tile_regs_wait();
                pack_tile(0, add_dst);
                pack_tile(1, add_dst);
                tile_regs_release();
                cb_push_back(add_dst, 2);
            }
            if (y2_pass) {  // y = r(S + y2): the finished routed sum (CB_WTMP) + this shard's y2 tiles
                cb_wait_front(CB_WTMP, 2);
                add_init(CB_WTMP, CB_Y2);
                reconfig_data_format(CB_WTMP, CB_Y2);
                pack_reconfig_data_format(CB_Y);
                cb_reserve_back(CB_Y, 2);
                tile_regs_acquire();
                add_tiles(CB_WTMP, CB_Y2, 0, 2 * s, 0);
                add_tiles(CB_WTMP, CB_Y2, 1, 2 * s + 1, 1);
                tile_regs_commit();
                cb_pop_front(CB_WTMP, 2);
                tile_regs_wait();
                pack_tile(0, CB_Y);
                pack_tile(1, CB_Y);
                tile_regs_release();
                cb_push_back(CB_Y, 2);
            }
        }
        cb_pop_front(CB_RS, 1);
        cb_pop_front(CB_M, I_TILES);
    };
    if (n == 0 && !Y2) {  // no local expert selected (expert parallelism): this chip's partial output is zero
        pack_reconfig_data_format(CB_Y);
        for (uint32_t s = 0; s < nsh; ++s) {
            cb_reserve_back(CB_Y, 2);
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, 0.0f);
            fill_tile(1, 0.0f);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_Y);
            pack_tile(1, CB_Y);
            tile_regs_release();
            cb_push_back(CB_Y, 2);
        }
    }
    if (npos > 0) {
        gate_up();
        for (uint32_t pos = 1; pos < npos; ++pos) {
            gate_up();
            down(pos - 1);
        }
        down(npos - 1);
    }
#if defined(MKX_Y2_INIT)
    cb_pop_front(CB_Y2, 2 * nsh);
#endif
#if defined(MKX_XINV)
    cb_wait_front(CB_XINV, 1);  // (also with no local expert)
    cb_pop_front(CB_XINV, 1);
#endif
#if defined(MKX_XPROG)
    cb_wait_front(CB_X, KT_GU);  // every chunk landed (also with no local expert) before the pop
#endif
    cb_pop_front(CB_X, KT_GU);
}
