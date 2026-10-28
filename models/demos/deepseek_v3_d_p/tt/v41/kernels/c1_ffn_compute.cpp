// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the decode-shaped routed expert FFN of the DeepSeek-V4.1 c1 MoE (tt/v41/moe_c1.py C1FfnMatvec;
// kernels/c1_ffn_reader.cpp has the dataflow). Per selected expert, if this core owns gate/up column c: two
// one-tile dot products over the K tiles of the resident row (gate into DST 0, up into DST 1, fp32 accumulation
// over the whole K) and the clamped SiLU-GLU of the two in DST (``clamped_silu_glu_tile``, the FFN op's
// activation), packed as the column's activated tile; then, over the gathered activated row, one dot product per
// owned down column (DST j), packed to the writer. The ``done`` token after the down matmul tells the reader this
// expert's half of the activated ring is free.
//
// compile_time_args = [cb_x, cb_gate, cb_up, cb_down, cb_act, cb_actf, cb_scr, cb_done, cb_out, EPC, E, KT,
//                      NT_GU, KD, BLK_GU, BLK_D]
// runtime args      = [my_ngu, nd_count]

#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"

constexpr uint32_t NONE = 0xFFFFFFFFu;

void kernel_main() {
    const uint32_t my_ngu = get_arg_val<uint32_t>(0);
    const uint32_t nd_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_actf = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(6);
    constexpr uint32_t cb_done = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t EPC = get_compile_time_arg_val(9);
    constexpr uint32_t E = get_compile_time_arg_val(10);
    constexpr uint32_t KT = get_compile_time_arg_val(11);
    constexpr uint32_t NT_GU = get_compile_time_arg_val(12);
    constexpr uint32_t KD = get_compile_time_arg_val(13);
    constexpr uint32_t BLK_GU = get_compile_time_arg_val(14);
    constexpr uint32_t BLK_D = get_compile_time_arg_val(15);
    constexpr uint32_t PROBE = get_compile_time_arg_val(16);  // 2 / 3: no matmuls (a timing probe, wrong output)
    constexpr uint32_t TPC = get_compile_time_arg_val(17);    // gate/up column tiles a core
    static_assert(TPC >= 1 && TPC <= 2, "gate and up of TPC columns must fit four fp32 DST tiles");
    static_assert(EPC <= 32, "the expert mask is one word");

    DataflowBuffer xb(cb_x), gate(cb_gate), up(cb_up), down(cb_down), act(cb_act), actf(cb_actf), scr(cb_scr);
    DataflowBuffer done(cb_done), out(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_gate, cb_act);

    // the selected local experts as a bit mask, read by UNPACK (the only thread a CB wait synchronizes) and handed
    // to MATH and PACK through the mailbox: the three threads' control flow must agree, and MATH / PACK reading the
    // scratch themselves read it before the reader has filled it (the first attempt hung)
    uint32_t mask = 0;
    UNPACK(({
        scr.wait_front(1);
        // a TRISC's CB pointers are in 16-byte words (circular_buffer_init.h's cb_addr_shift), the reader's in bytes:
        // the first two attempts read the scratch at the word address, took garbage expert ids and hung on a wild L1
        // read
        const uint32_t scr_l1 = ((scr.get_read_ptr() << 4) + 63) & ~63u;
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1);
        volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1 + E * 4);
        for (uint32_t l = 0; l < EPC; ++l) {
            mask |= (cnt[table[l]] != 0) << l;
        }
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, mask);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, mask);
    }));
    MATH(mask = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(mask = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    if (mask == 0) {
        return;
    }
    xb.wait_front(KT);

    for (uint32_t l = 0; l < EPC; ++l) {
        if (((mask >> l) & 1) == 0) {
            continue;
        }
        if (my_ngu != NONE) {
            tile_regs_acquire();
            matmul_init(cb_x, cb_gate);
            // gate and up block by block in lockstep with the reader (which pushes one block of each per K block:
            // consuming all of gate before any of up deadlocks on the rings; the third attempt). One matmul_tiles a
            // tile: matmul_block is one K step of an rt x ct outer product, not a K loop (the fifth attempt), and
            // the compute is hidden under the reads anyway (the probes)
            for (uint32_t kb = 0; kb < KT; kb += BLK_GU) {
                gate.wait_front(BLK_GU * TPC);
                up.wait_front(BLK_GU * TPC);
                for (uint32_t t = 0; t < BLK_GU && (PROBE & 2) == 0; ++t) {
                    for (uint32_t j = 0; j < TPC; ++j) {
                        matmul_tiles(cb_x, cb_gate, kb + t, t * TPC + j, j);      // DST[j] += x[k] @ gate[k, c_j]
                        matmul_tiles(cb_x, cb_up, kb + t, t * TPC + j, TPC + j);  // DST[TPC + j] += x[k] @ up[k, c_j]
                    }
                }
                gate.pop_front(BLK_GU * TPC);
                up.pop_front(BLK_GU * TPC);
            }
            clamped_silu_glu_tile_init();
            for (uint32_t j = 0; j < TPC; ++j) {
                clamped_silu_glu_tile(j, TPC + j, j);  // DST[j] = silu(min(gate, L)) * clamp(up, -L, L)
            }
            tile_regs_commit();
            tile_regs_wait();
            act.reserve_back(TPC);
            for (uint32_t j = 0; j < TPC; ++j) {
                pack_tile(j, cb_act);
            }
            act.push_back(TPC);
            tile_regs_release();
        }
        actf.wait_front(NT_GU);
        if (nd_count > 0) {
            tile_regs_acquire();
            // one K step of a 1 x nd_count outer product a call: the activated tile is unpacked once for both
            // columns (matmul_block's kt_dim is the in0 row stride, not a K loop)
            matmul_block_init(cb_actf, cb_down, 0, nd_count, 1, 1);
            for (uint32_t kb = 0; kb < KD; kb += BLK_D) {
                down.wait_front(BLK_D * nd_count);
                for (uint32_t t = 0; t < BLK_D && (PROBE & 2) == 0; ++t) {
                    matmul_block(
                        cb_actf, cb_down, kb + t, t * nd_count, 0, 0, nd_count, 1, 1);  // DST[j] += a[k] @ down[k, n_j]
                }
                down.pop_front(BLK_D * nd_count);
            }
            tile_regs_commit();
            tile_regs_wait();
            out.reserve_back(nd_count);
            for (uint32_t j = 0; j < nd_count; ++j) {
                pack_tile(j, cb_out);
            }
            out.push_back(nd_count);
            tile_regs_release();
        }
        actf.pop_front(NT_GU);
        done.reserve_back(1);
        done.push_back(1);
    }
}
