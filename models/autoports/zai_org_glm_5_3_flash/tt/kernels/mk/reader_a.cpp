// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel reader core — activation mover (NCRISC, NoC1), every matmul phase in order: wait for the phase's
// producers (hand-off semaphore), pull the activation K blocks from the phase's input tensor into CB_IN0, then write
// this reader's NR output tiles to the phase's output tensor and signal the phase's consumers.
//
// Compile-time args: NPH, CB_IN0, CB_OUT, NCONS (consumers signalled per phase), then per phase p 8 values at 4 + 8p:
//   NB, KB, NR, WAIT_SEM, WAIT_COUNT (0 = no wait), SIGNAL_SEM, XOFF, YOFF (compile-time-arg offsets of the
//   TensorAccessorArgs of the phase's input / output tensors, appended after the phase table).
//   + per phase a 9th value: OUT_CB (the phase's output CB: bf16 CB_OUT or an fp32 one).
// Runtime args: per phase p 3 values at 3p: x address, y address, n0 (first output column tile; 0xFFFFFFFF = this
// reader
//   sits the phase out: it still waits for the producers and signals the consumers); then NCONS consumer
//   (x, y) NoC coordinates at 3 * NPH.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#if defined(MK_TOPK_K)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/topk.hpp"
#endif

// Stage 14 TP (tt/tp8_layer.py): a phase whose bit is set in MK_FAB_MASK hands its PARTIAL output to the in-program
// fabric all-reduce (tt/fabric_ar.py FabricPhase): instead of writing its NR output tiles to y and signalling the
// phase's consumers, the reader writes the VALID rows of each tile (MK_FAB_BROWS <= 16: face-0 rows then face-1 rows,
// 32 B per row per face; 32: whole tiles) into the `own` buffer of the tile's lane (lane = tile / 64) at
// (tile % 64) * V, then increments the OWN_READY word (persistent, monotonic) of BOTH lanes -- also a reader that
// sits the phase out, so every lane counts all readers whatever the work split; the lanes signal the consumers
// once the reduced result is ready. Runtime args at MK_FAB_RT: lane 0 (x, y), lane 1 (x, y), own address, lane sync.
#if defined(MK_FAB_MASK)
constexpr uint32_t FAB_CH = MK_FAB_BROWS * 32;
constexpr uint32_t FAB_V = MK_FAB_BROWS >= 32 ? 2048 : 2 * FAB_CH;
constexpr uint32_t FAB_LANE_TILES = 64;
#endif

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_IN0 = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + 3);

// koff (Stage 14 MK_KSPLIT): the phase's K range starts at activation tile koff (a K-split phase's reader reads only
// its K chunk; the host encodes koff in the upper 16 bits of n0)
template <uint32_t NB, uint32_t KB, typename XA>
FORCE_INLINE void pull_blocks(const XA& x, uint32_t koff = 0) {
    const uint32_t xpage = get_tile_size(CB_IN0);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        cb_reserve_back(CB_IN0, KB);
        const uint32_t l1 = get_write_ptr(CB_IN0);
        for (uint32_t k = 0; k < KB; ++k) {
            noc_async_read_page(koff + blk * KB + k, x, l1 + k * xpage);
        }
        noc_async_read_barrier();
        cb_push_back(CB_IN0, KB);
    }
}

#if defined(MK_SCALE_MASK)
// Stage 14 (deferred 1/rms(h), mhc_pre MHC_DEFER_INV): before the first phase of each scale group the root's
// 1/rms(h) tile (row t, column 0) -> MK_INVH_CB for the compute, which scales this group's matmul outputs. Every
// reader fetches it, also one sitting the phase out (the compute pops it after the group's last phase).
// Runtime args at MK_SCALE_RT + i: the invh tensor address of the i-th group; compile time at MK_SCALE_CT: its
// accessor.
template <uint32_t P>
FORCE_INLINE void fetch_inv() {
    constexpr uint32_t idx = __builtin_popcount(static_cast<unsigned>(MK_SCALE_FIRST) & ((1u << P) - 1u));
    constexpr auto inv_args = TensorAccessorArgs<MK_SCALE_CT>();
    const auto inv = TensorAccessor(inv_args, get_arg_val<uint32_t>(MK_SCALE_RT + idx));
    mk::wait_handoff(MK_INVH_SEM, 1);
    cb_reserve_back(MK_INVH_CB, 1);
    noc_async_read_page(0, inv, get_write_ptr(MK_INVH_CB));
    noc_async_read_barrier();
    cb_push_back(MK_INVH_CB, 1);
}
#endif

template <uint32_t NR, uint32_t OUT_CB, typename YA>
FORCE_INLINE void write_out(const YA& y, uint32_t n0, uint32_t phase) {
    cb_wait_front(OUT_CB, NR);
    const uint32_t l1 = get_read_ptr(OUT_CB);
    const uint32_t opage = get_tile_size(OUT_CB);
    for (uint32_t j = 0; j < NR; ++j) {
        noc_async_write_page(n0 + j, y, l1 + j * opage);
    }
#if defined(MK_TOPK_K)  // the model head: scan the logits tiles while they are written out (kernels/mk/topk.hpp)
    MK_MARK();          // head probe (GLM53_HEAD_PROBE): the scan's span
    mktopk::scan(phase, l1, NR, n0, opage);
    MK_MARK();
#else
    (void)phase;
#endif
    noc_async_write_barrier();
    cb_pop_front(OUT_CB, NR);
    mk::cb_rewind(OUT_CB);  // the next phase's NR output tiles start at the base again
}

template <uint32_t NB, uint32_t KB, uint32_t NR, uint32_t OUT_CB, typename XA, typename YA>
FORCE_INLINE void move_blocks(const XA& x, const YA& y, uint32_t n0, uint32_t phase, uint32_t koff = 0) {
    pull_blocks<NB, KB>(x, koff);
    write_out<NR, OUT_CB>(y, n0, phase);
}

#if defined(MK_FAB_MASK)
// the fabric phase's output side (see the header): packed valid rows -> the lanes' own buffers
// koff (Stage 17 rung 3, MK_KSPLIT): the phase's activation K range starts at tile koff (the ctx fc K-split per
// chip); 2026-10-23 silicon: without it every chip read K tiles [0, NB KB) against its own weight slice
template <uint32_t NB, uint32_t KB, uint32_t NR, uint32_t OUT_CB, typename XA>
FORCE_INLINE void fab_blocks(const XA& x, uint32_t n0, uint32_t koff = 0) {
    const uint32_t xpage = get_tile_size(CB_IN0);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        cb_reserve_back(CB_IN0, KB);
        const uint32_t l1 = get_write_ptr(CB_IN0);
        for (uint32_t k = 0; k < KB; ++k) {
            noc_async_read_page(koff + blk * KB + k, x, l1 + k * xpage);
        }
        noc_async_read_barrier();
        cb_push_back(CB_IN0, KB);
    }
    cb_wait_front(OUT_CB, NR);
    const uint32_t l1 = get_read_ptr(OUT_CB);
    const uint32_t opage = get_tile_size(OUT_CB);
    const uint32_t own = get_arg_val<uint32_t>(MK_FAB_RT + 4);
    for (uint32_t j = 0; j < NR; ++j) {
        const uint32_t t = n0 + j, lane = t / FAB_LANE_TILES;
        const uint32_t src = l1 + j * opage;
        const uint64_t dst = get_noc_addr(
            get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane),
            get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane + 1),
            own + (t % FAB_LANE_TILES) * FAB_V);
        if constexpr (MK_FAB_BROWS >= 32) {
            noc_async_write(src, dst, 2048);
        } else {
            noc_async_write(src, dst, FAB_CH);
            noc_async_write(src + 512, dst + FAB_CH, FAB_CH);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(OUT_CB, NR);
    mk::cb_rewind(OUT_CB);  // the next phase's NR output tiles start at the base again
}

FORCE_INLINE void fab_ready() {
    const uint32_t lsync = get_arg_val<uint32_t>(MK_FAB_RT + 5);  // OWN_READY = word 0 of the lane's sync words
    for (uint32_t lane = 0; lane < 2; ++lane) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane), get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane + 1), lsync),
            1);
    }
    noc_async_atomic_barrier();
}
#endif

template <uint32_t P>
FORCE_INLINE void move_phase() {
    constexpr uint32_t B = 4 + 9 * P;
    constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + B + 0);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + B + 1);
    constexpr uint32_t NR = get_compile_time_arg_val(PH_CT0 + B + 2);
    constexpr uint32_t WAIT_SEM = get_compile_time_arg_val(PH_CT0 + B + 3);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + B + 4);
    constexpr uint32_t SIGNAL_SEM = get_compile_time_arg_val(PH_CT0 + B + 5);
    constexpr uint32_t XOFF = get_compile_time_arg_val(PH_CT0 + B + 6);
    constexpr uint32_t YOFF = get_compile_time_arg_val(PH_CT0 + B + 7);
    constexpr auto x_args = TensorAccessorArgs<XOFF>();
    constexpr auto y_args = TensorAccessorArgs<YOFF>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(ph_rt0 + 3 * P + 0));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(ph_rt0 + 3 * P + 1));
#if defined(MK_KSPLIT)
    const uint32_t n0_arg = get_arg_val<uint32_t>(ph_rt0 + 3 * P + 2);  // (koff << 16) | n0 (Stage 14 K-split phases)
    const uint32_t n0 = n0_arg == 0xFFFFFFFFu ? n0_arg : (n0_arg & 0xFFFFu);
    const uint32_t koff = n0_arg == 0xFFFFFFFFu ? 0u : (n0_arg >> 16);
#else
    const uint32_t n0 = get_arg_val<uint32_t>(ph_rt0 + 3 * P + 2);
    constexpr uint32_t koff = 0;
#endif

    if constexpr (WAIT_COUNT > 0) {
        mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
    }
    MK_MARK();
#if defined(MK_FAB_MASK)
    if constexpr ((((MK_FAB_MASK) >> P) & 1) != 0) {  // the partial goes to the fabric lanes, which signal the
                                                      // consumers
        if (n0 != 0xFFFFFFFFu) {
            fab_blocks<NB, KB, NR, get_compile_time_arg_val(PH_CT0 + B + 8)>(x, n0, koff);
        }
        fab_ready();
        MK_MARK();
        return;
    }
#endif
#if defined(MK_SCALE_MASK)
    if constexpr (((MK_SCALE_FIRST) >> P) & 1) {  // the activation blocks go first, then the 1/rms(h) for the pack
        if (n0 != 0xFFFFFFFFu) {
            pull_blocks<NB, KB>(x, koff);
        }
        fetch_inv<P>();
        if (n0 != 0xFFFFFFFFu) {
            write_out<NR, get_compile_time_arg_val(PH_CT0 + B + 8)>(y, n0, P);
        }
    } else
#endif
        if (n0 != 0xFFFFFFFFu) {
        move_blocks<NB, KB, NR, get_compile_time_arg_val(PH_CT0 + B + 8)>(x, y, n0, P, koff);
    }
    if constexpr (NCONS > 0) {
        mk::signal_handoff(SIGNAL_SEM, ph_rt0 + 3 * NPH, NCONS);
    }
    MK_MARK();
}

template <uint32_t P>
FORCE_INLINE void run_a() {
#if defined(MK_SPLIT)
    if constexpr (P == MK_SPLIT) {
        MK_BETWEEN();
    }
#endif
    if constexpr (P < NPH) {
        move_phase<P>();
#if defined(MK_AFTER_PHASE)
        if constexpr (P == MK_AFTER_PHASE) {  // a composite's hook once this phase's outputs are written
            MK_AFTER_HOOK();
        }
#endif
        run_a<P + 1>();
    }
}

}  // namespace

void kernel_main() {
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {  // stack program (kernels/mk/stack.hpp)
        MK_LAYER_BEGIN(layer);
        MK_MARK();
        if (layer > 0) {
            mkstack::load_args(layer);
        }
#if defined(MK_TOPK_K)
        mktopk::init();
#endif
        run_a<0>();
#if defined(MK_TOPK_K)
        mktopk::send();
#endif
#if defined(MK_PROBE_ARG)
        MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
    }
}
