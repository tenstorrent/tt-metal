// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear v3, writer (BRISC): streams the odd output-tile weight records into c_6 (tdec_stream.hpp),
// then writes this core's output tiles.
#include "tdec_stream.hpp"
#include "core_span.hpp"
#ifdef STEP_PREFETCH
#include "../../gdn_step/kernels/step_prefetch.hpp"
#endif
#ifdef KV_PREFETCH
#include "../../sdpa_dec/kernels/kv_prefetch.hpp"
#endif
#ifndef OUT_PER_NT
#define OUT_PER_NT 1
#endif
// SPLIT_NT > 0: output tile columns from SPLIT_NT on go to a second tensor (runtime arg 5), at column - SPLIT_NT.
#ifndef SPLIT_NT
#define SPLIT_NT 0
#endif

void kernel_main() {
#ifdef COMMON_ARGS
    // common args: w, out, Nt, ncores, grid_x, unit, out2
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    uint32_t first_nt, num_nt;
    core_span(
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        first_nt,
        num_nt);
#define OUT2_ADDR get_common_arg_val<uint32_t>(6)
#else
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_nt = get_arg_val<uint32_t>(2);
    const uint32_t num_nt = get_arg_val<uint32_t>(3);
#define OUT2_ADDR get_arg_val<uint32_t>(5)
#endif
    constexpr uint32_t ppr = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
#if SPLIT_NT > 0 || defined(QG_HT)
    constexpr auto out2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // No weight reads until this core's x' and scales have landed (NCRISC pushes c_2 after both): with every core's
    // stream in flight (~8 MB queued at the banks) the small reads waited 17 us behind them.
#ifdef STEP_PREFETCH
    // the next GDN step's inputs into its cores' L1 (gdn_step/kernels/step_prefetch.hpp): this RISC is otherwise idle
    // until the outputs. Common args from PF_ARG: pf, nv, group, qt, wab, consts, conv state, recurrent state;
    // compile-time accessor args from PF_CT.
    {
        constexpr auto pw = TensorAccessorArgs<PF_CT>();
        constexpr auto pc = TensorAccessorArgs<pw.next_compile_time_args_offset()>();
        constexpr auto pv = TensorAccessorArgs<pc.next_compile_time_args_offset()>();
        constexpr auto ps = TensorAccessorArgs<pv.next_compile_time_args_offset()>();
        const uint32_t nv = get_common_arg_val<uint32_t>(PF_ARG + 1);
        const uint32_t j = core_index(get_common_arg_val<uint32_t>(4));
        if (j < nv) {
            step_pf::prefetch(
                get_common_arg_val<uint32_t>(PF_ARG),
                j,
                nv,
                get_common_arg_val<uint32_t>(PF_ARG + 2),
                get_common_arg_val<uint32_t>(PF_ARG + 3),
                TensorAccessor(pw, get_common_arg_val<uint32_t>(PF_ARG + 4), 2048),
                TensorAccessor(pc, get_common_arg_val<uint32_t>(PF_ARG + 5), 2048),
                TensorAccessor(pv, get_common_arg_val<uint32_t>(PF_ARG + 6), 2048),
                TensorAccessor(ps, get_common_arg_val<uint32_t>(PF_ARG + 7), 2048));
            noc_async_read_barrier();
        }
    }
#endif
#ifdef KV_PREFETCH
    // the decode SDPA's K / V into its cores' L1 (sdpa_dec/kernels/kv_prefetch.hpp). Common args from KVP_ARG: record,
    // buffer, K cache, V cache, bs, NG, H, maxr; compile-time accessor args from KVP_CT.
    {
        constexpr auto ka = TensorAccessorArgs<KVP_CT>();
        constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
        kv_pf::prefetch(
            core_index(get_common_arg_val<uint32_t>(4)),
            get_common_arg_val<uint32_t>(KVP_ARG),
            get_common_arg_val<uint32_t>(KVP_ARG + 1),
            TensorAccessor(ka, get_common_arg_val<uint32_t>(KVP_ARG + 2), kv_pf::KB),
            TensorAccessor(va, get_common_arg_val<uint32_t>(KVP_ARG + 3), kv_pf::KB),
            get_common_arg_val<uint32_t>(KVP_ARG + 4),
            get_common_arg_val<uint32_t>(KVP_ARG + 5),
            get_common_arg_val<uint32_t>(KVP_ARG + 6),
            get_common_arg_val<uint32_t>(KVP_ARG + 7));
        noc_async_read_barrier();
    }
#endif
#if defined(GCB_STREAM) || defined(W_STREAM)
    // the weights arrive another way (prefetcher / weight-stream ring, read by NCRISC); this RISC only writes
#elif defined(X_PAGED) || defined(X_HUB)
    // x' comes from L1 (no DRAM queue to protect): stream at once
#elif defined(INLINE_SCALES)
    cb_wait_front(tt::CBIndex::c_0, 1);  // x' landed (NCRISC pushes c_0 first)
#else
    cb_wait_front(tt::CBIndex::c_2, num_nt * S_PER_NT);
#endif
#ifdef W_STAGED
    const StagedW w{get_common_arg_val<uint32_t>(STAGE_ARG_WR), first_nt * ppr};
    (void)w_addr;
#else
    const auto w = TensorAccessor(w_args, w_addr, 4096);
#endif
#if defined(GCB_STREAM) || defined(W_STREAM)
    (void)w;  // the weights come from elsewhere (W_STREAM: the reader takes every record from the ring)
#elif defined(PAGE_SPLIT)
    stream_pages<ppr, NB>(tt::CBIndex::c_6, w, first_nt, num_nt, 1);
#else
    stream_records<ppr, NB>(tt::CBIndex::c_6, w, first_nt, num_nt, 1);
#endif

    const uint32_t bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, bytes);
#if SPLIT_NT > 0 || defined(QG_HT)
    const auto out2 = TensorAccessor(out2_args, OUT2_ADDR, bytes);
#endif
#ifdef GLU_PAIRS
    // One output tile per (gate, up) record pair.
    for (uint32_t i = 0; i < num_nt / 2; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t nt = first_nt / 2 + i;
#else
    for (uint32_t i = 0; i < num_nt * OUT_PER_NT; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t nt = first_nt + i;
#endif
#if defined(QG_HT)
        // per head [q (QG_HT tiles) | gate (QG_HT tiles)] -> q head-major [1, heads, 1, D] and gate [1, 1, heads D]:
        // one tile row each (decode), so both are page remaps
        const uint32_t hq = nt / (2 * QG_HT), cq = nt % QG_HT;
        const uint64_t dst =
            ((nt / QG_HT) & 1) ? out2.get_noc_addr(hq * QG_HT + cq) : out.get_noc_addr(hq * QG_HT + cq);
#elif SPLIT_NT > 0
        const uint64_t dst = nt < SPLIT_NT ? out.get_noc_addr(nt) : out2.get_noc_addr(nt - SPLIT_NT);
#else
        const uint64_t dst = out.get_noc_addr(nt);
#endif
        noc_async_write(get_read_ptr(cb_out), dst, bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
