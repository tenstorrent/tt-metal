// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the decode step's flash-decode SDPA (tt/v41/flash_sdpa.py; fsdpa_common.hpp has the layout). At start
// the persistent compute tiles (the reduce scaler, the column identity, the -inf row, the per-head sink column: as
// sparse_sdpa's writer builds them) and the grid's NoC table. From the reader's plan: this core's share of the dead
// rows' output pages zeroed (the rows that are not live get a defined output). Per unit of this core: the compute's
// max, sum and column-v acc tiles written into each of the row's VT merge cores' partial CBs (page slot * UMAX + j
// of their cb_pm / cb_ps / cb_pa; with DEBUG_SCRATCH also the whole partial to slot (row rank, j) of the DRAM
// scratch), a write barrier, then one semaphore increment on each merge core. Per merge unit of this core: the
// untilized [32 heads x 32 columns] block to the row-major output, one 64-byte write a head.
//
// compile_time_args = fsdpa::ct, then the accessors of scratch (any DRAM tensor when DEBUG_SCRATCH is 0), out,
// sink, noc table
// common_runtime_args = [scratch_addr, out_addr, sink_addr, noc_table_addr]; runtime_args = [core]

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
#include "models/demos/deepseek_v3_d_p/tt/v41/kernels/fsdpa_common.hpp"

using namespace fsdpa;

constexpr uint32_t one_bf16_packed = 0x3F803F80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ct::H);
    constexpr uint32_t S = get_compile_time_arg_val(ct::S);
    constexpr uint32_t V = get_compile_time_arg_val(ct::V);
    constexpr uint32_t UMAX = get_compile_time_arg_val(ct::UMAX);
    constexpr uint32_t N = get_compile_time_arg_val(ct::N);
    constexpr uint32_t VT = get_compile_time_arg_val(ct::VT);
    constexpr uint32_t SLOT_TILES = get_compile_time_arg_val(ct::SLOT_TILES);
    constexpr bool debug_scratch = get_compile_time_arg_val(ct::DEBUG_SCRATCH) != 0;
    constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
    constexpr uint32_t out_row_bytes = V * sizeof(uint16_t);
    constexpr uint32_t noc_table_bytes = (N * 8 + 63) / 64 * 64;

    constexpr auto part_args = TensorAccessorArgs<ct::END, 0>();
    constexpr auto out_args = TensorAccessorArgs<part_args.next_compile_time_args_offset(), 0>();
    constexpr auto sink_args = TensorAccessorArgs<out_args.next_compile_time_args_offset(), 0>();
    constexpr auto tab_args = TensorAccessorArgs<sink_args.next_compile_time_args_offset(), 0>();
    const uint32_t part_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sink_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tab_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(0);
    const auto part = TensorAccessor(part_args, part_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto sink = TensorAccessor(sink_args, sink_addr);
    const auto tab = TensorAccessor(tab_args, tab_addr);

    Noc noc;
    const UnicastEndpoint local_l1;
    experimental::CB max_cb(cb_max_w), sum_cb(cb_sum), out_cb(cb_out), o_rm_cb(cb_o_rm), ctrl_w(cb_ctrl_w);
    experimental::CB scratch_cb(cb_wscratch), sink_cb(cb_sink);

    // scratch: the NoC table, the sink row (64 B), a zero output row
    scratch_cb.reserve_back(1);
    const uint32_t scratch = scratch_cb.get_write_ptr();
    const uint32_t tab_l1 = scratch;
    const uint32_t sink_l1 = scratch + noc_table_bytes;
    const uint32_t zero_l1 = sink_l1 + 64;
    volatile tt_l1_ptr uint32_t* noc_xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_l1);
    noc.async_read(tab, local_l1, N * 8, {.page_id = 0}, {.addr = tab_l1});
    noc.async_read(sink, local_l1, H * sizeof(uint16_t), {.page_id = 0}, {.addr = sink_l1});
    noc.async_write_zeros(scratch_cb, out_row_bytes, {.offset_bytes = zero_l1 - scratch});

    // the persistent compute tiles (sparse_sdpa's writer)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = NEG_INF_BF16;
            p[tt::constants::FACE_HW + c] = NEG_INF_BF16;
        }
        neginf_cb.push_back(1);
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
    {
        // the sink column: row h of the tile = head h's sink
        sink_cb.reserve_back(1);
        noc.async_write_zeros(sink_cb, tile_bytes);
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
        volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_l1);
        for (uint32_t h = 0; h < H; ++h) {
            dst[sparse_sdpa::tile_col0_offset(h)] = src[h];
        }
        sink_cb.push_back(1);
    }

    // the plan
    ctrl_w.wait_front(1);
    uint32_t n_live = 0;
    uint32_t live[32];
    bool is_live[32];
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_w.get_read_ptr());
        n_live = p[PLAN_N_LIVE];
        for (uint32_t s = 0; s < S; ++s) {
            is_live[s] = false;
        }
        for (uint32_t i = 0; i < n_live; ++i) {
            live[i] = p[PLAN_LIVE + i];
            is_live[live[i]] = true;
        }
    }
    ctrl_w.pop_front(1);

    // the dead rows' output pages (h * S + r), this core's share, zeroed
    {
        uint32_t p = 0;
        for (uint32_t r = 0; r < S; ++r) {
            if (is_live[r]) {
                continue;
            }
            for (uint32_t h = 0; h < H; ++h, ++p) {
                if (p % N == core) {
                    noc.async_write(local_l1, out, out_row_bytes, {.addr = zero_l1}, {.page_id = h * S + r});
                }
            }
        }
    }

    // ---- the units ----
    // The merge cores' partial CBs start at the same L1 address on every core (one program, one CB layout) and
    // this RISC never pushes to its own, so its write pointers are the CBs' bases: page (slot * UMAX + j) of the
    // merge core's cb_pm / cb_ps / cb_pa is where unit j of its slot-th merge lands.
    const uint32_t pm_base = experimental::CB(cb_pm).get_write_ptr();
    const uint32_t ps_base = experimental::CB(cb_ps).get_write_ptr();
    const uint32_t pa_base = experimental::CB(cb_pa).get_write_ptr();
    const UnicastEndpoint remote_l1;
    const uint32_t total_units = n_live * UMAX;
    for (uint32_t u = core; u < total_units; u += N) {
        const uint32_t i = u / UMAX, j = u % UMAX;
        ctrl_w.wait_front(1);
        const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_w.get_read_ptr())[0];
        ctrl_w.pop_front(1);
        if (valid == 0) {
            continue;
        }
        max_cb.wait_front(1);
        sum_cb.wait_front(1);
        out_cb.wait_front(VT);
        for (uint32_t v = 0; v < VT; ++v) {
            const uint32_t mu = i * VT + v;
            const uint32_t mc = merge_core(mu, N);
            const uint32_t x = noc_xy[2 * mc], y = noc_xy[2 * mc + 1];
            const uint32_t off = (merge_sem(mu, N) * UMAX + j) * tile_bytes;
            noc.async_write(
                max_cb, remote_l1, tile_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = pm_base + off});
            noc.async_write(
                sum_cb, remote_l1, tile_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = ps_base + off});
            noc.async_write(
                out_cb,
                remote_l1,
                tile_bytes,
                {.offset_bytes = v * tile_bytes},
                {.noc_x = x, .noc_y = y, .addr = pa_base + off});
        }
        if constexpr (debug_scratch) {
            const uint32_t base = (i * UMAX + j) * SLOT_TILES;
            noc.async_write(max_cb, part, tile_bytes, {.offset_bytes = 0}, {.page_id = base});
            noc.async_write(sum_cb, part, tile_bytes, {.offset_bytes = 0}, {.page_id = base + 1});
            for (uint32_t v = 0; v < VT; ++v) {
                noc.async_write(out_cb, part, tile_bytes, {.offset_bytes = v * tile_bytes}, {.page_id = base + 2 + v});
            }
        }
        noc.async_write_barrier();  // the partials are in the merge cores' L1 before their semaphores move
        max_cb.pop_front(1);
        sum_cb.pop_front(1);
        out_cb.pop_front(VT);
        for (uint32_t v = 0; v < VT; ++v) {
            const uint32_t mu = i * VT + v;
            const uint32_t mc = merge_core(mu, N);
            const uint64_t sem_addr = get_noc_addr(noc_xy[2 * mc], noc_xy[2 * mc + 1], get_semaphore(merge_sem(mu, N)));
            noc_semaphore_inc(sem_addr, 1);
        }
    }

    // ---- the merge units ----
    const uint32_t total_merges = n_live * VT;
    for (uint32_t mu = N - 1 - core; mu < total_merges; mu += N) {
        const uint32_t i = mu / VT, v = mu % VT, r = live[i];
        ctrl_w.wait_front(1);
        ctrl_w.pop_front(1);
        o_rm_cb.wait_front(1);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(o_rm_cb, out, 64, {.offset_bytes = h * 64}, {.page_id = h * S + r, .offset_bytes = v * 64});
        }
        noc.async_write_barrier();
        o_rm_cb.pop_front(1);
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
