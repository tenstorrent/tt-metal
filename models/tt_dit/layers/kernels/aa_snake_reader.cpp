// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused anti-aliased SnakeBeta activation (BigVGAN UpSample1d(2x) -> snake -> DownSample1d(2x)), reader.
//
// Row-major fp32 sticks of C channels; a "tile" here is a 4 KB block of R = 1024 / C consecutive sticks, fed
// to the unpacker as one fp32 32x32 tile. Unpack -> DST -> pack is a byte identity, and every operand shares the
// layout, so the elementwise math is layout-blind. This RISC stages the sticks this core needs in L1 (whole
// DRAM pages, then the replicate clamp of the sequence ends as local copies) and gathers the 7 tap-shifted
// up-stage tiles per block into CB_UP for the compute kernel.
//
// Math (see layers/audio_aa_snake.py): base[r] = x[clamp(r - 3)]; E[q] = sum_j s[2j] base[q + j],
// O[q] = sum_j s[2j + 1] base[q + 1 + j] (j = 0..5); out[n] = sum_k t[k] z[clamp(2n + k - 5)], z[2q] = E[q],
// z[2q + 1] = O[q]. Up blocks cover q in [o0 - 3, o0 - 3 + nblocks * R), so tap t of block q0 is x sticks
// [q0 + t - 3, q0 + t - 3 + R).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ab = get_compile_time_arg_val(2);
    constexpr uint32_t cb_fl = get_compile_time_arg_val(7);
    constexpr uint32_t C = get_compile_time_arg_val(8);
    constexpr uint32_t R = get_compile_time_arg_val(9);
    constexpr uint32_t K = get_compile_time_arg_val(10);  // sticks per DRAM page (the time-pack factor)
    constexpr int32_t T_LOCAL = get_compile_time_arg_val(11);
    constexpr int32_t HALO = get_compile_time_arg_val(12);
    constexpr uint32_t X_PAGES = get_compile_time_arg_val(13);  // pages per batch item of the halo'd input
    constexpr uint32_t NB_EXTRA = get_compile_time_arg_val(17);
    constexpr uint32_t STICK = C * 4;
    constexpr uint32_t PAGE = K * STICK;
    constexpr uint32_t TILE = R * STICK;  // 4096

    constexpr auto x_args = TensorAccessorArgs<18>();
    constexpr auto ab_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto fl_args = TensorAccessorArgs<ab_args.next_compile_time_args_offset()>();

    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t o0 = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    if (n_tiles == 0) {
        return;
    }
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ab_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t fl_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, PAGE);
    const auto ab_acc = TensorAccessor(ab_args, ab_addr, TILE);
    const auto fl_acc = TensorAccessor(fl_args, fl_addr, 64);
    experimental::CB x_cb(cb_x);
    experimental::CB up_cb(cb_up);
    experimental::CB ab_cb(cb_ab);
    experimental::CB fl_cb(cb_fl);

    // Edge flags for this device: [is_first, is_last] along the time axis; the writer's z clamp reads them, and the
    // sequence ends below clamp per stick (the halo's replicated row repeats k sticks when rows are packed).
    fl_cb.reserve_back(1);
    const uint32_t fl_l1 = fl_cb.get_write_ptr();
    noc.async_read(fl_acc, fl_cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    // Alpha and beta blocks (the per-channel vectors repeated R times), one tile each.
    ab_cb.reserve_back(2);
    noc.async_read(ab_acc, ab_cb, TILE, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(ab_acc, ab_cb, TILE, {.page_id = 1, .offset_bytes = 0}, {.offset_bytes = TILE});
    noc.async_read_barrier();
    fl_cb.push_back(1);
    ab_cb.push_back(2);

    // Sticks this core's up blocks read, in unpadded local coordinates: q range +-3.
    const int32_t q_lo = static_cast<int32_t>(o0) - 3;
    const uint32_t nblocks = n_tiles + NB_EXTRA;
    const int32_t x_lo = q_lo - 3;
    const int32_t x_hi = q_lo + static_cast<int32_t>(nblocks * R) + 3;
    // The halo'd tensor holds unpadded sticks [-HALO, T_LOCAL + HALO). On a sequence end the halo is ignored and the
    // edge stick is replicated per stick below, like the unpacked reference; elsewhere the neighbour's sticks are real.
    volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl_l1);
    const int32_t avail_lo = fl[0] != 0 ? 0 : -HALO;
    const int32_t avail_hi = fl[1] != 0 ? T_LOCAL : T_LOCAL + HALO;
    const int32_t r_lo = x_lo < avail_lo ? avail_lo : x_lo;
    const int32_t r_hi = x_hi > avail_hi ? avail_hi : x_hi;

    // Staging: stick r lives at stage + (r - x_lo) * STICK; one page of slack on each side because the
    // page reads land whole (DRAM reads need 64 B alignment on both ends, which whole pages at page-aligned
    // offsets satisfy; single sticks would not).
    x_cb.reserve_back(1);
    const uint32_t stage = x_cb.get_write_ptr() + PAGE;
    const uint32_t p_lo = static_cast<uint32_t>(r_lo + HALO) / K;
    const uint32_t p_hi = (static_cast<uint32_t>(r_hi + HALO) + K - 1) / K;
    for (uint32_t p = p_lo; p < p_hi; ++p) {
        const int32_t first = static_cast<int32_t>(p * K) - HALO;  // unpadded index of the page's first stick
        const uint32_t dst = static_cast<uint32_t>(static_cast<int32_t>(PAGE) + (first - x_lo) * static_cast<int32_t>(STICK));
        noc.async_read(x_acc, x_cb, PAGE, {.page_id = b * X_PAGES + p, .offset_bytes = 0}, {.offset_bytes = dst});
    }
    noc.async_read_barrier();
    if (r_lo > x_lo || r_hi < x_hi) {
        experimental::set_read_state<STICK>(noc, stage);
        for (int32_t r = x_lo; r < r_lo; ++r) {
            experimental::read_with_state(
                noc, x_cb, stage + static_cast<uint32_t>(r_lo - x_lo) * STICK, {.offset_bytes = PAGE + (r - x_lo) * STICK});
        }
        for (int32_t r = r_hi; r < x_hi; ++r) {
            experimental::read_with_state(
                noc,
                x_cb,
                stage + static_cast<uint32_t>(r_hi - 1 - x_lo) * STICK,
                {.offset_bytes = PAGE + (r - x_lo) * STICK});
        }
        noc.async_read_barrier();
    }

    // Up-stage gathers: 7 tap-shifted 4 KB tiles per block, local L1 -> CB_UP.
    experimental::set_read_state<TILE>(noc, stage);
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
        const int32_t q0 = q_lo + static_cast<int32_t>(blk * R);
        up_cb.reserve_back(7);
        for (uint32_t t = 0; t < 7; ++t) {
            const uint32_t src_stick = static_cast<uint32_t>((q0 + static_cast<int32_t>(t) - 3) - x_lo);
            experimental::read_with_state(noc, up_cb, stage + src_stick * STICK, {.offset_bytes = t * TILE});
        }
        noc.async_read_barrier();
        up_cb.push_back(7);
    }
}
