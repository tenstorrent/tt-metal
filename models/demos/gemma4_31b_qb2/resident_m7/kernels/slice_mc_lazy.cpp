// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident slice, second h multicaster + lazy tap rows (NCRISC, NOC 1; M6b lazy lever 2; slice_mc.cpp is the
// verified version, lazy_levers.enable_v3 selects this one; slice_dm.cpp stays the BRISC kernel, lazy off): after the
// h of an even round r >= 2 whose layer r / 2 - 1 is tapped is out (done reported, so the BRISC goes on), this RISC
// writes the lazy rows of that layer straight from the x tiles (x of an even round sits at cb_x's base until round
// r + 2 rewrites it): per live drafting row (position >= 0, slot < lazy slots, mask row nonzero) and 32-column chunk of
// the chip's quarter that this slice holds, one 32 B unicast DRAM write from each face row (no staging, no RISC
// copies), then a flush (the writes have left L1) before the next h. NoC (K2a N1-N4): unicast writes into DRAM on
// NOC 1 and three 128 B reads at the start; nothing on multicast destinations. The positions / slots / mask rows are
// read once into a 512 B L1 scratch (a host L1 tensor).
// Extra runtime args after the rectangles: lazy ring (0 = off), positions, slots, mask (0 = every live row), lazy
// slots, scratch, chip, K, T, R, M, tap mask lo, tap mask hi.
//
// slice_mc.cpp, second h multicaster (NCRISC, NOC 1; M2b F2). slice_dm (BRISC, NOC 0) multicasts each round's
// h tiles into the first rectangles of the streamer cover; this kernel takes the remaining rectangles, so the two
// NoCs carry the round's h at once. Per h (seed, then rounds 1 .. rounds): wait for the BRISC's go count (it has
// cb_hout's front: the compute's h of this round, at cb_hout's base, published once in sem_src), multicast the slice's
// h into this kernel's rectangles, then +1 on their sem_slots, then report done. The BRISC pops cb_hout only after
// both halves are out, so cb_hout keeps one consumer (the BRISC). This RISC issues NoC transactions on NOC 1 only.
//
// (M2b F10) h comes chunk by chunk: the go count is the BRISC's chunk sequence (r kChunks + c + 1), and this kernel
// multicasts each chunk's real bytes as soon as it is out; the slot semaphores still move once per h.
//
// Compile-time args: 0 rounds, 1 slice_bytes, 2 k, 3 sem_go, 4 sem_done, 5 sem_src, 6 sem_haddr, 7 sem_slots, 8 V,
// 9 kChunk (views per chunk)
// Runtime args: 0 rectangle count, then per rectangle physical x0, y0, x1, y1, cores

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t rounds = get_compile_time_arg_val(0);
constexpr uint32_t slice_bytes = get_compile_time_arg_val(1);
constexpr uint32_t k = get_compile_time_arg_val(2);
constexpr uint32_t sem_go = get_compile_time_arg_val(3);
constexpr uint32_t sem_done = get_compile_time_arg_val(4);
constexpr uint32_t sem_src = get_compile_time_arg_val(5);
constexpr uint32_t sem_haddr = get_compile_time_arg_val(6);
constexpr uint32_t sem_slots = get_compile_time_arg_val(7);
constexpr uint32_t V = get_compile_time_arg_val(8);
constexpr uint32_t kChunk = get_compile_time_arg_val(9);
constexpr uint32_t kChunks = (V + kChunk - 1) / kChunk;
constexpr uint32_t kView = 2048;

void kernel_main() {
    const uint32_t rects = get_arg_val<uint32_t>(0);
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(1 + 5 * q + i); };
    size_t a = 1 + 5 * rects;
    const uint32_t lazy_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slots_addr = get_arg_val<uint32_t>(a++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lazy_slots = get_arg_val<uint32_t>(a++);
    const uint32_t scratch = get_arg_val<uint32_t>(a++);
    const uint32_t chip = get_arg_val<uint32_t>(a++);
    const uint32_t K = get_arg_val<uint32_t>(a++);
    const uint32_t T = get_arg_val<uint32_t>(a++);
    const uint32_t R = get_arg_val<uint32_t>(a++);
    const uint32_t M = get_arg_val<uint32_t>(a++);
    const uint32_t tap_lo = get_arg_val<uint32_t>(a++), tap_hi = get_arg_val<uint32_t>(a++);
    auto tapped = [&](uint32_t l) { return ((l < 32 ? tap_lo >> l : tap_hi >> (l - 32)) & 1) != 0; };
    constexpr uint32_t kHt = 168, kQuarterBytes = 2688;
    const uint32_t quarter = kHt / 4;  // 42 tiles per chip
    const uint32_t lz0 = chip * quarter > k * T ? chip * quarter : k * T;
    const uint32_t lz1 = (chip + 1) * quarter < (k + 1) * T ? (chip + 1) * quarter : (k + 1) * T;
    const uint32_t tile_bytes = 64 * R;
    const uint32_t x_addr = [] {
        auto& iface = get_local_cb_interface(2);  // cb_x
        return iface.fifo_limit - iface.fifo_size;
    }();
    uint32_t lrow[32];
    for (uint32_t r = 0; r < 32; r++) {
        lrow[r] = 0xFFFFFFFF;
    }
    uint32_t n_live = 0, lazy_index = 0;
    if (lazy_addr && lz1 > lz0) {
        const InterleavedAddrGen<true> pg{.bank_base_address = pos_addr, .page_size = 128};
        const InterleavedAddrGen<true> sg{.bank_base_address = slots_addr, .page_size = 128};
        const InterleavedAddrGen<true> mg{.bank_base_address = mask_addr, .page_size = 128};
        noc_async_read(pg.get_noc_addr(0), scratch, 128);
        noc_async_read(sg.get_noc_addr(0), scratch + 128, 128);
        if (mask_addr) {
            noc_async_read(mg.get_noc_addr(0), scratch + 256, 128);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scratch);
        volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128);
        volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 256);
        for (uint32_t r = 0; r < M; r++) {
            const bool live = pos[r] >= 0 && slot[r] < lazy_slots && (!mask_addr || mask[r] != 0);
            lrow[r] = live ? slot[r] * 5 * 32 + static_cast<uint32_t>(pos[r]) % 32 : 0xFFFFFFFF;
            n_live += live ? 1 : 0;
        }
    }
    const InterleavedAddrGen<true> lazyg{.bank_base_address = lazy_addr, .page_size = kQuarterBytes};
    auto lazy_write = [&](uint32_t tap) {
        for (uint32_t r = 0; r < M; r++) {
            if (lrow[r] == 0xFFFFFFFF) {
                continue;
            }
            const uint64_t row = lazyg.get_noc_addr(lrow[r] + 32 * tap, (lz0 - chip * quarter) * 64);
            for (uint32_t t = lz0; t < lz1; t++) {
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t off = R == 32 ? (r / 16) * 1024 + f * 512 + (r % 16) * 32 : f * 32 * R + r * 32;
                    noc_async_write(x_addr + (t - k * T) * tile_bytes + off, row + (t - lz0) * 64 + f * 32, 32);
                }
            }
        }
        noc_async_writes_flushed();  // x is rewritten two rounds on
    };
    // NOC 1 multicasts run high -> low on Blackhole: the rectangle's end corner goes first (as tt-metal's NOC_1
    // multicast senders swap start and end)
    auto mc = [&](uint32_t q, uint32_t addr) { return get_noc_multicast_addr(rect(q, 2), rect(q, 3), rect(q, 0), rect(q, 1), addr); };
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done));
    volatile tt_l1_ptr uint32_t* srcw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_src));
    volatile tt_l1_ptr uint32_t* haddr_box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_haddr));
    uint32_t h_dst = 0;
    for (uint32_t r = 0; r <= rounds; r++) {
        uint32_t src = 0, dst = 0;
        for (uint32_t c = 0; c < kChunks; c++) {
            do {
                invalidate_l1_cache();
            } while (*go < r * kChunks + c + 1);
            if (c == 0) {
                while (h_dst == 0) {
                    invalidate_l1_cache();
                    h_dst = *haddr_box;
                }
                src = *srcw;
                dst = h_dst + k * slice_bytes;
            }
            const uint32_t v1 = (c + 1) * kChunk < V ? (c + 1) * kChunk : V;
            const uint32_t b0 = c * kChunk * kView, b1 = v1 * kView < slice_bytes ? v1 * kView : slice_bytes;
            if (b1 > b0) {
                for (uint32_t q = 0; q < rects; q++) {
                    noc_async_write_multicast(src + b0, mc(q, dst + b0), b1 - b0, rect(q, 4));
                }
            }
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_inc_multicast(mc(q, get_semaphore(sem_slots)), 1, rect(q, 4));
        }
        noc_async_atomic_barrier();
        *done = r + 1;
        if (lazy_addr && r >= 2 && r % 2 == 0 && tapped(r / 2 - 1)) {
            const uint32_t tap = lazy_index++;
            if (n_live && lz1 > lz0) {
                lazy_write(tap);
            }
        }
    }
    noc_async_write_barrier();
}
