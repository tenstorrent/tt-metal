// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA item cores — reader (BRISC, NoC0): P4 conv + P5 prepare inputs of every (head, chunk) item of this core.
// Derived from ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/
// reader_prepare_chunk_recurrence.cpp (fill_constant_tiles: eye / tril / ones / block masks; the chronology's
// valid_chunks / partial rows; the per-item q k v g beta reads) and kernels/kda_pre/reader.cpp (row-0 tap / dt_bias /
// rate tiles). New: the conv shift matrices, the previous chunk's u tiles (the conv needs rows t-3..t-1), the conv
// carry rows (state C, kernels/prefill/state_io.cpp layout) as chunk 0's virtual previous tile, the beta column tile
// built from the low-rank b columns, the row mask (padded rows of the partial chunk become identity steps: zero k, v,
// g, beta -- the op zeroes those rows in the reader; here the compute multiplies by the mask after the conv).
//
// Core = (head, slot): items are chunks c = slot, slot + CPH, ... of head `head` (CPH item cores per head), in order,
// and their writer signals them in chunk order (the SEM_KDA baton, kda_item_w.cpp).
// Compile-time (PH_CT0 +): Kt, VTF (value tiles of a head), H, NCH, CPH, WT (q width tiles of the chip = H Kt),
// UNT (u scratch tiles per row), LRNT (low-rank tiles per row), BTILE (the b column tile of the low-rank output),
// GNT (f scratch tiles per row), NPROD (H3 producers per 32-row block).
// Runtime (ph_rt0 +): head, slot, u, lr, f, cst (conv carry: FusedKDAState.C [1, 3, 1, 3 W] bf16 TILE, row 0 of every
// tile = carry row j of tile j * 3 WT + column tile), valid_end, taps 0..3
// ([1, 1, 3 W] bf16 TILE each), dt_bias, decay rate ([1, 1, 1, W] fp32 TILE), probe (row = head * CPH + slot,
// words 16..).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"

// W2 (standalone KdaPhase, PMK_ITEM_ZONES_SA): per-item reader zones -- wu = waiting for CB U room (the compute is
// behind), rb = the item's DRAM read barrier, sh = the PMK_ITEM_CONV_FPU shift copies
#if defined(PMK_ITEM_ZONES_SA) && !defined(MK_PHASED)
#include "api/debug/kernel_profiler.hpp"
#define PMK_IZR(n) DeviceZoneScopedN("PMK_IR_" n)
#else
#define PMK_IZR(n)
#endif

using namespace pk;

namespace {

FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, MEM_ZEROS_SIZE < bytes - off ? MEM_ZEROS_SIZE : bytes - off);
    }
}

constexpr uint32_t ONE_F32 = 0x3F800000u;

// the constant fp32 tiles of CB_CONST (zeroed by the caller): eye, tril (incl. diagonal), ones, the three
// nested-inverse block masks (reader_prepare_chunk_recurrence.cpp), the conv shift matrices Sdn_s[t][t-s] (t >= s) and
// Sup_s[t][32+t-s] (t < s) for s = 1..3
FORCE_INLINE void fill_constants(uint32_t base) {
    auto tile = [&](uint32_t page) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + page * T32); };
    volatile tt_l1_ptr uint32_t* eye = tile(K_EYE);
    volatile tt_l1_ptr uint32_t* tril = tile(K_TRIL);
    volatile tt_l1_ptr uint32_t* ones = tile(K_ONES);
    for (uint32_t r = 0; r < CHUNK; ++r) {
        for (uint32_t c = 0; c < CHUNK; ++c) {
            const uint32_t i = tile_off(r, c);
            ones[i] = ONE_F32;
            if (c <= r) {
                tril[i] = ONE_F32;
            }
            if (c == r) {
                eye[i] = ONE_F32;
            }
            const bool same_block = r / 4 == c / 4, same_pair = r / 8 == c / 8;
            if (same_block) {
                tile(K_MASK0)[i] = ONE_F32;
            } else if (same_pair) {
                tile(K_MASK0 + 1)[i] = ONE_F32;
            } else if (r / 8 > c / 8) {
                tile(K_MASK0 + 2)[i] = ONE_F32;
            }
        }
    }
    for (uint32_t s = 1; s <= 3; ++s) {
        for (uint32_t t = 0; t < CHUNK; ++t) {
            if (t >= s) {
                tile(K_SDN1 + s - 1)[tile_off(t, t - s)] = ONE_F32;
            } else {
                tile(K_SUP1 + s - 1)[tile_off(t, CHUNK + t - s)] = ONE_F32;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t VTF = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t NCH = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t CPH = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t WT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t UNT = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t LRNT = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t BTILE = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t GNT = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t NPROD = get_compile_time_arg_val(PH_CT0 + 10);
    constexpr uint32_t CT = 2 * Kt + VTF;  // conv column tiles of a head (q | k | v)
#ifdef PMK_ITEM_CONV_FPU
    constexpr uint32_t UPI = 2 * CT;  // CB U pages per item: previous | this (S1..S3 in CB_SH, one item deep)
#else
    constexpr uint32_t UPI = 2 * CT;
#endif
    static_assert(Kt == VTF, "KDA head: key dim == value dim");

    const uint32_t head = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t slot = get_arg_val<uint32_t>(ph_rt0 + 1);
    const InterleavedAddrGen<true> u_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 2), .page_size = T16};
    const InterleavedAddrGen<true> lr_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 3), .page_size = T16};
    const InterleavedAddrGen<true> f_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 4), .page_size = T32};
    const InterleavedAddrGen<true> cst_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 5), .page_size = T16};
    const InterleavedAddrGen<true> valid_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 6), .page_size = 4};
    const uint32_t taps_addr[4] = {
        get_arg_val<uint32_t>(ph_rt0 + 7),
        get_arg_val<uint32_t>(ph_rt0 + 8),
        get_arg_val<uint32_t>(ph_rt0 + 9),
        get_arg_val<uint32_t>(ph_rt0 + 10)};
    auto taps_noc = [&](uint32_t j, uint32_t tile, uint32_t off) {
        return InterleavedAddrGen<true>{.bank_base_address = taps_addr[j], .page_size = T16}.get_noc_addr(tile, off);
    };
    const InterleavedAddrGen<true> dt_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 11), .page_size = T32};
    const InterleavedAddrGen<true> rate_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 12), .page_size = T32};
    const InterleavedAddrGen<true> probe_g{
        .bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 13), .page_size = PROBE_ROW_BYTES};
    uint32_t pr_items = 0, pr_chk_u = 0, pr_chk_f = 0;

    // valid_end (device tensor: the trace is request-agnostic) -> the chunks with valid rows
    cb_reserve_back(CB_MASK, 1);
    const uint32_t scratch = get_write_ptr(CB_MASK);
    noc_async_read(valid_g.get_noc_addr(0), scratch, 4);
    noc_async_read_barrier();
    const uint32_t ve = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];

    // constants (once per layer program)
    cb_reserve_back(CB_CONST, K_PAGES);
    const uint32_t kbase = get_write_ptr(CB_CONST);
    zero_l1(kbase, K_PAGES * T32);
    noc_async_read_barrier();
    fill_constants(kbase);
    cb_push_back(CB_CONST, K_PAGES);

    // head-resident row-0 tiles: taps (bf16: row 0 = bytes [0, 32) of face 0 and [512, 544) of face 1), dt_bias / rate
    // (fp32: [0, 64) and [1024, 1088))
    cb_reserve_back(CB_TAPS, 4 * CT);
    const uint32_t tbase = get_write_ptr(CB_TAPS);
    for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t t = 0; t < CT; ++t) {
            const uint32_t tile = (t / Kt) * WT + head * Kt + t % Kt;
            const uint32_t dst = tbase + (j * CT + t) * T16;
            noc_async_read(taps_noc(j, tile, 0), dst, 32);
            noc_async_read(taps_noc(j, tile, 512), dst + 512, 32);
        }
    }
    cb_reserve_back(CB_DTR, 2 * Kt);
    const uint32_t dbase = get_write_ptr(CB_DTR);
    for (uint32_t t = 0; t < Kt; ++t) {
        const uint32_t tile = head * Kt + t;
        noc_async_read(dt_g.get_noc_addr(tile, 0), dbase + t * T32, 64);
        noc_async_read(dt_g.get_noc_addr(tile, 1024), dbase + t * T32 + 1024, 64);
        noc_async_read(rate_g.get_noc_addr(tile, 0), dbase + (Kt + t) * T32, 64);
        noc_async_read(rate_g.get_noc_addr(tile, 1024), dbase + (Kt + t) * T32 + 1024, 64);
    }
    noc_async_read_barrier();
    cb_push_back(CB_TAPS, 4 * CT);
    cb_push_back(CB_DTR, 2 * Kt);

    const uint32_t hcol = head % CHUNK;  // this head's b column within the beta tile
    for (uint32_t c = slot; c < NCH; c += CPH) {
        const uint32_t rows = chunk_rows(ve, c);
        // control words first (the compute reads them before any other wait): live, partial rows
        cb_reserve_back(CB_BETA, 1);
        const uint32_t bt = get_write_ptr(CB_BETA);
        volatile tt_l1_ptr uint32_t* bw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bt);
        if (rows == 0) {  // a dead chunk: no inputs, no outputs (the scan never reads it)
            bw[CTL_COL] = 0;
            bw[CTL_COL + 1] = 0;
            cb_push_back(CB_BETA, 1);
            continue;
        }
        // H3: the P3 / P4 stream phases' 32-row blocks landed (per-block monotonic counter, >= never ==)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QKV)), (c + 1) * NPROD);
        // beta column tile: col 0 rows r = bf16 -> fp32 of lr[c * 32 + r, 2 d + head] (sigmoid in the compute); the
        // bf16 tile is staged in the MASK page, which is then rebuilt as the row mask
        const uint32_t mt = get_write_ptr(CB_MASK);
        zero_l1(bt, T32);
        noc_async_read(lr_g.get_noc_addr(c * LRNT + BTILE), mt, T16);
        noc_async_read_barrier();
        {
            volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mt);
            for (uint32_t r = 0; r < CHUNK; ++r) {
                bw[tile_off(r, 0)] = static_cast<uint32_t>(src[tile_off(r, hcol)]) << 16;
            }
            bw[CTL_COL] = 1;
            bw[CTL_COL + 1] = rows < CHUNK ? rows : 0;
        }
        zero_l1(mt, T32);
        noc_async_read_barrier();
        {
            volatile tt_l1_ptr uint32_t* mw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mt);
            for (uint32_t r = 0; r < rows; ++r) {
                mw[tile_off(r, 0)] = ONE_F32;
            }
        }
        cb_push_back(CB_BETA, 1);
        cb_push_back(CB_MASK, 1);
        cb_reserve_back(CB_MASK, 1);  // the next item's staging / mask page (and the loop's exit leaves one reserved)

        // u tiles: [previous chunk | this chunk], q k v column tiles of the head (PMK_ITEM_CONV_FPU: + the shifted
        // S1..S3)
        {
            PMK_IZR("wu");
            cb_reserve_back(CB_U, UPI);
        }
        const uint32_t ub = get_write_ptr(CB_U);
        if (c == 0) {  // virtual previous tile: rows 29..31 = the conv carry rows 0..2 (state C), the rest zero
            // NoC rule (Blackhole, watcher-checked): a DRAM transfer's source and L1 destination share their residue
            // mod 64; a tile row lands at +1440 / +1952 B (32 mod 64), so the carry tiles are read WHOLE into a
            // staging area (the F pages reserved below) and their row 0 copied into rows 29..31 in L1
            cb_reserve_back(CB_F, Kt);
            const uint32_t stg = get_write_ptr(CB_F);
            zero_l1(ub, CT * T16);
            noc_async_read_barrier();
            for (uint32_t t = 0; t < CT; ++t) {
                const uint32_t ct = (t / Kt) * WT + head * Kt + t % Kt;  // column tile of 3 W
                for (uint32_t i = 0; i < 3; ++i) {
                    noc_async_read(cst_g.get_noc_addr(i * 3 * WT + ct), stg + i * T16, T16);  // carry row i's tile
                }
                noc_async_read_barrier();
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ub + t * T16);
                for (uint32_t i = 0; i < 3; ++i) {
                    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg + i * T16);
                    const uint32_t r = CHUNK - 3 + i;
                    for (uint32_t hf = 0; hf < 2; ++hf) {  // the row's two 16-element face halves (8 words each)
                        const uint32_t so = tile_off(0, 16 * hf) / 2, dO = tile_off(r, 16 * hf) / 2;
                        for (uint32_t w = 0; w < 8; ++w) {
                            dst[dO + w] = src[so + w];
                        }
                    }
                }
            }
        } else {
            for (uint32_t t = 0; t < CT; ++t) {
                const uint32_t tile = (c - 1) * UNT + (t / Kt) * WT + head * Kt + t % Kt;
                noc_async_read(u_g.get_noc_addr(tile), ub + t * T16, T16);
            }
        }
        for (uint32_t t = 0; t < CT; ++t) {
            const uint32_t tile = c * UNT + (t / Kt) * WT + head * Kt + t % Kt;
            noc_async_read(u_g.get_noc_addr(tile), ub + (CT + t) * T16, T16);
        }
        // raw f_b output tiles of the chunk (the decay gate is computed in the compute)
        cb_reserve_back(CB_F, Kt);
        const uint32_t fb = get_write_ptr(CB_F);
        for (uint32_t t = 0; t < Kt; ++t) {
            noc_async_read(f_g.get_noc_addr(c * GNT + head * Kt + t), fb + t * T32, T32);
        }
        {
            PMK_IZR("rb");
            noc_async_read_barrier();
        }
#ifdef PMK_ITEM_CONV_FPU
        // S_s (s = 1..3) of column tile t at pages (1 + s) CT + t: row r = this chunk's row r - s (r >= s), else the
        // previous tile's row 32 + r - s (= the shift-matmul pair SDN(s) / SUP(s)). Face rows are contiguous (32 B), so
        // each 16-row face of S_s is two runs: 4 faces x 2 = 8 local NoC copies per shifted tile
        {
            PMK_IZR("sh");
            cb_reserve_back(CB_SH, 3 * CT);
            const uint32_t shb = get_write_ptr(CB_SH);
            const uint64_t me = get_noc_addr(0);
            constexpr uint32_t FACE = 512, ROWB = 32;
            for (uint32_t t = 0; t < CT; ++t) {
                const uint32_t pv = ub + t * T16, cu = ub + (CT + t) * T16;
                for (uint32_t s = 1; s <= 3; ++s) {
                    const uint32_t d = shb + ((s - 1) * CT + t) * T16;
                    for (uint32_t hf = 0; hf < 2; ++hf) {
                        const uint32_t dt = d + hf * FACE, db = d + (2 + hf) * FACE;
                        const uint32_t pb = pv + (2 + hf) * FACE, ct_ = cu + hf * FACE, cb_ = cu + (2 + hf) * FACE;
                        noc_async_read(
                            me + pb + (16 - s) * ROWB, dt, s * ROWB);  // rows 0..s-1 <- previous rows 32-s..31
                        noc_async_read(me + ct_, dt + s * ROWB, (16 - s) * ROWB);  // rows s..15 <- rows 0..15-s
                        noc_async_read(me + ct_ + (16 - s) * ROWB, db, s * ROWB);  // rows 16..15+s <- rows 16-s..15
                        noc_async_read(me + cb_, db + s * ROWB, (16 - s) * ROWB);  // rows 16+s..31 <- rows 16..31-s
                    }
                }
            }
            noc_async_read_barrier();
        }
#endif
        if (pr_items == 0) {  // probe: the first live item's first u tile (this chunk) and first f tile
            pr_chk_u = probe_sum(ub + CT * T16, T16 / 4);
            pr_chk_f = probe_sum(fb, T32 / 4);
        }
        ++pr_items;
        cb_push_back(CB_U, UPI);
#ifdef PMK_ITEM_CONV_FPU
        cb_push_back(CB_SH, 3 * CT);
#endif
        cb_push_back(CB_F, Kt);
    }
    {  // probe row (words 16..31): the MASK page reserved by the loop's exit is free scratch
        const uint32_t st = get_write_ptr(CB_MASK);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        w[0] = PROBE_ITEM_R;
        w[1] = ve;
        w[2] = NCH;
        w[3] = pr_items;
        w[4] = pr_chk_u;
        w[5] = pr_chk_f;
        w[6] = head;
        w[7] = slot;
        noc_async_write(st, probe_g.get_noc_addr(head * CPH + slot, 64), 64);
        noc_async_write_barrier();
    }
}
