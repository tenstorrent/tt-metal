// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA scan cores — writer (NCRISC, NoC1): per valid chunk the gated-norm partial-sum exchange with the head's
// other SPREAD - 1 cores (kernels/kda_rec/writer.cpp's REC_SPREAD protocol: my PSUM tile into slot (c % 2, my rank
// among the peer's others) of every peer's CB_PIN, +1 on their SEM_RMS; my PIN pages are reserved here and pushed
// to the compute once SEM_RMS >= (SPREAD - 1)(c + 1)), the `on` tiles to the H4 tensor (page c WT + head VTF + vblk Vt
// + v), the chunk-done aggregation (+1 on the leader's SEM_ONAGG; the leader raises SEM_ON on every P7 consumer once
// all NSCAN scan cores finished chunk c). After the chunks: the final state tiles to the state scratch (P9, the
// kda.write_slot / StateIO layout: page h Kt VTF + k VTF + v), the dead chunks' `on` tiles zeroed, and on the carry
// core the conv carry rows (state C = FusedKDAState.C [1, 3, 1, 3 W] bf16 TILE: carry row j = row 0 of the tiles
// j * 3 WT + t) = rows valid_end - 3 .. valid_end - 1 of u (rows before 0 keep the previous carry's rows:
// ChronologicalSelections.select_local_final_history).
// Derived from writer_recurrent_chunk_scan.cpp (write_recurrent / final state), kda_rec/writer.cpp (exchange) and
// kernels/prefill/state_io.cpp (state layouts, zero reads).
// Compile-time (PH_CT0 +): Kt, Vt, VTF, NCH, SPREAD, WT, NSCAN, NCONS, UNT (u tiles per row), IN_BYTES (the IN ring's
// bytes: the carry's whole-tile staging after the chunks).
// Runtime (ph_rt0 +): head, vblk, on, s, valid_end, cst, u, is_leader, leader x, leader y, carry_core, probe,
// N_ITEM, then (x, y) of the SPREAD - 1 peers in v-block order, then (x, y) of the NCONS consumers.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"

using namespace pk;

namespace {
FORCE_INLINE void zero_dram(const InterleavedAddrGen<true>& g, uint32_t page, uint32_t bytes) {
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_write(
            MEM_ZEROS_BASE, g.get_noc_addr(page, off), MEM_ZEROS_SIZE < bytes - off ? MEM_ZEROS_SIZE : bytes - off);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t VTF = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t NCH = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t SPREAD = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t WT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t NSCAN = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t UNT = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t IN_BYTES = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t A_PEERS = ph_rt0 + 13, A_CONS = A_PEERS + 2 * (SPREAD - 1);

    const uint32_t head = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t vblk = get_arg_val<uint32_t>(ph_rt0 + 1);
    const InterleavedAddrGen<true> on_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 2), .page_size = T16};
    const InterleavedAddrGen<true> s_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 3), .page_size = T32};
    const InterleavedAddrGen<true> valid_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 4), .page_size = 4};
    const InterleavedAddrGen<true> cst_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 5), .page_size = T16};
    const InterleavedAddrGen<true> u_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 6), .page_size = T16};
    // the leader / carry core is (head 0, v-block 0): derived from the (proven) args 0 / 1, not from the flags at
    // 7 / 10 (r3-probe: carry rows 0 on every core; the flags are dumped into the probe row for the post-mortem)
    const bool leader = head == 0 && vblk == 0;
    const uint32_t lx = get_arg_val<uint32_t>(ph_rt0 + 8), ly = get_arg_val<uint32_t>(ph_rt0 + 9);
    const bool carry_core = leader;
    const InterleavedAddrGen<true> probe_g{
        .bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 11), .page_size = PROBE_ROW_BYTES};
    const uint32_t n_item = get_arg_val<uint32_t>(ph_rt0 + 12);
    uint32_t pr_chunks = 0, pr_chk_on = 0, pr_chk_fs = 0, pr_carry = 0;

    // valid_end staged in the (still unused) final-state CB
    const uint32_t scratch = get_write_ptr(CB_FS);
    noc_async_read(valid_g.get_noc_addr(0), scratch, 4);
    noc_async_read_barrier();
    const uint32_t ve = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];
    uint32_t nvalid = valid_chunks(ve);
    if (nvalid > NCH) {
        nvalid = NCH;
    }
    volatile tt_l1_ptr uint32_t* sem_rms = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RMS));
    volatile tt_l1_ptr uint32_t* sem_agg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ONAGG));
    const uint64_t leader_agg = get_noc_addr(lx, ly, get_semaphore(SEM_ONAGG));

    auto chunk_done = [&](uint32_t c) {
        noc_async_write_barrier();
        noc_semaphore_inc(leader_agg, 1);
        if (leader) {
            noc_semaphore_wait_min(sem_agg, NSCAN * (c + 1));
            for (uint32_t i = 0; i < NCONS; ++i) {
                const uint32_t x = get_arg_val<uint32_t>(A_CONS + 2 * i), y = get_arg_val<uint32_t>(A_CONS + 2 * i + 1);
                noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_ON)), 1);
            }
        }
    };

    for (uint32_t c = 0; c < nvalid; ++c) {
        if constexpr (SPREAD > 1) {
            // my reservation of this chunk's PIN pages has the same L1 address on every peer (identical CB layout and
            // reservation history): peers write their partial into slot `rank` of it
            cb_reserve_back(CB_PIN, SPREAD - 1);
            const uint32_t pin = get_write_ptr(CB_PIN);
            cb_wait_front(CB_PSUM, 1);
            const uint32_t src = get_read_ptr(CB_PSUM);
            for (uint32_t p = 0; p < SPREAD - 1; ++p) {
                const uint32_t px = get_arg_val<uint32_t>(A_PEERS + 2 * p),
                               py = get_arg_val<uint32_t>(A_PEERS + 2 * p + 1);
                const uint32_t peer_vblk = p < vblk ? p : p + 1;
                const uint32_t slot = vblk < peer_vblk ? vblk : vblk - 1;
                noc_async_write(src, get_noc_addr(px, py, pin + slot * T32), T32);
            }
            noc_async_write_barrier();
            for (uint32_t p = 0; p < SPREAD - 1; ++p) {
                const uint32_t px = get_arg_val<uint32_t>(A_PEERS + 2 * p),
                               py = get_arg_val<uint32_t>(A_PEERS + 2 * p + 1);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_RMS)), 1);
            }
            cb_pop_front(CB_PSUM, 1);
            noc_semaphore_wait_min(sem_rms, (SPREAD - 1) * (c + 1));
            cb_push_back(CB_PIN, SPREAD - 1);
        } else {
            cb_wait_front(CB_PSUM, 1);
            cb_pop_front(CB_PSUM, 1);
        }
        cb_wait_front(CB_ON, Vt);
        const uint32_t ob = get_read_ptr(CB_ON);
        if (pr_chunks == 0) {
            pr_chk_on = probe_sum(ob, T16 / 4);
        }
        ++pr_chunks;
        for (uint32_t v = 0; v < Vt; ++v) {
            noc_async_write(ob + v * T16, on_g.get_noc_addr(c * WT + head * VTF + vblk * Vt + v), T16);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_ON, Vt);
        chunk_done(c);
    }
    // P9: the final state (unchanged when no chunk was valid)
    if (nvalid > 0) {
        cb_wait_front(CB_FS, KV);
        const uint32_t fb = get_read_ptr(CB_FS);
        pr_chk_fs = probe_sum(fb, T32 / 4);
        for (uint32_t k = 0; k < Kt; ++k) {
            for (uint32_t v = 0; v < Vt; ++v) {
                noc_async_write(
                    fb + (k * Vt + v) * T32, s_g.get_noc_addr(head * Kt * VTF + k * VTF + vblk * Vt + v), T32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_FS, KV);
    }
    // dead chunks: finite (zero) output rows, still counted as done for the P7 consumers
    for (uint32_t c = nvalid; c < NCH; ++c) {
        for (uint32_t v = 0; v < Vt; ++v) {
            zero_dram(on_g, c * WT + head * VTF + vblk * Vt + v, T16);
        }
        chunk_done(c);
    }
    // the conv carry (one core per chip): rows ve - 3 .. ve - 1 of u (rows before 0: the previous carry's rows ve + i)
    // -> row 0 of the carry tiles (i, t). NoC rule (Blackhole): DRAM source and L1 destination share their residue mod
    // 64, so every DRAM transfer here is a WHOLE tile: u / carry tiles are read in batches into the drained IN ring,
    // the row is extracted in L1 into a row buffer (the drained SR ring), and the carry tiles are read, patched in
    // row 0 and written back whole.
    if (carry_core && ve > 0) {
        constexpr uint32_t NT = 3 * WT;          // column tiles of a carry row
        constexpr uint32_t NB = IN_BYTES / T16;  // tiles per staging batch
        static_assert(NB >= 1, "carry staging");
        const uint32_t stg = get_write_ptr(CB_IN);     // this RISC never moves the IN / SR pointers: their bases
        const uint32_t rowbuf = get_write_ptr(CB_SR);  // NT x 64 B (the host checks it fits the SR ring)
        auto copy_row = [](uint32_t src_tile, uint32_t row, uint32_t dst64) {  // tile row -> 64 B at dst64
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst64);
            for (uint32_t hf = 0; hf < 2; ++hf) {
                const uint32_t so = tile_off(row, 16 * hf) / 2;
                for (uint32_t w = 0; w < 8; ++w) {
                    dst[8 * hf + w] = src[so + w];
                }
            }
        };
        auto patch_row0 = [](uint32_t src64, uint32_t dst_tile) {  // 64 B -> row 0 of a tile
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src64);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile);
            for (uint32_t hf = 0; hf < 2; ++hf) {
                const uint32_t dO = tile_off(0, 16 * hf) / 2;
                for (uint32_t w = 0; w < 8; ++w) {
                    dst[dO + w] = src[8 * hf + w];
                }
            }
        };
        for (uint32_t i = 0; i < 3; ++i) {
            ++pr_carry;
            const int32_t r = static_cast<int32_t>(ve) - 3 + static_cast<int32_t>(i);
            const bool old = r < 0;  // the previous carry's row ve + i (its tiles are not yet overwritten: ve + i > i)
            const uint32_t rt = old ? 0 : static_cast<uint32_t>(r) / CHUNK,
                           rr = old ? 0 : static_cast<uint32_t>(r) % CHUNK;
            for (uint32_t b0 = 0; b0 < NT; b0 += NB) {
                const uint32_t n = NT - b0 < NB ? NT - b0 : NB;
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t t = b0 + k;
                    if (old) {
                        noc_async_read(cst_g.get_noc_addr((ve + i) * NT + t), stg + k * T16, T16);
                    } else {
                        noc_async_read(u_g.get_noc_addr(rt * UNT + t), stg + k * T16, T16);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t k = 0; k < n; ++k) {
                    copy_row(stg + k * T16, rr, rowbuf + (b0 + k) * 64);
                }
            }
            for (uint32_t b0 = 0; b0 < NT; b0 += NB) {
                const uint32_t n = NT - b0 < NB ? NT - b0 : NB;
                for (uint32_t k = 0; k < n; ++k) {
                    noc_async_read(cst_g.get_noc_addr(i * NT + b0 + k), stg + k * T16, T16);
                }
                noc_async_read_barrier();
                for (uint32_t k = 0; k < n; ++k) {
                    patch_row0(rowbuf + (b0 + k) * 64, stg + k * T16);
                    noc_async_write(stg + k * T16, cst_g.get_noc_addr(i * NT + b0 + k), T16);
                }
                noc_async_write_barrier();
            }
        }
    }
    noc_async_atomic_barrier();
    {  // probe row (words 0..15): the drained FS pages as scratch
        const uint32_t st = get_write_ptr(CB_FS);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        w[0] = PROBE_SCAN_W;
        w[1] = ve;
        w[2] = nvalid;
        w[3] = pr_chunks;
        w[4] = pr_chk_on;
        w[5] = pr_chk_fs;
        w[6] = head;
        w[7] = vblk;
        w[8] = pr_carry;
        w[9] = leader ? 1u : 0u;
        for (uint32_t a = 0; a < 6; ++a) {  // the flags / coordinates as this kernel reads them: args 7..12
            w[10 + a] = get_arg_val<uint32_t>(ph_rt0 + 7 + a);
        }
        noc_async_write(st, probe_g.get_noc_addr(n_item + head * SPREAD + vblk, 0), 64);
        noc_async_write_barrier();
    }
}
