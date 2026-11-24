// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA scan cores — reader (BRISC, NoC0): core (head, v-block): the head's initial state S (its Vt value-column
// tiles of every key tile) from the per-layer state scratch, the o_norm row tile(s) + a ones tile, then per valid
// chunk, after the item cores' hand-off (SEM_KDA >= c + 1, signalled in chunk order), the chunk's terms from the TERMS
// scratch (this core's v-block of v_beta; the whole kd / q_decay / intra / t_inv / final_decay / k_dec_t) and the
// gate_out tiles. Derived from ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/
// reader_recurrent_chunk_scan.cpp (read_and_publish_value_slice / contiguous tiles, the chronology's valid chunks)
// with one merged per-chunk ring and the kda_rec reader's norm weight tile.
// Compile-time (PH_CT0 +): Kt, Vt (value tiles of this core), VTF (value tiles of a head), NCH, CPH, WT (the chip's
// width tiles = H VTF), TB (term tile bytes: the TERMS / gate_out tensors' and the IN ring's page size: 4096 fp32 /
// 2048 bf16 -- the chunk-term knob). Runtime (ph_rt0 +): head, vblk, s (state scratch [1, H, 128, 128] fp32 TILE),
// terms, go ([1,1,C,W] fp32 TILE), w ([1, 1, 32, 128] fp32 TILE, row 0 = o_norm), valid_end, probe (row = N_ITEM + head
// * SPREAD
// + vblk, words 16..), N_ITEM, SPREAD.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"

using namespace pk;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t VTF = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t NCH = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t CPH = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t WT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t TB = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t NP = out_pages(Kt, VTF);  // TERMS pages per item
    constexpr uint32_t PG = in_pages(Kt, Vt);    // IN ring pages per chunk

    const uint32_t head = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t vblk = get_arg_val<uint32_t>(ph_rt0 + 1);
    const InterleavedAddrGen<true> s_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 2), .page_size = T32};
    const InterleavedAddrGen<true> terms{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 3), .page_size = TB};
    const InterleavedAddrGen<true> go_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 4), .page_size = TB};
    const InterleavedAddrGen<true> w_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 5), .page_size = T32};
    const InterleavedAddrGen<true> valid_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 6), .page_size = 4};
    const InterleavedAddrGen<true> probe_g{
        .bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 7), .page_size = PROBE_ROW_BYTES};
    const uint32_t n_item = get_arg_val<uint32_t>(ph_rt0 + 8), spread = get_arg_val<uint32_t>(ph_rt0 + 9);
    uint32_t pr_chunks = 0, pr_chk_in = 0, pr_chk_s = 0;

    // W: pages 0 .. Vt-1 = o_norm row tiles of this v-block (+ control words), page Vt = ones (the reduce scaler)
    cb_reserve_back(CB_W, Vt + 1);
    const uint32_t wb = get_write_ptr(CB_W);
    noc_async_read(valid_g.get_noc_addr(0), wb, 4);
    noc_async_read_barrier();
    const uint32_t ve = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb)[0];
    uint32_t nvalid = valid_chunks(ve);
    if (nvalid > NCH) {
        nvalid = NCH;
    }
    for (uint32_t v = 0; v < Vt; ++v) {
        noc_async_read(w_g.get_noc_addr(vblk * Vt + v), wb + v * T32, T32);
    }
    noc_async_read_barrier();
    {
        volatile tt_l1_ptr uint32_t* w0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb);
        w0[CTL_ROW] = nvalid;
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb + Vt * T32);
        for (uint32_t i = 0; i < 1024; ++i) {
            ones[i] = 0x3F800000u;
        }
    }
    cb_push_back(CB_W, Vt + 1);

    // S: tiles (k, v) of the head: page h Kt VTF + k VTF + vblk Vt + v
    cb_reserve_back(CB_S, Kt * Vt);
    const uint32_t sb = get_write_ptr(CB_S);
    for (uint32_t k = 0; k < Kt; ++k) {
        for (uint32_t v = 0; v < Vt; ++v) {
            noc_async_read(s_g.get_noc_addr(head * Kt * VTF + k * VTF + vblk * Vt + v), sb + (k * Vt + v) * T32, T32);
        }
    }
    noc_async_read_barrier();
    pr_chk_s = probe_sum(sb, T32 / 4);
    cb_push_back(CB_S, Kt * Vt);

    for (uint32_t c = 0; c < nvalid; ++c) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_KDA)), c + 1);
        cb_reserve_back(CB_IN, PG);
        const uint32_t ib = get_write_ptr(CB_IN);
        const uint32_t base = (head * NCH + c) * NP;
        auto rd = [&](uint32_t page, uint32_t term) {
            noc_async_read(terms.get_noc_addr(base + term), ib + page * TB, TB);
        };
        for (uint32_t v = 0; v < Vt; ++v) {
            rd(in_vbeta(Kt, Vt) + v, off_vbeta(Kt, VTF) + vblk * Vt + v);
        }
        for (uint32_t k = 0; k < Kt; ++k) {
            rd(in_kd(Kt, Vt) + k, off_kd(Kt, VTF) + k);
            rd(in_qdec(Kt, Vt) + k, off_qdec(Kt, VTF) + k);
            rd(in_fd(Kt, Vt) + k, off_fd(Kt, VTF) + k);
            rd(in_kdt(Kt, Vt) + k, off_kdt(Kt, VTF) + k);
        }
        rd(in_intra(Kt, Vt), off_intra(Kt, VTF));
        rd(in_tinv(Kt, Vt), off_tinv(Kt, VTF));
        for (uint32_t v = 0; v < Vt; ++v) {
            noc_async_read(go_g.get_noc_addr(c * WT + head * VTF + vblk * Vt + v), ib + (in_go(Kt, Vt) + v) * TB, TB);
        }
        noc_async_read_barrier();
        if (pr_chunks == 0) {
            pr_chk_in = probe_sum(ib, TB / 4);
        }
        ++pr_chunks;
        cb_push_back(CB_IN, PG);
    }
    {  // probe row (words 16..31): one more IN reservation (free once the compute drained the ring) as scratch
        cb_reserve_back(CB_IN, PG);
        const uint32_t st = get_write_ptr(CB_IN);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        w[0] = PROBE_SCAN_R;
        w[1] = ve;
        w[2] = nvalid;
        w[3] = pr_chunks;
        w[4] = pr_chk_in;
        w[5] = pr_chk_s;
        w[6] = head;
        w[7] = vblk;
        noc_async_write(st, probe_g.get_noc_addr(n_item + head * spread + vblk, 64), 64);
        noc_async_write_barrier();
    }
}
