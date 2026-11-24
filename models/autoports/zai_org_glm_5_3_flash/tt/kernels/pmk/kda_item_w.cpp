// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA item cores — writer (NCRISC, NoC1): the chunk terms of every live item of this core (v_beta, q_decay, kd,
// intra, t_inv, final_decay, k_dec_t: OUT_PAGES fp32 tiles in kda_common.hpp's order) to the TERMS DRAM scratch at
// item (head, chunk), then the hand-off to the head's scan cores: +1 on their SEM_KDA counter, in CHUNK ORDER across
// the head's CPH item cores (the baton: before signalling chunk c = slot + k CPH the writer waits for its own SEM_KDA
// >= the number of earlier chunks the previous core signalled (k + 1 for slot > 0, k for slot 0), and after signalling
// it increments SEM_KDA on the next core (slot + 1) % CPH), so the scan cores' counter >= c + 1 means chunks 0..c.
// Derived from ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/
// writer_prepare_chunk_recurrence.cpp (drain per work item, dead chunks skipped) with one merged output CB.
// Compile-time (PH_CT0 +): Kt, VTF, NCH, CPH, NSCAN (scan cores per head = SPREAD), TB (term tile bytes: 4096 fp32 /
// 2048 bf16 = the OUT CB's page size and the TERMS scratch's).
// Runtime (ph_rt0 +): head, slot, terms, valid_end, next core (x, y) (slot + 1 of the head), probe, then (x, y) of
// the head's NSCAN scan cores.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"

using namespace pk;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t VTF = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t NCH = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t CPH = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t NSCAN = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t TB = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t NP = out_pages(Kt, VTF);

    const uint32_t head = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t slot = get_arg_val<uint32_t>(ph_rt0 + 1);
    const InterleavedAddrGen<true> terms{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 2), .page_size = TB};
    const InterleavedAddrGen<true> valid_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 3), .page_size = 4};

    // valid_end: staged in the tail of the OUT ring's last page (the compute reaches it only at the end of item 1)
    const uint32_t scratch = get_write_ptr(CB_OUT) + (2 * NP - 1) * TB + TB - 64;
    noc_async_read(valid_g.get_noc_addr(0), scratch, 4);
    noc_async_read_barrier();
    const uint32_t ve = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];

    const uint32_t nx = get_arg_val<uint32_t>(ph_rt0 + 4), ny = get_arg_val<uint32_t>(ph_rt0 + 5);
    const InterleavedAddrGen<true> probe_g{
        .bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 6), .page_size = PROBE_ROW_BYTES};
    uint32_t pr_items = 0, pr_chk = 0;
    volatile tt_l1_ptr uint32_t* baton = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_KDA));
    for (uint32_t c = slot; c < NCH; c += CPH) {
        if (chunk_rows(ve, c) == 0) {
            break;  // dead chunks are a suffix: nothing to signal (the scan reads nvalid chunks only)
        }
        cb_wait_front(CB_OUT, NP);
        const uint32_t src = get_read_ptr(CB_OUT);
        if (pr_items == 0) {
            pr_chk = probe_sum(src, 256);  // probe: the first item's first terms page (its first 1 KB)
        }
        ++pr_items;
        const uint32_t base = (head * NCH + c) * NP;
        for (uint32_t j = 0; j < NP; ++j) {
            noc_async_write(src + j * TB, terms.get_noc_addr(base + j), TB);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, NP);
        const uint32_t k = c / CPH;
        if (c > 0) {
            noc_semaphore_wait_min(baton, slot > 0 ? k + 1 : k);  // chunk c - 1 signalled by the previous core
        }
        for (uint32_t s = 0; s < NSCAN; ++s) {
            const uint32_t x = get_arg_val<uint32_t>(ph_rt0 + 7 + 2 * s), y = get_arg_val<uint32_t>(ph_rt0 + 8 + 2 * s);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_KDA)), 1);
        }
        noc_async_atomic_barrier();  // the scan cores' counter lands before the baton moves on
        noc_semaphore_inc(get_noc_addr(nx, ny, get_semaphore(SEM_KDA)), 1);  // the baton (CPH 1: itself)
    }
    noc_async_atomic_barrier();
    {  // probe row (words 0..15): the OUT ring's tail (past the valid_end staging) as scratch
        const uint32_t st = get_write_ptr(CB_OUT) + (2 * NP - 1) * TB + TB - 128;
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        w[0] = PROBE_ITEM_W;
        w[1] = ve;
        w[2] = NCH;
        w[3] = pr_items;
        w[4] = pr_chk;
        w[5] = head;
        w[6] = slot;
        for (uint32_t a = 0; a < 3; ++a) {  // args 4..6 (baton core x / y, probe) as read
            w[7 + a] = get_arg_val<uint32_t>(ph_rt0 + 4 + a);
        }
        noc_async_write(st, probe_g.get_noc_addr(head * CPH + slot, 0), 64);
        noc_async_write_barrier();
    }
}
