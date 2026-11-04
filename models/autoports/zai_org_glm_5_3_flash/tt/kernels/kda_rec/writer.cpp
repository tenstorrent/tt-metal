// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// kda_rec writer (NCRISC / NoC1): per item (b, h) the gated-normed o row (bf16, row 0 of CB_ON's Vt tiles) into row b
// of this head's Vt o_proj-input tiles assembled in L1 (CB_OT, rows >= B zero; a head owns whole column tiles of the
// [1, 1, B <= 32, H * V] input), written as whole pages after the last item; and the item's [K, V] fp32 state tiles
// back into the state tensor (in place).
// Megakernel: after the last item's writes landed, increments MK_REC_SIGNAL_SEM on each consumer (the o_proj readers).
// REC_CHAIN (Stage 17 verify): after row b's state write landed (b < items - 1), this core's REC_CHAIN_SEM += 1 (the
// reader's row b + 1 starts from it; kernels/kda_rec/reader.cpp).
// Compile-time args (PH_CT0 + i): H, Kt, Vt, the o_proj input geometry (cols, rows, row tiles, col tiles), then the
// tensor accessors of the o_proj input and the state.
// Runtime args (ph_rt0 + i): core index (head = index / REC_SPREAD, value-column block = index % REC_SPREAD), items,
// o_proj input address, state address, debug tap address (REC_DEBUG: the pre-norm o rows into a zeroed [1,1,B,H*V] fp32
// tensor; 0 otherwise), consumers n, then n NoC (x, y) (MK_REC_SIGNAL_SEM), then (REC_SPREAD > 1) the NoC (x, y) of the
// head's other cores in block order.
// REC_SPREAD > 1: per item, this core's partial sum of squares (CB_PSUM, first 64 B = row 0 of face 0) is written to
// page (b % 2) * (S - 1) + slot of each peer's CB_PIN (slot = this block's rank among the peer's peers) + REC_RMS_SEM
// increment; once the peers' S - 1 partials of item b landed, CB_PIN's S - 1 pages are pushed to compute. Two item
// slots suffice: a peer's partial of item b + 2 is causally after this core's partial of item b + 1, which compute
// packs only after it popped item b's pages.

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#if defined(REC_ITEM_MARKS)  // Stage 14 (probe runs): per item, the peers' norm partials in / the o row out / the state
                             // written
#define REC_MARK() MK_MARK()
#else
#define REC_MARK() \
    do {           \
    } while (0)
#endif

void kernel_main() {
    using namespace kda_rec;
    constexpr uint32_t FACE_BYTES16 = 16 * 16 * 2;
    constexpr uint32_t FACE_ROW_BYTES16 = 16 * 2;
    constexpr uint32_t H = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t o_cols = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t o_rows = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t o_row_tiles = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t o_col_tiles = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr auto o_a = TensorAccessorArgs<PH_CT0 + 7>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
#if defined(REC_DEBUG)
    constexpr auto tap_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();
    const auto tap_acc = TensorAccessor(tap_a, get_arg_val<uint32_t>(ph_rt0 + 4), 4096);
#endif
    constexpr uint32_t V = Vt * 32;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t VTC = Vt / REC_SPREAD;  // this core's value-column tiles
    constexpr uint32_t KVC = Kt * VTC;
    static_assert(Vt % REC_SPREAD == 0, "REC_SPREAD divides the value tiles");
    static_assert(o_rows <= 32 && o_cols % 32 == 0, "decode geometry");

    const uint32_t hc = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t h = hc / REC_SPREAD, blk = hc % REC_SPREAD;
    const uint32_t items = get_arg_val<uint32_t>(ph_rt0 + 1) & 0xFFFF;  // (b0 << 16) | n (Stage 14 user groups)
    [[maybe_unused]] const uint32_t b0 = get_arg_val<uint32_t>(ph_rt0 + 1) >> 16;
    if (h >= H) {
        return;  // a core past the shard's heads (megakernel owner list): no recurrence work, no hand-off
    }
    const auto o_acc = TensorAccessor(o_a, get_arg_val<uint32_t>(ph_rt0 + 2), get_tile_size(CB_ON));
    const auto st_acc = TensorAccessor(st_a, get_arg_val<uint32_t>(ph_rt0 + 3), get_tile_size(CB_ST));
    const uint32_t tb16 = get_tile_size(CB_ON);
    const uint32_t tb32 = get_tile_size(CB_ST);

    static_assert(o_row_tiles == 1, "decode: B <= 32 rows (one row of tiles)");
    Noc noc;
    CircularBuffer con(CB_ON), cst(CB_ST), cot(CB_OT);

    // this head's o_proj input tiles (columns h * V .., B <= 32 rows): assembled in L1 with rows >= B zero, written
    // once as whole pages after the last item — o_proj's padding rows stay finite, Vt page writes instead of 2 Vt
    // face-row writes per item
    cot.reserve_back(VTC);
    const uint32_t ot = cot.get_write_ptr();
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ot);
        for (uint32_t i = 0; i < VTC * tb16 / 4; i++) {
            z[i] = 0;
        }
    }
    const uint32_t page0 = (h * V % o_cols) / 32 + blk * VTC;  // row tile 0, this core's first column tile
#if REC_SPREAD > 1
#if defined(MK_REC_SIGNAL_SEM)
    const uint32_t peers_rt = ph_rt0 + 6 + 2 * get_arg_val<uint32_t>(ph_rt0 + 5);
#else
    const uint32_t peers_rt = ph_rt0 + 5;
#endif
    constexpr uint32_t NP = REC_SPREAD - 1;
    CircularBuffer cpsum(CB_PSUM), cpin(CB_PIN);
    const uint32_t pin_base = cpin.get_write_ptr();  // this RISC's CB_PIN pointer is at the base before the first push
    const uint32_t pin_page = get_tile_size(CB_PIN);
    const uint32_t rms_sem = get_semaphore(REC_RMS_SEM);
    uint64_t peer_pin[NP], peer_sem[NP];
    for (uint32_t i = 0; i < NP; i++) {
        const uint32_t px = get_arg_val<uint32_t>(peers_rt + 2 * i), py = get_arg_val<uint32_t>(peers_rt + 2 * i + 1);
        const uint32_t pb = i < blk ? i : i + 1;         // the peer's block
        const uint32_t slot = blk < pb ? blk : blk - 1;  // this block's page among the peer's peers
        peer_pin[i] = get_noc_addr(px, py, pin_base + slot * pin_page);
        peer_sem[i] = get_noc_addr(px, py, rms_sem);
    }
#endif

    for (uint32_t bl = 0; bl < items; bl++) {
        const uint32_t b = b0 + bl;  // the output row (REC_UG: rows of several cores share the o_proj input tiles)
        const uint32_t item = b * H + h;
#if REC_SPREAD > 1
        {
            cpsum.wait_front(1);
            const uint32_t src = cpsum.get_read_ptr();
            const uint32_t off = (bl % 2) * NP * pin_page;
            for (uint32_t i = 0; i < NP; i++) {
                noc_async_write(src, peer_pin[i] + off, 64);  // row 0 of face 0: [0][0] is the item's partial
            }
            noc_async_write_barrier();
            for (uint32_t i = 0; i < NP; i++) {
                noc_semaphore_inc(peer_sem[i], 1);
            }
            cpsum.pop_front(1);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rms_sem), NP * (bl + 1));
            REC_MARK();  // item b: the head's norm partials exchanged
            cpin.reserve_back(NP);
            cpin.push_back(NP);
        }
#endif
        con.wait_front(VTC);
        invalidate_l1_cache();  // packer-written tiles read by this RISC
        const uint32_t src = con.get_read_ptr();
        for (uint32_t j = 0; j < VTC; j++) {
            for (uint32_t f = 0; f < 2; f++) {  // the two 16-value face rows of tile row 0 -> tile row b
                volatile tt_l1_ptr uint32_t* s =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + j * tb16 + f * FACE_BYTES16);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    ot + j * tb16 + ((b / 16) * 2 + f) * FACE_BYTES16 + (b % 16) * FACE_ROW_BYTES16);
                for (uint32_t w = 0; w < FACE_ROW_BYTES16 / 4; w++) {
                    d[w] = s[w];
                }
            }
        }
        con.pop_front(VTC);
#if defined(REC_UG)
        // user groups: this core owns only some rows of the o_proj input tiles -> its row's 2 face rows per tile
        // (the padding rows stay the zeros the host wrote once)
        for (uint32_t j = 0; j < VTC; j++) {
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t off = ((b / 16) * 2 + f) * FACE_BYTES16 + (b % 16) * FACE_ROW_BYTES16;
                noc_async_write(ot + j * tb16 + off, o_acc.get_noc_addr(page0 + j, off), FACE_ROW_BYTES16);
            }
        }
#endif
#if defined(REC_DEBUG)
        {
            CircularBuffer ctap(CB_TAP);
            ctap.wait_front(Vt);
            invalidate_l1_cache();
            const uint32_t tsrc = ctap.get_read_ptr();
            for (uint32_t j = 0; j < Vt; j++) {
                for (uint32_t f = 0; f < 2; f++) {  // fp32 face rows of tile row 0 -> row b of the tap page
                    noc_async_write(
                        tsrc + j * 4096 + f * 1024,
                        tap_acc.get_noc_addr(page0 + j, ((b / 16) * 2 + f) * 1024 + (b % 16) * 64),
                        64);
                }
            }
            noc_async_write_barrier();
            ctap.pop_front(Vt);
        }
#endif

#if defined(REC_EARLY_SIGNAL)
        if (bl + 1 == items) {  // Stage 14 glue trim: publish the o_proj input before the last item's state write-back
#if !defined(REC_UG)
            for (uint32_t j = 0; j < VTC; j++) {
                noc_async_write_page(page0 + j, o_acc, ot + j * tb16);
            }
#endif
            noc_async_write_barrier();
#if defined(MK_REC_SIGNAL_SEM)
            mk::signal_handoff(MK_REC_SIGNAL_SEM, ph_rt0 + 6, get_arg_val<uint32_t>(ph_rt0 + 5));
#endif
        }
#endif
        cst.wait_front(KVC);
        auto ssrc = use<CircularBuffer::AddrSelector::READ_PTR>(cst);
        for (uint32_t mi = 0; mi < Kt; mi++) {  // CB tile mi * VTC + ni = state tile (mi, blk * VTC + ni)
            for (uint32_t ni = 0; ni < VTC; ni++) {
                noc.async_write(
                    ssrc,
                    st_acc,
                    tb32,
                    {.offset_bytes = (mi * VTC + ni) * tb32},
                    {.page_id = item * KV + mi * Vt + blk * VTC + ni});
            }
        }
        noc.async_write_barrier();
        cst.pop_front(KVC);
#if defined(REC_CHAIN)
        if (bl + 1 < items) {
            mk::signal_self(REC_CHAIN_SEM);  // Stage 17 verify: the next row's reader starts from this state
        }
#endif
        REC_MARK();  // item b: state written back
    }
#if defined(REC_EARLY_SIGNAL) && defined(REC_UG) && defined(MK_REC_SIGNAL_SEM)
    if (items == 0) {  // a user group without rows still hands off (o_proj counts every head core)
        mk::signal_handoff(MK_REC_SIGNAL_SEM, ph_rt0 + 6, get_arg_val<uint32_t>(ph_rt0 + 5));
    }
#endif
#if !defined(REC_EARLY_SIGNAL)
#if !defined(REC_UG)
    for (uint32_t j = 0; j < VTC; j++) {
        noc_async_write_page(page0 + j, o_acc, ot + j * tb16);
    }
#endif
    noc_async_write_barrier();

#if defined(MK_REC_SIGNAL_SEM)
    mk::signal_handoff(MK_REC_SIGNAL_SEM, ph_rt0 + 6, get_arg_val<uint32_t>(ph_rt0 + 5));
#endif
#endif
}
