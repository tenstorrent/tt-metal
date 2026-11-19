// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase (kernels/mla/att_common.hpp) — NCRISC (NoC1) of owner core (g, c): the partial
// tree of the group, per token: at a receive level, free CB_ARX, tell the child (SEM_RDY), wait for its partial
// (SEM_DAT), hand it to the compute; at the send level, wait for the compute's partial in CB_ASEND and for the
// parent's ready, write the 18 fp32 tiles into the parent's CB_ARX (the same L1 address on every owner: identical CB
// layout, one message per CB fill so the CB always starts at its base) and signal SEM_DAT. Core c = 0 writes the
// group's o tile row of o_lat [1, T, H, 512]; after the last token it signals the o_lat consumers (SIGNAL_SEM).
// Compile-time args: CB_ARX, CB_ASEND, CB_AOUT, C, LV (tree levels), HPT, SEM_RDY, SEM_DAT, SIGNAL_SEM, NCONS, CB_ARXO,
//   CB_ASENDO (the O halves of the receive / send partials), SEM_DATO (O tiles landed, cumulative per message), then
//   the TensorAccessorArgs of o_lat.
// Runtime args: o_lat address, g, c, T, then per level L 2 values (child or parent NoC x, y), then NCONS consumer
//   (x, y) pairs (core c = 0 only) (+ the token lane with MLA_ATT_LANES: c / C / LV are the lane's, att_common.hpp)
//   (+ the o_T address with MLA_ATT_OT).
// MLA_ATT_OT = HR (rev 30, tt/mk_mla.py ot(t)): the root also writes W_uv's in0 rows -- head (g * 32 + h)'s row of
// token t = row h of o tile d -> row t of column tile (g * 32 + h) * 16 + d of o_T [1, 1, T, H * 512] (token rows), two
// 32-B face-row writes per (head, tile), h < HR (the group's real heads) -- so the W_uv readers pull whole tiles
// instead of gathering 1024 face rows each (o_T's accessor args follow o_lat's).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"

void kernel_main() {
    constexpr uint32_t CB_ARX = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t CB_ASEND = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t CB_AOUT = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t C = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t LV = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t HPT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t SEM_RDY = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t SEM_DAT = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t SIGNAL_SEM = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t CB_ARXO = get_compile_time_arg_val(PH_CT0 + 10);
    constexpr uint32_t CB_ASENDO = get_compile_time_arg_val(PH_CT0 + 11);
    constexpr uint32_t SEM_DATO = get_compile_time_arg_val(PH_CT0 + 12);
    constexpr auto o_args = TensorAccessorArgs<PH_CT0 + 13>();
    constexpr uint32_t ST = mlaatt::STAT_TILES, DT = mlaatt::DT;
    const auto olat = TensorAccessor(o_args, get_arg_val<uint32_t>(ph_rt0 + 0));
#if defined(MLA_ATT_OT)
    constexpr auto ot_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t HR = MLA_ATT_OT;
    static_assert(HR >= 1 && HR <= 32, "o_T: real heads per 32-row group");
#endif
    const uint32_t g = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t lv0 = ph_rt0 + 4;
    const uint32_t arx_base = get_local_cb_interface(CB_ARX).fifo_limit - get_local_cb_interface(CB_ARX).fifo_size;
    const uint32_t arxo_base = get_local_cb_interface(CB_ARXO).fifo_limit - get_local_cb_interface(CB_ARXO).fifo_size;
    const uint32_t ptb = get_tile_size(CB_ARX);   // fp32 stats tile
    const uint32_t otb = get_tile_size(CB_ARXO);  // O tile (fp32, or bf16 with GLM53_MLA_ATT_OBF16)
    constexpr uint32_t OG = 4;                    // O tiles per streamed group
    static_assert(mlaatt::DT % OG == 0, "O groups");
    volatile tt_l1_ptr uint32_t* dato = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DATO));

    MK_MARK();
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(lv0 + 2 * LV + 2 * NCONS);
    constexpr uint32_t LANE_ARGS = 1;
#else
    const uint32_t t0 = 0;
    constexpr uint32_t LANE_ARGS = 0;
#endif
#if defined(MLA_ATT_OT)
    const auto ot = TensorAccessor(ot_args, get_arg_val<uint32_t>(lv0 + 2 * LV + 2 * NCONS + LANE_ARGS));
#endif
    (void)LANE_ARGS;
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        for (uint32_t L = 0; L < LV; ++L) {
            const uint32_t px = get_arg_val<uint32_t>(lv0 + 2 * L), py = get_arg_val<uint32_t>(lv0 + 2 * L + 1);
            if (mlaatt::recv_at(c, L, C)) {
                // streamed message: the stats, then O in groups of OG tiles as the child packs them
                cb_reserve_back(CB_ARX, ST);  // the previous message consumed: the regions are free
                cb_reserve_back(CB_ARXO, DT);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_RDY)), 1);
                noc_async_atomic_barrier();
                mk::wait_handoff(SEM_DAT, 1);
                if (t == 0) {
                    MK_MARK();  // probe: level L's child stats in
                }
                cb_push_back(CB_ARX, ST);
                for (uint32_t d0 = 0; d0 < DT; d0 += OG) {
                    noc_semaphore_wait_min(dato, d0 + OG);
                    cb_push_back(CB_ARXO, OG);
                }
                noc_semaphore_set(dato, 0);  // before the next level's ready: that child counts from 0
                if (t == 0) {
                    MK_MARK();  // probe: level L's child O in
                }
            } else if (mlaatt::send_at(c, L)) {
                mk::wait_handoff(SEM_RDY, 1);
                cb_wait_front(CB_ASEND, ST);
                if (t == 0) {
                    MK_MARK();  // probe: this core's partial (stats) ready to send
                }
                noc_async_write(get_read_ptr(CB_ASEND), get_noc_addr(px, py, arx_base), ST * ptb);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_DAT)), 1);
                cb_pop_front(CB_ASEND, ST);
                const uint64_t dst_o = get_noc_addr(px, py, arxo_base);
                const uint64_t sem_o = get_noc_addr(px, py, get_semaphore(SEM_DATO));
                for (uint32_t d0 = 0; d0 < DT; d0 += OG) {
                    cb_wait_front(CB_ASENDO, d0 + OG);
                    noc_async_write(get_read_ptr(CB_ASENDO) + d0 * otb, dst_o + d0 * otb, OG * otb);
                    noc_async_write_barrier();  // the tiles land before the count that announces them
                    noc_semaphore_inc(sem_o, OG);
                }
                noc_async_atomic_barrier();
                cb_pop_front(CB_ASENDO, DT);
                break;
            }
        }
        if (c == 0) {  // the root's o tiles as the compute packs them
            const uint32_t tb = get_tile_size(CB_AOUT);
            for (uint32_t d = 0; d < mlaatt::DT; ++d) {
                cb_wait_front(CB_AOUT, d + 1);
                noc_async_write_page((t * HPT + g) * mlaatt::DT + d, olat, get_read_ptr(CB_AOUT) + d * tb);
#if defined(MLA_ATT_OT)
                // the tile's head rows -> row t of each head's column tile d of o_T. A NoC transfer must stay
                // congruent mod 64 (mla_common.hpp): a row whose source offset differs from its destination's goes
                // through a 64-B staging slot at the destination's offset (the RISC copies it); the root never
                // sends, so its send CB is the staging area. Slots are reused per tile after the previous tile's
                // writes have read them (flushed).
                const uint32_t src = get_read_ptr(CB_AOUT) + d * tb;
                const uint32_t drow = mla::row_off(t);
                const uint32_t stg = (get_write_ptr(CB_ASEND) + 63) & ~63u;
                uint64_t dst[HR];
                uint32_t from[HR];
                uint32_t staged = 0;         // bit h: head h's rows went through slots 2 h / 2 h + 1
                noc_async_writes_flushed();  // the previous tile's staged rows have left L1
                invalidate_l1_cache();       // the packer's o tile, not stale cache lines
                for (uint32_t h = 0; h < HR; ++h) {
                    dst[h] = ot.get_noc_addr((g * 32 + h) * mlaatt::DT + d, drow);
                    const uint32_t off = static_cast<uint32_t>(dst[h]) & 63;  // the same for both faces (+512)
                    from[h] = src + mla::row_off(h);
                    if ((from[h] & 63) != off) {
                        const uint32_t slot = stg + 2 * h * 64 + off;
                        mla::copy_row32(from[h], slot);
                        mla::copy_row32(from[h] + 512, slot + 64);
                        from[h] = slot;
                        staged |= 1u << h;
                    }
                }
                invalidate_l1_cache();  // the staged rows land before the NoC reads them
                for (uint32_t h = 0; h < HR; ++h) {
                    noc_async_write(from[h], dst[h], 32);
                    noc_async_write(from[h] + (((staged >> h) & 1) ? 64 : 512), dst[h] + 512, 32);
                }
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(CB_AOUT, mlaatt::DT);
        }
    }
    MK_MARK();
    if constexpr (NCONS > 0) {
        if (c == 0) {
            mk::signal_handoff(SIGNAL_SEM, lv0 + 2 * LV, NCONS);
        }
    }
}
