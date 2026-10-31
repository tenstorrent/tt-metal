// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Split decode attention, writer (BRISC) (fused_decode/split_attn.py, sda_common.hpp): tells compute its slice (one
// mailbox word), then
//   - all-gathers the slice's row max: m[0..7] (column 0 of rows 0..7 of CB_MOUT) into slot s of CB_MG on every
//     active slice of kv head g (itself included), incrementing their program semaphore SDA_M_SEM;
//   - writes the scaled state (CB_PART: FP32 [O_0, O_1, L], rows 0..7) into slot s % 4 of pack tile s / 4 of the
//     leader's CB_OPACK (rows 8 i .. 8 i + 7) and increments the leader's SDA_P_SEM;
//   - leader (slice 0): writes its 2 output tiles' rows 0..7 into rows 8 (g % 4) .. of tile row g / 4 of the
//     [heads, 64] output (the SDPA output layout o_proj reads).
//
// common runtime args: [out_addr, core index table (SDA_NIDX words), noc (x << 8 | y) per core (2 per word)]

// Size-optimised: the binaries are written to every core range of the (fused QKV) program at each launch.
#pragma GCC optimize("Os")

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sda_common.hpp"

using namespace sda;

namespace {
inline uint32_t core_index(uint32_t base) {
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * SDA_GX + get_absolute_logical_x();
    return (get_common_arg_val<uint32_t>(base + c / 4) >> (8 * (c % 4))) & 0xFF;
}
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t window = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t min_chunks = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr uint32_t tile = 2048, tile32 = 4096;
    const auto s_out = TensorAccessor(out_args, out_addr, tile);

    const uint32_t ci = core_index(1);
    const uint32_t g = ci / S, s = ci % S;

    cb_wait_front(CB_POS, 1);
    invalidate_l1_cache();
    const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_read_ptr(CB_POS))[0];
    cb_pop_front(CB_POS, 1);
    const Split sp = split(p, window, S, min_chunks, s);
    const uint32_t info = info_word(sp, p, s);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, info);
    if (info == 0) {
        return;
    }
    // NOC coordinates of kv head g's slice t
    auto noc_of = [&](uint32_t t, uint32_t& x, uint32_t& y) {
        const uint32_t c = g * S + t;
        const uint32_t nw = get_common_arg_val<uint32_t>(1 + SDA_NIDX + c / 2) >> (16 * (c % 2));
        x = (nw >> 8) & 0xFF;
        y = nw & 0xFF;
    };

    // the row max to every active slice of the kv head (8 words, compacted at the end of the m tile)
    {
        DeviceZoneScopedN("sda-wr-m");
        cb_wait_front(CB_MOUT, 1);
        invalidate_l1_cache();
        const uint32_t mt = get_read_ptr(CB_MOUT);
        volatile tt_l1_ptr uint32_t* mw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mt);
        const uint32_t packed = mt + tile32 - 32;
        volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed);
        for (uint32_t r = 0; r < 8; ++r) {
            pw[r] = mw[r * 16];  // column 0 of row r (face 0)
        }
        const uint32_t slot = get_write_ptr(CB_MG) + 32 * s;  // CB_MG is never pushed: its base on every core
        for (uint32_t t = 0; t < sp.active; ++t) {
            uint32_t x, y;
            noc_of(t, x, y);
            noc_async_write(packed, get_noc_addr(x, y, slot), 32);
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < sp.active; ++t) {
            uint32_t x, y;
            noc_of(t, x, y);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SDA_M_SEM)), 1);
        }
        cb_pop_front(CB_MOUT, 1);
    }

    // the scaled state into the leader's pack tile s / 4, slot s % 4
    {
        uint32_t lx, ly;
        noc_of(0, lx, ly);
        const uint32_t k = s / 4, i = s % 4;
        const uint32_t opack = get_write_ptr(CB_OPACK) + 3 * k * tile32;  // the same address on every core
        {
            DeviceZoneScopedN("sda-wr-partwait");
            cb_wait_front(CB_PART, 3);
        }
        const uint32_t src = get_read_ptr(CB_PART);
        for (uint32_t t = 0; t < 3; ++t) {
            for (uint32_t f = 0; f < (t < 2 ? 2u : 1u); ++f) {  // L: column 0 is enough (face 0)
                noc_async_write(
                    src + t * tile32 + rows8_32(0, f), get_noc_addr(lx, ly, opack + t * tile32 + rows8_32(i, f)), 512);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(SDA_P_SEM)), 1);
        cb_pop_front(CB_PART, 3);
    }

    if (s == 0) {
        // out rows 0..7 -> heads 8 g .. 8 g + 7
        cb_wait_front(CB_OUT, 2);
        const uint32_t o = get_read_ptr(CB_OUT);
        for (uint32_t d = 0; d < 2; ++d) {
            const uint64_t dst = s_out.get_noc_addr(2 * (g / 4) + d);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(o + d * tile + rows8(0, f), dst + rows8(g % 4, f), 256);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 2);
    }
    noc_async_atomic_barrier();
}
