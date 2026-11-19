// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14 rev 40 (MkMla DSA with token lanes, t > 4; tt/mk_mla.py att_lanes): the top-512 radix-select COORDINATOR
// of one token lane -- kernels/mla/ix_dm0.cpp's I5 loop on a core of its own, so the LANES lanes select concurrently.
// Lane l takes tokens l, l + LANES, ...; per sparse token, per round: wait for the C (lane-local) group-0 histograms
// (H_SEM), sum them, pick the digit holding the k-th largest key, send (prefix, k_remaining) to the lane's owners'
// mailboxes and increment their SEL_SEM; the tie round turns the C counts of keys == T into quotas in chunk order.
// BRISC. Compile-time args: H_SEM, SEL_SEM, C (chunk cores per token lane), NOWN (the lane's owners, every head group),
//   then the TensorAccessorArgs of cur_pos.
// Runtime args: cur_pos address, T, the histogram tensor's (local) address, the owners' mailbox address, the lane, then
//   NOWN owner NoC (x, y) pairs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"

void kernel_main() {
    constexpr uint32_t H_SEM = get_compile_time_arg_val(0);
    constexpr uint32_t SEL_SEM = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t NOWN = get_compile_time_arg_val(3);
    constexpr auto p_args = TensorAccessorArgs<4>();
    static_assert(C <= mladsa::MAX_CHUNKS, "chunks");
    static_assert(mladsa::T_MAX * 4 <= mladsa::POS_BYTES, "cur_pos scratch");
    const auto cur = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const uint32_t t_rows = get_arg_val<uint32_t>(1);
    const uint32_t hist_addr = get_arg_val<uint32_t>(2);
    const uint32_t mbox = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    constexpr uint32_t OWN0 = 5;
    // scratch past the C histogram slots of this lane: cur_pos [0, POS_BYTES), the message, the quotas. Rev 43: rounded
    // up to a 64-B-aligned ABSOLUTE address -- cur_pos comes from DRAM, and a Blackhole DRAM read lands its
    // 64-B-aligned-down bytes otherwise (mla_common.hpp); the L1 tensor itself is only 16-B aligned (the host allocates
    // 64 B more)
    const uint32_t scr = (hist_addr + C * mladsa::HIST_SLOT_BYTES + 63) & ~63u;
    constexpr uint32_t MSG_OFF = mladsa::POS_BYTES, QUOTA_L1 = mladsa::POS_BYTES + 64;
    static_assert(
        QUOTA_L1 + ((C * 4 + 15) & ~15u) <= 512, "the coordinator's 512-B scratch (mk_mla.py _coord hist words)");
    ASSERT((scr & 63) == 0);
    noc_async_read_page(0, cur, scr);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* pos_l1 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
    volatile tt_l1_ptr uint32_t* msg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + MSG_OFF);
    volatile tt_l1_ptr uint32_t* quota = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + QUOTA_L1);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_addr);
    // src [scr + off, + bytes) -> every owner of the lane at mbox + dst_off, then SEL_SEM += 1 on each
    auto broadcast = [&](uint32_t src, uint32_t dst_off, uint32_t bytes) {
        invalidate_l1_cache();  // the RISC's stores land before the NoC reads them (mla_common.hpp)
        for (uint32_t k = 0; k < NOWN; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_async_write(src, get_noc_addr(x, yy, mbox + dst_off), bytes);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < NOWN; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_semaphore_inc(get_noc_addr(x, yy, get_semaphore(SEL_SEM)), 1);
        }
        noc_async_atomic_barrier();
    };
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = pos_l1[t];
        if (!mladsa::sparse(pos < 0 ? 0u : static_cast<uint32_t>(pos))) {
            continue;
        }
        uint32_t prefix = 0, k_rem = mladsa::TOPK;
        for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
            const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
            mk::wait_handoff(H_SEM, C);
            invalidate_l1_cache();
            uint32_t h[mladsa::BINS];
#if defined(MLA_DSA_HSUM_OLD)  // GLM53_MLA_DSA_HSUM=0: the rev <= 51 sum
            for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                h[b] = 0;
            }
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                    h[b] += hist[c * (mladsa::HIST_SLOT_BYTES / 4) + b];
                }
            }
#else
            mladsa::sum_hist(hist, C, h);  // rev 52: register accumulators
#endif
            uint32_t d = mladsa::BINS - 1, above = 0;
            while (d > 0 && above + h[d] < k_rem) {
                above += h[d];
                --d;
            }
            k_rem -= above;
            prefix |= d << shift;
            msg[0] = prefix;
            msg[1] = k_rem;
            broadcast(scr + MSG_OFF, (t * mladsa::MSGS + r) * mladsa::MSG_BYTES, mladsa::MSG_BYTES);
        }
        mk::wait_handoff(H_SEM, C);  // tie round: k_rem keys == prefix, taken in chunk order
        invalidate_l1_cache();
        uint32_t rem = k_rem;
        for (uint32_t c = 0; c < C; ++c) {
            const uint32_t n = hist[c * (mladsa::HIST_SLOT_BYTES / 4)];
            const uint32_t q = n < rem ? n : rem;
            quota[c] = q;
            rem -= q;
        }
        broadcast(scr + QUOTA_L1, mladsa::QUOTA_OFF + t * mladsa::MAX_CHUNKS * 4, (C * 4 + 15) & ~15u);
    }
}
