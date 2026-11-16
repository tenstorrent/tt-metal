// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a head-core writer (BRISC, NoC 0; see k2a_common.hpp). Per layer: the prepped Q of q head h (32 launch rows) into
// the DRAM Q staging in the unit cores' packed layout (Pack: head h's rows of a packed tile are k M contiguous rows, so
// one write per face and d), the K / V role's BF8 tiles into the K / V staging, then +1 READY on every unit core.
// Staging is single-buffered: layer l + 1's prep follows the dense half, which follows every root's O(l), which
// follows every read of staging(l).
//
// Runtime args: 0 h. Common: 0 Q staging (DRAM, 2 KB pages: (g * tiles + T) * Dt + d), 1 K staging, 2 V staging
// (DRAM, BF8 pages: g * Dt + d), 3 unit core xy list offset (x | y << 16 each)

#include "api/dataflow/dataflow_api.h"
#include "k2a_common.hpp"

using namespace k2a;

namespace {

template <bool full>
FORCE_INLINE void stage_q(uint32_t h, uint32_t src, uint32_t q_addr) {
    using P = Pack<full>;
    constexpr uint32_t Dt = full ? kDtf : kDts;
    const uint32_t g = full ? 0 : h / 2, hl = full ? h : h % 2;
    constexpr uint32_t nr = P::k * M;  // head hl's rows in a packed tile (4, 8 or 16: inside one face half)
    const InterleavedAddrGen<true> qg{.bank_base_address = q_addr, .page_size = kBf16};
    for (uint32_t j = 0; j < P::groups; j++) {  // user groups (D16: the last may be part-filled: its rows are padding)
        const uint32_t T = j * P::Tu + hl / P::Ht;
        const uint32_t s = j * nr;                     // source rows (launch rows of users j k ..)
        const uint32_t a = (hl % P::Ht) * nr;          // destination rows in the packed tile
        for (uint32_t d = 0; d < Dt; d++) {
            const uint32_t page = (g * P::tiles + T) * Dt + d;
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(src + d * kBf16 + (s / 16) * 1024 + f * 512 + (s % 16) * 32,
                                qg.get_noc_addr(page, (a / 16) * 1024 + f * 512 + (a % 16) * 32), 32 * nr);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1), v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t uxy = get_common_arg_val<uint32_t>(3);
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts;
        const uint32_t role = full ? (h == 0 ? 1 : (h == 1 ? 2 : 0)) : (h % 2 == 0 ? 1 : 2);
        cb_wait_front(hcb_qo, 16);
        if (full) {
            stage_q<true>(h, get_read_ptr(hcb_qo), q_addr);
        } else {
            stage_q<false>(h, get_read_ptr(hcb_qo), q_addr);
        }
        if (role != 0) {
            cb_wait_front(hcb_kv8, 16);
            const InterleavedAddrGen<true> kvg{.bank_base_address = role == 1 ? k_addr : v_addr, .page_size = kBf8};
            const uint32_t g = full ? 0 : h / 2;
            const uint32_t src = get_read_ptr(hcb_kv8);
            for (uint32_t d = 0; d < Dt; d++) {
                noc_async_write(src + d * kBf8, kvg.get_noc_addr(g * Dt + d), kBf8);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(hcb_qo, 16);
        if (role != 0) {
            cb_pop_front(hcb_kv8, 16);
        }
        for (uint32_t c = 0; c < kNumUnits; c++) {
            const uint32_t xy = get_common_arg_val<uint32_t>(uxy + c);
            noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(U_READY)), 1);
        }
        noc_async_atomic_barrier();
        K2A_MARK(l, 1, kTlHead);
    }
}
