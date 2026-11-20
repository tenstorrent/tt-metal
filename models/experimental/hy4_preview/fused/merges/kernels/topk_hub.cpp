// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-chip top-k merge, hub form (fused/merges/topk_merge.py TopkMergeHub): the kernel of the chip's hub core,
// the one landing core of the merge's site. Per round it waits for the chip's user cores (topk_user.cpp) to have
// written their vectors into the collective sender's staging buffer, raises the round's ready bit (ONE sender op
// and one packet per route for all users), waits for the 32 chips' payloads and decides for every user:
//   a round before the last   the bucket whose count reaches topk -> a 16-byte record to the user core
//   the last round            the cut key, and from every chip's counts above and at it this chip's `keep`
//                             (topk_merge.hpp) -> the user's result record, and the chip's cut page to DRAM
// A slot of the landing is one chip's payload: user u's vector at u * vector bytes; vector[i] = entries with
// key >= base + (i << shift), i = 0 .. n; vector[n + 1] = the bucket that chip's user core expected.
// The search for the last bucket whose total reaches topk starts at this chip's own expectation and gallops:
// two totals when the chips' lists are alike, 2 log2(n) at worst.
//
// Compile-time args: rounds, bits of round 0..3, topk, users, cb_records.
// Runtime args: this chip's slot, state address, cut buffer address, vector bytes, per user the core's NoC
// x | y << 16 (8 words), then one hy4_collective::Use per round.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "topk_merge.hpp"

namespace {

constexpr uint32_t rounds = get_compile_time_arg_val(0);
constexpr uint32_t bits[4] = {
    get_compile_time_arg_val(1), get_compile_time_arg_val(2), get_compile_time_arg_val(3), get_compile_time_arg_val(4)};
constexpr uint32_t topk = get_compile_time_arg_val(5);
constexpr uint32_t users = get_compile_time_arg_val(6);
constexpr uint32_t cb_records = get_compile_time_arg_val(7);
constexpr uint32_t MAX_USERS = 8;
static_assert(rounds >= 1 && rounds <= 4 && users <= MAX_USERS);

constexpr uint32_t ST_RECORD = 16, ST_GO = 32, ST_STAMPS = 8, ST_STATUS = 1;

// One user's vectors in a landing: chip c's at `first + c * stride`.
struct Column {
    uint32_t first;
    uint32_t stride;
    uint32_t members;
    FORCE_INLINE uint32_t at(uint32_t chip, uint32_t i) const {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(first + chip * stride)[i];
    }
    inline uint32_t total(uint32_t i) const {
        uint32_t sum = 0;
        uint32_t address = first + 2 * i;
        for (uint32_t c = 0; c < members; ++c, address += stride) {
            sum += *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address);
        }
        return sum;
    }
    // The last i in [0, n) whose total reaches `want`, searched from `hint`; total(0) reaches it, total(n) does not.
    inline uint32_t last_reaching(uint32_t n, uint32_t want, uint32_t hint) const {
        uint32_t lo = 0, hi = n, step = 1;
        if (hint >= n) {
            hint = n - 1;
        }
        if (hint == 0 || total(hint) >= want) {
            lo = hint;
            while (lo + step < n) {
                if (total(lo + step) >= want) {
                    lo += step;
                    step <<= 1;
                } else {
                    hi = lo + step;
                    break;
                }
            }
        } else {
            hi = hint;
            while (hi > step) {
                if (total(hi - step) < want) {
                    hi -= step;
                    step <<= 1;
                } else {
                    lo = hi - step;
                    break;
                }
            }
        }
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) >> 1;
            if (total(mid) >= want) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

}  // namespace

void kernel_main() {
#ifdef HY4_MERGE_NOP
    return;
#else
    size_t arg = 0;
    const uint32_t chip = get_arg_val<uint32_t>(arg++);
    const uint32_t state = get_arg_val<uint32_t>(arg++);
    const uint32_t cuts_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vector_bytes = get_arg_val<uint32_t>(arg++);
    uint32_t user_xy[MAX_USERS];
    for (uint32_t u = 0; u < MAX_USERS; ++u) {
        user_xy[u] = get_arg_val<uint32_t>(arg++);
    }
    hy4_collective::Use use[4];
    for (uint32_t r = 0; r < rounds; ++r) {
        use[r] = hy4_collective::read_use(arg);
    }

    volatile tt_l1_ptr uint32_t* st = hy4_collective::word(state);
    volatile tt_l1_ptr uint32_t* records = hy4_collective::word(get_write_ptr(cb_records));  // 16 bytes per user
    st[ST_STAMPS + 0] = hy4_collective::now();
    hy4_collective::stamp(use[0].ctrl, hy4_collective::STAMP_START);
    const uint32_t sx = use[0].sender_xy & 0xFFFF;
    const uint32_t sy = use[0].sender_xy >> 16;
    uint32_t base[MAX_USERS], flags[MAX_USERS];
    for (uint32_t u = 0; u < MAX_USERS; ++u) {
        base[u] = 0;
        flags[u] = 0;
    }
    uint32_t shift = 16, status = 0;

    for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t n = 1u << bits[r];
        const bool last = r + 1 == rounds;
        shift -= bits[r];

        // The users' vectors of this round are in the sender's staging buffer.
        WAYPOINT("THDW");
        uint32_t arrived = 0;
        if (hy4_collective::wait_min(state + 4 * ST_GO, users, arrived)) {
            noc_semaphore_inc(get_noc_addr(state + 4 * ST_GO), 0u - users);
        } else {
            status |= hy4_topk::FLAG_TIMEOUT;
        }
        // collective.hpp hand_over without the copy: the op's ready bit of this round's parity
        noc_semaphore_inc(
            get_noc_addr(sx, sy, get_semaphore(use[r].ready_id)),
            1u << (2 * use[r].op + hy4_collective::parity(use[r])));
        noc_async_atomic_barrier();
        if (r < 2) {
            st[ST_STAMPS + 1 + 3 * r] = hy4_collective::now();
        }
        uint32_t parity = 0;
        if (!hy4_collective::wait_landed(use[r], parity)) {
            status |= hy4_topk::FLAG_TIMEOUT;
        }
        if (r < 2) {
            st[ST_STAMPS + 2 + 3 * r] = hy4_collective::now();
        }
        invalidate_l1_cache();

        WAYPOINT("THDC");
        const uint32_t landing = use[r].landing + parity * use[r].parity_stride;
        for (uint32_t u = 0; u < users; ++u) {
            const Column column{landing + u * vector_bytes, use[r].slot_stride, use[r].members};
            volatile tt_l1_ptr uint32_t* record = records + 4 * u;
            if (r == 0 && column.total(0) < topk) {
                flags[u] |= hy4_topk::FLAG_ALL;
            }
            const bool all = (flags[u] & hy4_topk::FLAG_ALL) != 0;
            const uint32_t i = all ? 0 : column.last_reaching(n, topk, column.at(chip, n + 1));
            if (!last) {
                base[u] += i << shift;
                record[0] = i;
                record[1] = flags[u];
                record[2] = 0;
                record[3] = 0;
                continue;
            }
            uint32_t keep = column.at(chip, 0), key = 0, above = 0, before = 0;  // everything seen, when all is kept
            if (!all) {
                for (uint32_t c = 0; c < column.members; ++c) {
                    const uint32_t a = column.at(c, i + 1);
                    above += a;
                    if (c < chip) {
                        before += column.at(c, i) - a;
                    }
                }
                const uint32_t own_above = column.at(chip, i + 1);
                const uint32_t own_tied = column.at(chip, i) - own_above;
                const uint32_t left = topk - above;
                keep = own_above + (before >= left ? 0 : (left - before < own_tied ? left - before : own_tied));
                key = base[u] + i;
            }
            record[0] = (1u << 31) | keep;
            record[1] = key | ((flags[u] | status) << 16);
            record[2] = above;
            record[3] = before;
        }
        if (r < 2) {
            st[ST_STAMPS + 3 + 3 * r] = hy4_collective::now();
        }

        // The records to the user cores: the bucket (word 16 of their state tile) or the result (word 0).
        WAYPOINT("THRW");
        const uint32_t target = state + 4 * (last ? 0 : ST_RECORD);
        for (uint32_t u = 0; u < users; ++u) {
            noc_async_write(
                reinterpret_cast<uint32_t>(records + 4 * u),
                get_noc_addr(user_xy[u] & 0xFFFF, user_xy[u] >> 16, target),
                16);
        }
        noc_async_write_barrier();
        for (uint32_t u = 0; u < users; ++u) {
            noc_semaphore_inc(get_noc_addr(user_xy[u] & 0xFFFF, user_xy[u] >> 16, state + 4 * ST_GO), 1);
        }
        noc_async_atomic_barrier();
        hy4_collective::finish(use[r]);
    }

    // The chip's cut page for the scan's cut program: the users' cut words.
    for (uint32_t u = 0; u < users; ++u) {  // ascending: word u is read (as word 4 k, k <= u) before it is written
        records[u] = records[4 * u];
    }
    for (uint32_t u = users; u < 16; ++u) {
        records[u] = 0;
    }
    noc_async_write(get_write_ptr(cb_records), get_noc_addr_from_bank_id<true>(0, cuts_addr), 64);
    noc_async_write_barrier();
    st[ST_STATUS] = status;
    st[ST_STAMPS + 7] = hy4_collective::now();
    WAYPOINT("THDN");
#endif
}
